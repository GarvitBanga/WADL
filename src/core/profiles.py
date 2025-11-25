from typing import List, Optional, Tuple, Dict
import asyncio
import httpx
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import random
from sqlalchemy.orm import Session
from src.db.models import Candidate, CandidateEmbedding
from src.core.llm_client import LLMClient, EmbeddingClient
from src.config import settings

PROFILE_TTL_DAYS = 30

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

PROFILE_ENRICH_PROMPT = """
You are analyzing a LinkedIn profile snippet from a search result. The text below is typically a title and short description from Google/SerpAPI.

Extract ALL available information and make reasonable inferences. Even with limited text, you can infer:
- Skills from technologies/tools mentioned
- Experience level from titles and descriptions
- Domains from industry keywords
- Current role from the title format

Return a JSON object with these fields:

{{
  "headline": string (extract from title, e.g. "Software Engineer at Company"),
  "summary": string (expand the snippet into 2-4 sentences about their background),
  "years_experience": float (estimate from title seniority and description - "Senior" = 5-8, "Lead" = 7-10, "Director" = 10+, "Engineer" = 2-5),
  "skills": string[] (extract ALL technologies, tools, frameworks, languages mentioned - be aggressive, aim for 10-20 skills),
  "domains": string[] (extract industry domains: "healthcare", "telehealth", "EHR", "HIPAA", "microservices", "cloud", "backend", "frontend", etc.),
  "roles": [
    {{
      "title": string (extract from title or description),
      "company": string (extract from title or description),
      "location": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "is_current": boolean (true if title suggests current role),
      "years": float | null,
      "description": string (expand snippet into role description)
    }}
  ],
  "education": [
    {{
      "school": string | null,
      "degree": string | null,
      "field": string | null,
      "start_year": int | null,
      "end_year": int | null
    }}
  ]
}}

CRITICAL INSTRUCTIONS:
- Parse the title format: "Name - Title at Company | LinkedIn" or similar
- Extract every technology/tool mentioned (Python, JavaScript, AWS, PostgreSQL, etc.)
- Infer experience years from title keywords: "Senior" = 5-8, "Lead" = 7-10, "Principal" = 10+, "Director" = 10+
- Extract domains from keywords: healthcare, behavioral health, residential care, HIPAA, EHR, cloud, microservices, etc.
- If snippet mentions "X years" or "X+ years", use that for years_experience
- Create at least one role entry from the title/description
- Be aggressive with skill extraction - if someone is a "Software Engineer" with "Python" mentioned, infer common skills like "Python", "Software Development", "Backend Development", etc.

Profile text:
---
{profile_text}
---
"""

def build_profile_text_for_llm(
    html: Optional[str],
    title: str,
    snippet: str,
) -> str:
    parts = []
    
    combined_text = f"{title}\n\n{snippet}".strip()
    if combined_text:
        parts.append(combined_text)

    if html and len(html) > 1000:
        soup = BeautifulSoup(html, "html.parser")
        
        extracted_text = []
        
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                import json
                data = json.loads(script.string)
                if isinstance(data, dict):
                    extracted_text.append(json.dumps(data, indent=2))
            except Exception:
                pass
        
        for script in soup.find_all("script"):
            script_text = script.string
            if script_text and ("experience" in script_text.lower() or "education" in script_text.lower() or "skills" in script_text.lower()):
                if len(script_text) < 5000:
                    extracted_text.append(script_text[:2000])
        
        body = soup.find("body")
        if body:
            for elem in body.find_all(["div", "section", "article", "main"]):
                text = elem.get_text(separator=" ", strip=True)
                if len(text) > 50 and any(keyword in text.lower() for keyword in ["experience", "education", "skills", "work", "company", "university", "degree", "engineer", "developer", "manager"]):
                    extracted_text.append(text[:1000])
        
        if not extracted_text:
            text = soup.get_text(separator="\n")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) > 10]
            clean = "\n".join(lines[:200])
            extracted_text.append(clean)
        
        parts.extend(extracted_text)

    return "\n\n".join(p for p in parts if p and len(p.strip()) > 0)

async def enrich_candidate_with_llm(llm: LLMClient, profile_text: str) -> dict:
    prompt = PROFILE_ENRICH_PROMPT.format(profile_text=profile_text[:15000])
    data = await llm.complete_json(prompt)
    return data

class ProfileFetcher:
    def __init__(self, llm: LLMClient, embedder: EmbeddingClient, max_concurrent: int = 3, use_browser: bool = False, request_delay: float = 2.0):
        self.llm = llm
        self.embedder = embedder
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.use_browser = use_browser
        self._browser = None
        self.request_delay = request_delay
        self._last_request_time = 0
        self._playwright = None
        self._fetch_failures = 0
        self._max_failures = 5

    async def _get_browser(self):
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
                import os
                import logging
                logger = logging.getLogger(__name__)
                
                logger.info("Starting Playwright...")
                self._playwright = await async_playwright().start()
                
                user_data_dir = os.path.expanduser("~/.wadl_browser_data")
                os.makedirs(user_data_dir, exist_ok=True)
                
                logger.info(f"Launching Chromium (headless=False, user_data={user_data_dir})...")
                self._browser = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=False,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--no-sandbox',
                    ],
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                )
                
                logger.info("Browser launched! Opening Google to initialize session...")
                page = await self._browser.new_page()
                await page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(2)
                await page.close()
                logger.info("Browser ready!")
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to launch browser: {e}", exc_info=True)
                return None
        return self._browser

    async def _http_get_with_browser(self, url: str, retry: int = 0) -> Optional[str]:
        import logging
        logger = logging.getLogger(__name__)
        
        if self._fetch_failures >= self._max_failures:
            logger.warning(f"Too many failures ({self._fetch_failures}), skipping browser fetch")
            return None
        
        context = await self._get_browser()
        if not context:
            return None
        
        page = None
        try:
            page = await context.new_page()
            
            await asyncio.sleep(random.uniform(1, 2))
            
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            await asyncio.sleep(random.uniform(2, 4))
            
            for _ in range(3):
                await page.evaluate("window.scrollBy(0, window.innerHeight / 2)")
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(random.uniform(1, 2))
            
            close_selectors = [
                'button[aria-label="Dismiss"]',
                'button[data-tracking-control-name="public_profile_contextual-sign-in-modal_modal_dismiss"]',
                'button.contextual-sign-in-modal__modal-dismiss-btn',
                'button.modal__dismiss',
                'button[aria-label="Close"]',
                'button.artdeco-modal__dismiss',
                '[data-tracking-control-name*="dismiss"]',
                '.contextual-sign-in-modal button',
                '.modal__dismiss',
                'button.dismiss-icon',
            ]
            
            for selector in close_selectors:
                try:
                    close_btn = await page.query_selector(selector)
                    if close_btn:
                        logger.info(f"Found popup close button: {selector}")
                        await close_btn.click()
                        await asyncio.sleep(1)
                        break
                except:
                    continue
            
            await page.keyboard.press("Escape")
            await asyncio.sleep(1)
            
            for _ in range(2):
                await page.evaluate("window.scrollBy(0, 400)")
                await asyncio.sleep(0.5)
            
            html = await page.content()
            
            if html and len(html) > 5000:
                page_text = await page.inner_text("body")
                if "experience" in page_text.lower() or "about" in page_text.lower():
                    self._fetch_failures = max(0, self._fetch_failures - 1)
                    logger.info(f"Browser fetch success for {url}: {len(html)} chars (has profile content)")
                    return html
                elif len(html) > 20000:
                    logger.info(f"Browser fetch got {len(html)} chars (may have profile)")
                    return html
            
            logger.warning(f"HTML too short or no profile content for {url}: {len(html) if html else 0} chars")
            self._fetch_failures += 1
            return None
            
        except Exception as e:
            logger.error(f"Browser error for {url}: {e}")
            self._fetch_failures += 1
            if retry < 2:
                await asyncio.sleep(random.uniform(3, 6))
                return await self._http_get_with_browser(url, retry + 1)
            return None
        finally:
            if page:
                try:
                    await page.close()
                except:
                    pass

    async def _wait_for_captcha(self, page, logger) -> bool:
        captcha_indicators = [
            "unusual traffic",
            "not a robot",
            "verify you're human",
            "captcha",
            "recaptcha",
            "prove you're not a robot"
        ]
        
        for attempt in range(30):
            try:
                page_text = await page.inner_text("body")
                page_lower = page_text.lower()
                
                if any(indicator in page_lower for indicator in captcha_indicators):
                    if attempt == 0:
                        logger.warning("Google CAPTCHA detected! Please solve it in the browser window...")
                    await asyncio.sleep(2)
                    continue
                
                search_box = await page.query_selector('textarea[name="q"], input[name="q"]')
                if search_box:
                    logger.info("CAPTCHA cleared or not present, continuing...")
                    return True
                    
                await asyncio.sleep(1)
            except:
                await asyncio.sleep(1)
        
        logger.error("Timed out waiting for CAPTCHA to be solved")
        return False

    async def _http_get_via_brightdata_api_batch(self, urls: List[str]) -> Dict[str, Optional[str]]:
        try:
            from src.config import settings
            import httpx
            import logging
            import json
            logger = logging.getLogger(__name__)
            
            if not settings.brightdata_api_key:
                return {url: None for url in urls}
            
            dataset_id = settings.brightdata_dataset_id or "gd_l1viktl72bvl7bjuj0"
            
            logger.info(f"Using Bright Data LinkedIn API (batch) for {len(urls)} profiles")
            
            api_url = f"https://api.brightdata.com/datasets/v3/scrape?dataset_id={dataset_id}&notify=false&include_errors=true"
            headers = {
                "Authorization": f"Bearer {settings.brightdata_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": [{"url": url} for url in urls]
            }
            
            results = {}
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(api_url, json=payload, headers=headers)
                
                if resp.status_code == 202:
                    data = resp.json()
                    snapshot_id = data.get("snapshot_id")
                    if snapshot_id:
                        logger.info(f"Bright Data batch request in progress (202), snapshot_id: {snapshot_id}")
                        
                        monitor_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}"
                        max_polls = 20
                        poll_interval = 5
                        
                        data = None
                        for poll in range(max_polls):
                            await asyncio.sleep(poll_interval)
                            monitor_resp = await client.get(monitor_url, headers=headers)
                            
                            if monitor_resp.status_code == 200:
                                try:
                                    data = monitor_resp.json()
                                    logger.info(f"Snapshot ready after {(poll + 1) * poll_interval} seconds")
                                    break
                                except json.JSONDecodeError:
                                    text = monitor_resp.text.strip()
                                    if text:
                                        parsed = []
                                        for line in text.split('\n'):
                                            line = line.strip()
                                            if line:
                                                try:
                                                    parsed.append(json.loads(line))
                                                except:
                                                    pass
                                        if parsed:
                                            data = parsed
                                            logger.info(f"Snapshot ready: {len(parsed)} objects")
                                            break
                                    continue
                            elif monitor_resp.status_code == 202:
                                if (poll + 1) % 3 == 0:
                                    logger.info(f"Still processing... ({poll + 1}/{max_polls} polls)")
                                continue
                            else:
                                logger.warning(f"Monitor endpoint failed: {monitor_resp.status_code}")
                                for url in urls:
                                    results[url] = None
                                return results
                        
                        if data is None:
                            for url in urls:
                                results[url] = None
                            return results
                
                if resp.status_code == 200:
                    if data is None:
                        try:
                            data = resp.json()
                        except json.JSONDecodeError:
                            text = resp.text.strip()
                            if text:
                                parsed = []
                                for line in text.split('\n'):
                                    line = line.strip()
                                    if line:
                                        try:
                                            parsed.append(json.loads(line))
                                        except:
                                            pass
                                if parsed:
                                    data = parsed
                            if not data:
                                for url in urls:
                                    results[url] = None
                                return results
                
                if not isinstance(data, list):
                    if isinstance(data, dict):
                        data = data.get("data") or data.get("results")
                    if not isinstance(data, list):
                        for url in urls:
                            results[url] = None
                        return results
                
                for i, profile_data in enumerate(data):
                    url = urls[i] if i < len(urls) else None
                    if not url or not isinstance(profile_data, dict):
                        if url:
                            results[url] = None
                        continue
                    
                    if "warning" in profile_data or "warning_code" in profile_data:
                        results[url] = None
                    elif "id" in profile_data or "name" in profile_data:
                        results[url] = json.dumps(profile_data, indent=2)
                    else:
                        results[url] = None
                else:
                    logger.warning(f"Bright Data API batch failed: {resp.status_code} - {resp.text[:200]}")
                    for url in urls:
                        results[url] = None
            
            success_count = sum(1 for v in results.values() if v is not None)
            logger.info(f"Bright Data API batch: {success_count}/{len(urls)} profiles fetched successfully")
            return results
                    
        except Exception as e:
            logger.warning(f"Bright Data API batch error: {e}")
            return {url: None for url in urls}
    
    async def _http_get_via_brightdata_api(self, url: str) -> Optional[str]:
        try:
            from src.config import settings
            import httpx
            import logging
            import json
            logger = logging.getLogger(__name__)
            
            if not settings.brightdata_api_key:
                return None
            
            dataset_id = settings.brightdata_dataset_id or "gd_l1viktl72bvl7bjuj0"
            
            logger.info(f"Using Bright Data LinkedIn API for: {url}")
            
            api_url = f"https://api.brightdata.com/datasets/v3/scrape?dataset_id={dataset_id}&notify=false&include_errors=true"
            headers = {
                "Authorization": f"Bearer {settings.brightdata_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": [{"url": url}]
            }
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(api_url, json=payload, headers=headers)
                
                if resp.status_code == 202:
                    data = resp.json()
                    snapshot_id = data.get("snapshot_id")
                    if snapshot_id:
                        logger.info(f"Bright Data API request in progress (202), snapshot_id: {snapshot_id}")
                        
                        monitor_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}"
                        max_polls = 20
                        poll_interval = 5
                        
                        for poll in range(max_polls):
                            await asyncio.sleep(poll_interval)
                            monitor_resp = await client.get(monitor_url, headers=headers)
                            
                            if monitor_resp.status_code == 200:
                                data = monitor_resp.json()
                                logger.info(f"Snapshot ready after {(poll + 1) * poll_interval} seconds")
                                break
                            elif monitor_resp.status_code == 202:
                                if (poll + 1) % 3 == 0:
                                    logger.info(f"Still processing... ({poll + 1}/{max_polls} polls)")
                                continue
                            else:
                                logger.warning(f"Monitor endpoint failed: {monitor_resp.status_code}")
                                return None
                        else:
                            logger.warning("Snapshot polling timeout")
                            return None
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    if isinstance(data, list) and len(data) > 0:
                        profile_data = data[0]
                        if isinstance(profile_data, dict):
                            if "warning" in profile_data or "warning_code" in profile_data:
                                warning = profile_data.get("warning", "Unknown warning")
                                logger.warning(f"Bright Data API warning: {warning}")
                                return None
                            
                            if "id" in profile_data or "name" in profile_data:
                                profile_text = json.dumps(profile_data, indent=2)
                                logger.info(f"Bright Data API success: {len(profile_text)} chars")
                                return profile_text
                    
                    elif isinstance(data, dict):
                        if "warning" in data or "warning_code" in data:
                            warning = data.get("warning", "Unknown warning")
                            logger.warning(f"Bright Data API warning: {warning}")
                            return None
                        
                        if "id" in data or "name" in data or "about" in data:
                            profile_text = json.dumps(data, indent=2)
                            logger.info(f"Bright Data API success: {len(profile_text)} chars")
                            return profile_text
                        
                        if "data" in data:
                            results = data["data"]
                            if isinstance(results, list) and len(results) > 0:
                                profile_data = results[0]
                                if isinstance(profile_data, dict):
                                    profile_text = json.dumps(profile_data, indent=2)
                                    logger.info(f"Bright Data API success: {len(profile_text)} chars")
                                    return profile_text
                        elif "results" in data:
                            results = data["results"]
                            if isinstance(results, list) and len(results) > 0:
                                profile_data = results[0]
                                if isinstance(profile_data, dict):
                                    profile_text = json.dumps(profile_data, indent=2)
                                    logger.info(f"Bright Data API success: {len(profile_text)} chars")
                                    return profile_text
                    
                    logger.warning(f"Bright Data API returned unexpected format: {str(data)[:500]}")
                    return None
                else:
                    logger.warning(f"Bright Data API failed: {resp.status_code} - {resp.text[:200]}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Bright Data API error: {e}")
            return None
    
    async def _http_get_via_staffspy(self, url: str) -> Optional[str]:
        try:
            from staffspy import LinkedInAccount
            import logging
            logger = logging.getLogger(__name__)
            
            user_id = url.split("/in/")[-1].split("/")[0].split("?")[0]
            if not user_id:
                return None
            
            logger.info(f"Using StaffSpy to fetch profile: {user_id}")
            
            import os
            session_file = os.path.expanduser("~/.wadl_linkedin_session.pkl")
            account = LinkedInAccount(
                session_file=session_file,
                log_level=0,
            )
            
            users = account.scrape_users(
                user_ids=[user_id],
                extra_profile_data=True
            )
            
            if users is not None and len(users) > 0:
                user = users.iloc[0]
                skills_list = user.get('skills', [])
                skills_str = ', '.join([str(s.get('name', '')) for s in skills_list]) if skills_list else ''
                profile_text = f"""
Name: {user.get('name', '')}
Position: {user.get('position', '')}
Company: {user.get('company', '')}
Location: {user.get('location', '')}
Bio: {user.get('bio', '')}
Skills: {skills_str}
Experience: {user.get('experiences', [])}
Education: {user.get('education', [])}
"""
                return profile_text
            
            return None
        except ImportError:
            raise
        except Exception as e:
            logger.warning(f"StaffSpy error: {e}")
            return None
    
    async def _http_get_via_google_search(self, url: str, name: str = None, title: str = None) -> Optional[str]:
        import logging
        logger = logging.getLogger(__name__)
        
        context = await self._get_browser()
        if not context:
            return None
        
        page = None
        
        try:
            if name and title:
                search_query = f'"{name}" "{title}" site:linkedin.com/in/'
            elif name:
                search_query = f'"{name}" site:linkedin.com/in/'
            else:
                linkedin_username = url.split("/in/")[-1].split("/")[0].split("?")[0]
                search_query = f'"{linkedin_username}" site:linkedin.com/in/'
            
            logger.info(f"Google search: {search_query}")
            
            page = await context.new_page()
            
            await page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(random.uniform(2, 3))
            
            if not await self._wait_for_captcha(page, logger):
                return None
            
            search_box = await page.query_selector('textarea[name="q"], input[name="q"]')
            if not search_box:
                logger.error("Could not find Google search box")
                return None
                
            await search_box.click()
            await asyncio.sleep(random.uniform(0.5, 1))
            
            for char in search_query:
                await search_box.type(char, delay=random.randint(30, 80))
                if random.random() < 0.1:
                    await asyncio.sleep(random.uniform(0.1, 0.3))
            
            await asyncio.sleep(random.uniform(0.5, 1.5))
            await page.keyboard.press("Enter")
            
            await asyncio.sleep(random.uniform(3, 5))
            
            if not await self._wait_for_captcha(page, logger):
                return None
            
            await asyncio.sleep(2)
            
            target_username = url.split("/in/")[-1].split("/")[0].split("?")[0].lower() if "/in/" in url else None
            
            all_links = await page.query_selector_all('a[href]')
            linkedin_links = []
            best_match = None
            
            for link in all_links:
                try:
                    href = await link.get_attribute("href")
                    if href and href.startswith("https://www.linkedin.com/in/"):
                        linkedin_links.append(href)
                        if target_username and target_username in href.lower():
                            best_match = href
                            logger.info(f"Found exact match: {href}")
                            break
                except:
                    continue
            
            linkedin_href = best_match or (linkedin_links[0] if linkedin_links else None)
            
            if linkedin_links:
                logger.info(f"Found {len(linkedin_links)} LinkedIn profiles, using: {linkedin_href}")
            else:
                logger.warning("No LinkedIn profile links found in search results")
            
            if linkedin_href:
                for attempt in range(3):
                    try:
                        if attempt > 0:
                            logger.info(f"Attempt {attempt+1}: Going back to search results...")
                            await page.go_back()
                            await asyncio.sleep(2)
                            
                            all_links = await page.query_selector_all('a[href]')
                            for link in all_links:
                                try:
                                    href = await link.get_attribute("href")
                                    if href and href == linkedin_href:
                                        logger.info(f"Found LinkedIn link again, clicking...")
                                        await link.click()
                                        await asyncio.sleep(random.uniform(3, 5))
                                        break
                                except:
                                    continue
                        else:
                            logger.info(f"Attempt {attempt+1}: Navigating to LinkedIn: {linkedin_href}")
                            await asyncio.sleep(random.uniform(1, 2))
                            await page.goto(linkedin_href, wait_until="domcontentloaded", timeout=30000)
                            await asyncio.sleep(random.uniform(3, 5))
                        
                        for _ in range(2):
                            await page.evaluate("window.scrollBy(0, 300)")
                            await asyncio.sleep(0.5)
                        
                        page_text = await page.inner_text("body")
                        has_signin_popup = any(x in page_text.lower() for x in ["sign in", "join now", "join linkedin"])
                        
                        if has_signin_popup:
                            logger.info("Sign-in popup detected, trying to close it...")
                            
                            close_selectors = [
                                'button[aria-label="Dismiss"]',
                                'button[aria-label="Close"]',
                                'button.modal__dismiss',
                                'button.artdeco-modal__dismiss',
                                '[data-tracking-control-name*="dismiss"]',
                            ]
                            
                            popup_closed = False
                            for selector in close_selectors:
                                try:
                                    close_btn = await page.query_selector(selector)
                                    if close_btn:
                                        logger.info(f"Clicking close button: {selector}")
                                        await close_btn.click()
                                        popup_closed = True
                                        await asyncio.sleep(1)
                                        break
                                except:
                                    continue
                            
                            if not popup_closed:
                                try:
                                    await page.keyboard.press("Escape")
                                    await asyncio.sleep(0.5)
                                except:
                                    pass
                            
                            await asyncio.sleep(1)
                            page_text = await page.inner_text("body")
                            still_has_popup = any(x in page_text.lower() for x in ["sign in", "join now", "join linkedin"])
                            
                            if still_has_popup and attempt < 2:
                                logger.info("Popup still present, going back to retry...")
                                continue
                        
                        html = await page.content()
                        if html and len(html) > 15000:
                            page_text = await page.inner_text("body")
                            if "experience" in page_text.lower() or "about" in page_text.lower():
                                logger.info(f"SUCCESS! Got {len(html)} chars with profile content")
                                return html
                        
                        logger.info(f"Attempt {attempt+1} got {len(html) if html else 0} chars, retrying...")
                        
                    except Exception as e:
                        logger.warning(f"Attempt {attempt+1} error: {e}")
                        try:
                            await page.go_back()
                            await asyncio.sleep(2)
                        except:
                            pass
                
                logger.warning("Could not get LinkedIn content after 3 attempts")
            else:
                logger.warning("No LinkedIn link found in Google results")
                    
            return None
            
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return None
        finally:
            if page:
                try:
                    await page.close()
                except:
                    pass

    async def _http_get(self, url: str, name: str = None, title: str = None) -> Optional[str]:
        import logging
        logger = logging.getLogger(__name__)
        
        if settings.skip_html_fetch:
            logger.info(f"Skipping HTML fetch for {url} (skip_html_fetch=True)")
            return None
        
        try:
            html = await self._http_get_via_brightdata_api(url)
            if html and len(html) > 100:
                logger.info(f"Bright Data API success: {len(html)} chars")
                return html
        except Exception as e:
            logger.debug(f"Bright Data API failed: {e}, trying other methods")
        
        if not settings.brightdata_api_key:
            try:
                html = await self._http_get_via_staffspy(url)
                if html and len(html) > 5000:
                    logger.info(f"StaffSpy success: {len(html)} chars")
                    return html
            except ImportError:
                logger.debug("StaffSpy not available, using other methods")
            except Exception as e:
                logger.debug(f"StaffSpy failed: {e}, trying other methods")
        
        if self.use_browser:
            logger.info(f"Using browser (Google search method) for {url}")
            html = await self._http_get_via_google_search(url, name=name, title=title)
            if html and len(html) > 5000:
                return html
            logger.warning("Browser method failed, trying direct HTTP with proxies...")
        
        logger.info(f"Fetching {url} via direct HTTP")
        
        now = asyncio.get_event_loop().time()
        time_since_last = now - self._last_request_time
        if time_since_last < self.request_delay:
            delay = self.request_delay - time_since_last
            logger.info(f"Rate limiting: waiting {delay:.1f}s before next request")
            await asyncio.sleep(delay)
        self._last_request_time = asyncio.get_event_loop().time()
        
        timeout = httpx.Timeout(15.0, connect=10.0)
        user_agent = random.choice(USER_AGENTS)
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }
        
        proxy_list = settings.get_proxy_list()
        if proxy_list:
            random.shuffle(proxy_list)
            proxy_attempts = proxy_list[:20]
            proxy_attempts.append(None)
        else:
            proxy_attempts = [None]
        
        for idx, proxy in enumerate(proxy_attempts):
            if idx > 0:
                delay = random.uniform(2, 4)
                logger.info(f"Waiting {delay:.1f}s before next proxy attempt...")
                await asyncio.sleep(delay)
            try:
                client_kwargs = {"timeout": timeout, "follow_redirects": True, "headers": headers}
                
                if proxy:
                    proxy_str = proxy.strip()
                    if not proxy_str.startswith(("http://", "https://", "socks5://")):
                        proxy_str = f"http://{proxy_str}"
                    
                    client_kwargs["proxy"] = proxy_str
                    logger.info(f"Trying proxy: {proxy_str.split('@')[-1] if '@' in proxy_str else proxy_str[:50]}...")
                else:
                    logger.info("Trying direct connection (no proxy)")
                
                async with httpx.AsyncClient(**client_kwargs) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        logger.info(f"HTTP success: {len(resp.text)} chars")
                        return resp.text
                    elif resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", self.request_delay * 2))
                        logger.warning(f"Rate limited (429), waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        if proxy:
                            continue
                        return None
                    elif resp.status_code == 999:
                        logger.warning(f"LinkedIn blocked (999) - {'trying next proxy' if proxy and len(proxy_attempts) > proxy_attempts.index(proxy) + 1 else 'all proxies failed'}")
                        if proxy:
                            continue
                        return None
                    else:
                        logger.warning(f"HTTP failed: status {resp.status_code}")
                        if proxy:
                            continue
                        return None
            except httpx.ProxyError as e:
                logger.warning(f"Proxy error: {e}")
                if proxy:
                    continue
            except Exception as e:
                logger.warning(f"HTTP exception: {e}")
                if proxy:
                    continue
        
        logger.error(f"All fetch methods failed for {url}")
        return None

    async def _http_get_via_scraperapi(self, url: str) -> Optional[str]:
        from src.config import settings
        import urllib.parse
        import logging
        logger = logging.getLogger(__name__)
        
        encoded_url = urllib.parse.quote(url, safe='')
        scraperapi_url = f"http://api.scraperapi.com?api_key={settings.scraperapi_key}&url={encoded_url}&render=true"
        timeout = httpx.Timeout(30.0, connect=15.0)
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                logger.info(f"ScraperAPI request: {scraperapi_url[:100]}...")
                resp = await client.get(scraperapi_url, headers=headers)
                logger.info(f"ScraperAPI response: status {resp.status_code}, length {len(resp.text)}")
                if resp.status_code == 200:
                    return resp.text
                else:
                    logger.warning(f"ScraperAPI returned status {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                logger.error(f"ScraperAPI exception: {e}")
        return None

    async def close(self):
        if self._browser:
            await self._browser.close()
            self._browser = None

def parse_profile(
    url: str,
    html: Optional[str],
    search_title: str,
    search_snippet: str,
) -> Tuple[str, str, str]:
    title_text = search_title or ""

    if html:
        soup = BeautifulSoup(html, "html.parser")
        html_title = soup.find("title")
        if html_title:
            html_title_text = html_title.get_text(strip=True)
            if "linkedin" in html_title_text.lower():
                title_text = html_title_text

    parts = re.split(r"[-|–]", title_text)
    parts = [p.strip() for p in parts if p.strip()]

    name = "Unknown"
    current_title = ""
    current_company = ""

    if parts:
        name = parts[0]
    if len(parts) >= 2:
        current_title = parts[1]
    if len(parts) >= 3:
        current_company = parts[2]

    return name, current_title, current_company

async def fetch_and_parse_batch(
    fetcher: ProfileFetcher,
    session: Session,
    results: List[dict],
) -> List[Candidate]:
    import logging
    logger = logging.getLogger(__name__)
    
    candidates: List[Candidate] = []
    url_map = {r["url"]: r for r in results}
    urls = list(url_map.keys())
    
    existing = (
        session.query(Candidate)
        .filter(Candidate.profile_url.in_(urls))
        .all()
    )
    existing_by_url = {c.profile_url: c for c in existing}
    
    to_fetch: List[str] = []
    now = datetime.utcnow()
    for url in urls:
        cand = existing_by_url.get(url)
        if cand and cand.last_fetched_at and cand.last_fetched_at > now - timedelta(days=PROFILE_TTL_DAYS):
            candidates.append(cand)
        else:
            to_fetch.append(url)

    brightdata_results = {}
    if to_fetch and settings.brightdata_api_key:
        try:
            batch_size = 10
            for i in range(0, len(to_fetch), batch_size):
                batch_urls = to_fetch[i:i + batch_size]
                batch_results = await fetcher._http_get_via_brightdata_api_batch(batch_urls)
                brightdata_results.update(batch_results)
        except Exception:
            pass

    async def process_url(url: str) -> Optional[Candidate]:
        search_meta = url_map[url]
        title = search_meta.get("title") or ""
        snippet = search_meta.get("snippet") or ""
        
        name_from_title, title_from_title, _ = parse_profile(url, None, title, snippet)
        
        html = brightdata_results.get(url)
        if not html:
            html = await fetcher._http_get(url, name=name_from_title, title=title_from_title)

        name, current_title, current_company = parse_profile(url, html, title, snippet)

        if html:
            raw_text = f"{title}\n\n{snippet}\n\n{html[:5000]}"
        else:
            raw_text = f"{title}\n\n{snippet}"

        profile_text_for_llm = build_profile_text_for_llm(html, title, snippet)

        headline = None
        summary = None
        years_exp = None
        skills = []
        domains = []
        roles = []
        education = []

        try:
            enrichment = await enrich_candidate_with_llm(fetcher.llm, profile_text_for_llm)
            headline = enrichment.get("headline")
            summary = enrichment.get("summary")
            years_exp = enrichment.get("years_experience")
            skills = enrichment.get("skills") or []
            domains = enrichment.get("domains") or []
            roles = enrichment.get("roles") or []
            education = enrichment.get("education") or []
        except Exception:
            pass

        if roles and not current_title:
            current_role = next((r for r in roles if r.get("is_current")), roles[0] if roles else None)
            if current_role:
                current_title = current_title or current_role.get("title", "")
                current_company = current_company or current_role.get("company", "")

        cand = existing_by_url.get(url)
        if not cand:
            cand = Candidate(
                profile_url=url,
                name=name or "Unknown",
                headline=headline or title,
                current_title=current_title,
                current_company=current_company,
                location=None,
                years_experience=years_exp,
                skills=skills,
                domains=domains,
                experience=roles,
                education=education,
                raw_text=raw_text,
                source="linkedin",
                last_fetched_at=now,
            )
            session.add(cand)
            session.flush()
        else:
            cand.raw_text = raw_text
            cand.headline = headline or cand.headline or title
            cand.current_title = current_title or cand.current_title
            cand.current_company = current_company or cand.current_company
            cand.years_experience = years_exp or cand.years_experience
            cand.skills = skills or cand.skills
            cand.domains = domains or cand.domains
            cand.experience = roles or cand.experience
            cand.education = education or cand.education
            cand.last_fetched_at = now

        emb_row = (
            session.query(CandidateEmbedding)
            .filter_by(candidate_id=cand.id)
            .one_or_none()
        )
        if not emb_row:
            emb = await fetcher.embedder.embed_text(raw_text)
            emb_row = CandidateEmbedding(
                candidate_id=cand.id,
                embedding=emb,
                model_name=fetcher.embedder.model,
            )
            session.add(emb_row)

        return cand

    async def worker(url: str) -> Optional[Candidate]:
        async with fetcher.semaphore:
            await asyncio.sleep(random.uniform(0.5, 1.5))
            return await process_url(url)

    tasks = [worker(url) for url in to_fetch]
    fetched_candidates = await asyncio.gather(*tasks)

    for c in fetched_candidates:
        if c:
            candidates.append(c)

    session.commit()
    return candidates

