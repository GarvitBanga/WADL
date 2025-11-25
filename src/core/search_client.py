from typing import List, Dict
import httpx
import re
from src.config import settings

PROFILE_URL_PATTERNS = [
    r"linkedin\.com/in/",
    r"linkedin\.com/pub/",
]

def looks_like_profile(url: str) -> bool:
    return any(re.search(p, url) for p in PROFILE_URL_PATTERNS)

class SearchClient:
    def __init__(self):
        self.api_key = settings.search_api_key or settings.bing_api_key

    async def search(self, query: str, num_results: int = 20) -> List[Dict[str, str]]:
        if not self.api_key:
            return []
        
        if settings.search_api_key:
            return await self._serpapi_search(query, num_results)
        elif settings.bing_api_key:
            return await self._bing_search(query, num_results)
        return []

    async def _serpapi_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "num": num_results * 2,
            }
            try:
                resp = await client.get("https://serpapi.com/search", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    cleaned = []
                    for item in data.get("organic_results", []):
                        url = item.get("link", "")
                        title = item.get("title", "")
                        snippet = item.get("snippet", "")

                        if not url:
                            continue

                        lower = url.lower()

                        if "linkedin.com" not in lower:
                            continue

                        if "linkedin.com/jobs" in lower:
                            continue

                        if not looks_like_profile(lower):
                            continue

                        job_posting_domains = [
                            "jobs.lever.co",
                            "boards.greenhouse.io",
                            "apply.workable.com",
                            "jobs.ashbyhq.com",
                            "jobs.smartrecruiters.com",
                        ]
                        if any(domain in lower for domain in job_posting_domains):
                            continue

                        cleaned.append({
                            "url": url,
                            "title": title,
                            "snippet": snippet,
                        })

                        if len(cleaned) >= num_results:
                            break

                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"SerpAPI search '{query}': {len(data.get('organic_results', []))} raw results, {len(cleaned)} after filtering")
                    return cleaned
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"SerpAPI search error: {e}")
                pass
        return []

    async def _bing_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {"q": query, "count": num_results}
            try:
                resp = await client.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers=headers,
                    params=params,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    for item in data.get("webPages", {}).get("value", [])[:num_results]:
                        results.append({
                            "url": item.get("url", ""),
                            "title": item.get("name", ""),
                            "snippet": item.get("snippet", ""),
                        })
                    return results
            except Exception:
                pass
        return []

