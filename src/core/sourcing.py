from dataclasses import dataclass, field
from typing import List, Dict, Set
from time import perf_counter
import json
import asyncio
from sqlalchemy.orm import Session
from src.db.models import JobDescription, Run
from src.core.agents import AgentLogger
from src.core.llm_client import LLMClient
from src.core.search_client import SearchClient
from src.core.profiles import ProfileFetcher, fetch_and_parse_batch

@dataclass
class SourcingState:
    jd_id: int
    queries_run: List[str] = field(default_factory=list)
    urls_collected: Set[str] = field(default_factory=set)
    total_profiles: int = 0
    candidate_ids: List[int] = field(default_factory=list)
    domain_hits: Dict[str, int] = field(default_factory=lambda: {
        "behavioral_health": 0,
        "residential": 0,
        "idd": 0,
        "nursing": 0,
    })
    target_profiles: int = 100

def update_domain_counts(state: SourcingState, candidate_text: str):
    text = candidate_text.lower()
    if "behavioral health" in text:
        state.domain_hits["behavioral_health"] += 1
    if "residential" in text:
        state.domain_hits["residential"] += 1
    if "i/dd" in text or "intellectual and developmental" in text:
        state.domain_hits["idd"] += 1
    if "nurse" in text or "rn" in text:
        state.domain_hits["nursing"] += 1

INITIAL_QUERIES_PROMPT = """
You are a sourcing agent for recruiters.

You receive this structured job description JSON:
{jd_json}

Propose 2-3 web search queries to find public professional profiles 
(especially LinkedIn) of strong candidates for this role.

Vary the queries by:
- title synonyms
- domain keywords

ALWAYS include `site:linkedin.com/in/` to target LinkedIn profiles specifically.

Return ONLY a JSON array of query strings.
"""

REFINE_QUERIES_PROMPT = """
You are refining a sourcing strategy.

Job description JSON:
{jd_json}

Current coverage summary:
{coverage_summary}

We need to improve coverage for missing domains or seniority segments.

Propose 1-2 new *targeted* search queries to find more relevant profiles.

IMPORTANT:
- Use natural language search terms, NOT field:value syntax
- ALWAYS include `site:linkedin.com/in/` to target LinkedIn profiles
- Use AND/OR operators with quotes for exact phrases
- Example: "Software Engineer" AND (healthcare OR "behavioral health") site:linkedin.com/in/

Return ONLY a JSON array of query strings.
"""

async def initial_queries_from_llm(llm: LLMClient, jd: JobDescription) -> List[str]:
    jd_json = json.dumps({
        "title": jd.title,
        "seniority": jd.seniority,
        "domain": jd.domain,
        "must_have_skills": jd.must_have_skills,
        "location": jd.location,
    })
    prompt = INITIAL_QUERIES_PROMPT.format(jd_json=jd_json)
    data = await llm.complete_json(prompt)
    if isinstance(data, dict):
        if "queries" in data:
            return data["queries"]
        if "query" in data:
            return [data["query"]]
    if isinstance(data, list):
        return data
    return []

def summarize_state_for_llm(state: SourcingState, jd: JobDescription) -> str:
    total = max(state.total_profiles, 1)
    return json.dumps({
        "jd_title": jd.title,
        "target_profiles": state.target_profiles,
        "total_profiles": state.total_profiles,
        "domain_hits": {
            k: f"{v} ({v/total:.2f} ratio)" for k, v in state.domain_hits.items()
        },
    })

async def refine_queries_with_llm(llm: LLMClient, jd: JobDescription, state: SourcingState) -> List[str]:
    jd_json = json.dumps({
        "title": jd.title,
        "seniority": jd.seniority,
        "domain": jd.domain,
    })
    coverage_summary = summarize_state_for_llm(state, jd)
    prompt = REFINE_QUERIES_PROMPT.format(
        jd_json=jd_json,
        coverage_summary=coverage_summary,
    )
    data = await llm.complete_json(prompt)
    if isinstance(data, dict):
        if "queries" in data:
            return data["queries"]
        if "query" in data:
            return [data["query"]]
    if isinstance(data, list):
        return data
    return []

def is_satisfied(state: SourcingState) -> bool:
    if state.total_profiles < state.target_profiles * 0.8:
        return False
    total = max(state.total_profiles, 1)
    bh_ratio = state.domain_hits["behavioral_health"] / total
    return bh_ratio >= 0.4

class SourcingAgent:
    def __init__(
        self,
        llm: LLMClient,
        search_client: SearchClient,
        profile_fetcher: ProfileFetcher,
        logger: AgentLogger,
    ):
        self.llm = llm
        self.search_client = search_client
        self.profile_fetcher = profile_fetcher
        self.logger = logger

    async def run(
        self,
        session: Session,
        jd: JobDescription,
        target_profiles: int = 100,
        max_rounds: int = 3,
    ) -> Run:
        run = Run(jd_id=jd.id)
        session.add(run)
        session.flush()

        state = SourcingState(jd_id=jd.id, target_profiles=target_profiles)

        self.logger.log("Manager", "Start", f"Starting sourcing for '{jd.title}'", "Thinking")

        t0 = perf_counter()
        queries = await initial_queries_from_llm(self.llm, jd)
        self.logger.log("Scout", "GenerateQueries", f"Initial queries: {queries}", "Done")

        for round_idx in range(max_rounds):
            self.logger.log(
                "Manager",
                "RoundStart",
                f"Round {round_idx+1}, running {len(queries)} queries",
                "Acting",
            )

            results: List[dict] = []
            for q in queries:
                search_res = await self.search_client.search(q, num_results=20)
                results.extend(search_res)
                self.logger.log(
                    "Scout",
                    "Search",
                    f"Query '{q}' returned {len(search_res)} results",
                    "Done",
                )

            unique_by_url = {}
            for r in results:
                url = r["url"]
                if url not in state.urls_collected:
                    unique_by_url[url] = r

            new_results = list(unique_by_url.values())
            for r in new_results:
                state.urls_collected.add(r["url"])

            self.logger.log(
                "Scout",
                "URLCollection",
                f"Collected {len(new_results)} new unique URLs (total: {len(state.urls_collected)})",
                "Done",
            )

            needed = state.target_profiles - state.total_profiles
            if needed <= 0:
                break

            batch_size = min(needed + 1, len(new_results))
            batch = new_results[:batch_size]
            self.logger.log(
                "Scout",
                "FetchBatch",
                f"Fetching {len(batch)} profiles (needed: {needed})",
                "Acting",
            )

            candidates = await fetch_and_parse_batch(self.profile_fetcher, session, batch)
            self.logger.log(
                "Scout",
                "FetchComplete",
                f"Fetched {len(candidates)} candidates from batch",
                "Done",
            )

            for cand in candidates:
                if cand:
                    state.total_profiles += 1
                    state.candidate_ids.append(cand.id)
                    update_domain_counts(state, cand.raw_text)

            self.logger.log(
                "Manager",
                "CoverageUpdate",
                f"Now have {state.total_profiles} profiles. Domain hits: {state.domain_hits}",
                "Done",
            )

            if is_satisfied(state):
                self.logger.log(
                    "Manager",
                    "Stop",
                    "Target profiles and domain coverage achieved.",
                    "Done",
                )
                break

            queries = await refine_queries_with_llm(self.llm, jd, state)
            self.logger.log(
                "Scout",
                "RefineQueries",
                f"Refined queries: {queries}",
                "Acting",
            )

        sourcing_time_ms = int((perf_counter() - t0) * 1000)

        run.urls_found = len(state.urls_collected)
        run.profiles_parsed = state.total_profiles
        run.sourcing_time_ms = sourcing_time_ms

        self.logger.log(
            "Manager",
            "SourcingComplete",
            f"Final: {len(state.urls_collected)} URLs found, {state.total_profiles} profiles parsed, {len(state.candidate_ids)} candidates linked",
            "Done",
        )

        from src.db.models import RunCandidate
        for cid in state.candidate_ids:
            rc = RunCandidate(
                run_id=run.id,
                candidate_id=cid,
                score=0.0,
                feature_breakdown={},
                placement_similarity=0.0,
                closest_placement_id=None,
            )
            session.add(rc)

        session.commit()
        self.logger.flush_to_db(session, run)
        return run

