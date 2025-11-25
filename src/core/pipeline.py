import asyncio
from sqlalchemy.orm import Session
from src.core.llm_client import LLMClient, EmbeddingClient
from src.core.jd import create_jd
from src.core.sourcing import SourcingAgent
from src.core.search_client import SearchClient
from src.core.profiles import ProfileFetcher
from src.core.agents import AgentLogger
from src.core.scoring import score_run
from src.db.models import Run

async def run_matching_pipeline(
    session: Session,
    raw_jd_text: str,
    target_profiles: int,
    llm: LLMClient,
    embedder: EmbeddingClient,
    search_client: SearchClient,
    use_browser: bool = True,
) -> Run:
    logger = AgentLogger()
    profile_fetcher = ProfileFetcher(
        llm=llm, 
        embedder=embedder, 
        use_browser=use_browser,
        max_concurrent=2,
        request_delay=5.0
    )

    jd = await create_jd(session, llm, embedder, raw_jd_text)

    sourcing_agent = SourcingAgent(
        llm=llm,
        search_client=search_client,
        profile_fetcher=profile_fetcher,
        logger=logger,
    )

    max_rounds = 2 if target_profiles <= 10 else 3
    run = await sourcing_agent.run(
        session=session,
        jd=jd,
        target_profiles=target_profiles,
        max_rounds=max_rounds,
    )

    score_run(session, run)
    session.refresh(run)
    return run

def run_matching_pipeline_sync(
    session: Session,
    raw_jd_text: str,
    target_profiles: int,
    llm: LLMClient,
    embedder: EmbeddingClient,
    search_client: SearchClient,
    use_browser: bool = True,
) -> Run:
    return asyncio.run(
        run_matching_pipeline(
            session=session,
            raw_jd_text=raw_jd_text,
            target_profiles=target_profiles,
            llm=llm,
            embedder=embedder,
            search_client=search_client,
            use_browser=use_browser,
        )
    )

