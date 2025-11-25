import asyncio
import sys
from src.db.session import SessionLocal
from src.core.llm_client import LLMClient, EmbeddingClient
from src.core.placements import build_placement_profiles
from src.config import settings

async def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Building {limit} placement profiles...")
    
    if not settings.llm_api_key:
        print("ERROR: LLM_API_KEY not set in .env file")
        return
    
    llm = LLMClient(model=settings.llm_model)
    embedder = EmbeddingClient(model=settings.embedding_model)
    
    session = SessionLocal()
    try:
        created = await build_placement_profiles(
            session=session,
            llm=llm,
            embedder=embedder,
            limit=limit,
        )
        print(f"\nCreated {created} placement profiles")
    finally:
        session.close()

if __name__ == "__main__":
    asyncio.run(main())

