from typing import List
from dataclasses import dataclass
import json
import asyncio
from openai import OpenAI
from src.config import settings

_client = None

def get_client():
    global _client
    if _client is None:
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY not set in config")
        _client = OpenAI(api_key=settings.llm_api_key, timeout=30.0)
    return _client

@dataclass
class LLMClient:
    model: str
    
    async def complete_json(self, prompt: str) -> dict:
        client = get_client()
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
            ),
            timeout=60.0
        )
        return json.loads(response.choices[0].message.content)

@dataclass
class EmbeddingClient:
    model: str
    
    async def embed_text(self, text: str) -> List[float]:
        client = get_client()
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.embeddings.create(
                    model=self.model,
                    input=text
                )
            ),
            timeout=30.0
        )
        return response.data[0].embedding

