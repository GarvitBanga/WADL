from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.db.models import JobDescription, JDEmbedding
from src.core.llm_client import LLMClient, EmbeddingClient

class JDParsed(BaseModel):
    title: str
    seniority: str
    domain: List[str]
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    min_years_experience: int
    location: Optional[str]

JD_PARSE_PROMPT = """
You are extracting structured information from a job description.

Given the following job description text, extract and return a JSON object with:
- "title": string
- "seniority": string, one of ["IC", "Manager", "Director", "VP", "C-level"]
- "domain": array of 1-5 domain keywords (e.g. ["behavioral health", "residential services"])
- "must_have_skills": array of 5-20 core skills/technologies
- "nice_to_have_skills": array of optional or bonus skills
- "min_years_experience": integer (best estimate from the JD)
- "location": string or null

Return ONLY JSON.

Job Description:
---
{jd_text}
---
"""

async def parse_jd_text(llm: LLMClient, raw_text: str) -> JDParsed:
    prompt = JD_PARSE_PROMPT.format(jd_text=raw_text)
    data = await llm.complete_json(prompt)
    return JDParsed(**data)

async def create_jd(
    session: Session,
    llm: LLMClient,
    embedder: EmbeddingClient,
    raw_text: str,
) -> JobDescription:
    parsed = await parse_jd_text(llm, raw_text)

    jd = JobDescription(
        raw_text=raw_text,
        title=parsed.title,
        seniority=parsed.seniority,
        domain=parsed.domain,
        must_have_skills=parsed.must_have_skills,
        nice_to_have_skills=parsed.nice_to_have_skills,
        min_years_experience=parsed.min_years_experience,
        location=parsed.location,
    )
    session.add(jd)
    session.flush()

    summary_parts = [
        parsed.title,
        parsed.seniority,
        ", ".join(parsed.domain),
        f"min {parsed.min_years_experience} years experience",
        "Must-have: " + ", ".join(parsed.must_have_skills[:10]),
    ]
    summary_text = " | ".join(summary_parts)

    emb = await embedder.embed_text(summary_text)

    jd_emb = JDEmbedding(
        jd_id=jd.id,
        embedding=emb,
        model_name=embedder.model,
    )
    session.add(jd_emb)
    session.commit()

    return jd

