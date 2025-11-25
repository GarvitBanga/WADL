from typing import List
from sqlalchemy.orm import Session
from src.db.models import Placement, Candidate, CandidateEmbedding, PlacementProfile
from src.core.llm_client import LLMClient, EmbeddingClient

PLACEMENT_PROFILE_PROMPT = """
You are generating a realistic LinkedIn-style profile summary for a person.

Input:
- Name: {name}
- Job Title: {job_title}
- Company: {company}

Task:
Invent a plausible professional profile for this person, consistent with the title and company.

Return a JSON object with:
- "headline": short 1-line headline for their LinkedIn profile
- "summary": 3-6 sentences describing their background and responsibilities
- "history": list of 2-4 past roles, each as an object:
    {{ "title": str, "company": str, "years": float, "description": str }}
- "skills": list of 8-20 skills/keywords (strings)
- "total_years_experience": float (approx years of experience)

Return ONLY JSON, no extra text.
"""

async def build_placement_profiles(
    session: Session,
    llm: LLMClient,
    embedder: EmbeddingClient,
    limit: int = 200,
) -> int:
    placements = (
        session.query(Placement)
        .filter(~Placement.profiles.any())
        .limit(limit)
        .all()
    )

    total = len(placements)
    created = 0
    for idx, placement in enumerate(placements, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"Processing {idx}/{total}...")
        prompt = PLACEMENT_PROFILE_PROMPT.format(
            name=placement.name,
            job_title=placement.job_title,
            company=placement.company,
        )
        try:
            profile_json = await llm.complete_json(prompt)
        except Exception as e:
            print(f"Error on placement {placement.id}: {e}")
            continue

        headline = profile_json.get("headline") or f"{placement.job_title} at {placement.company}"
        summary = profile_json.get("summary") or ""
        history = profile_json.get("history") or []
        skills = profile_json.get("skills") or []
        total_yrs = profile_json.get("total_years_experience") or None

        history_text_parts = []
        for h in history:
            history_text_parts.append(
                f"{h.get('title', '')} at {h.get('company', '')} "
                f"({h.get('years', '')} years): {h.get('description', '')}"
            )
        history_text = "\n".join(history_text_parts)
        skills_text = ", ".join(skills)
        raw_text = "\n\n".join([summary, history_text, "Skills: " + skills_text])

        candidate = Candidate(
            profile_url=f"placement://{placement.id}",
            name=placement.name,
            headline=headline,
            current_title=placement.job_title,
            current_company=placement.company,
            location=None,
            years_experience=total_yrs,
            skills=skills,
            domains=[],
            raw_text=raw_text,
            source="synthetic_placement",
        )
        session.add(candidate)
        session.flush()

        try:
            emb = await embedder.embed_text(raw_text)
        except Exception as e:
            print(f"Embedding error on placement {placement.id}: {e}")
            session.rollback()
            continue

        cand_emb = CandidateEmbedding(
            candidate_id=candidate.id,
            embedding=emb,
            model_name=embedder.model,
        )
        session.add(cand_emb)

        link = PlacementProfile(
            placement_id=placement.id,
            candidate_id=candidate.id,
        )
        session.add(link)

        created += 1

    session.commit()
    return created

