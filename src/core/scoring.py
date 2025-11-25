from typing import Dict, List, Tuple, Optional
import numpy as np
from time import perf_counter
from sqlalchemy.orm import Session
from src.db.models import (
    JobDescription,
    JDEmbedding,
    Candidate,
    CandidateEmbedding,
    PlacementProfile,
    Run,
    RunCandidate,
    Placement,
)

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def compute_ideal_embedding_for_jd(
    session: Session,
    jd: JobDescription,
    top_k: int = 3,
) -> Tuple[Optional[List[float]], Optional[Placement]]:
    jd_emb = (
        session.query(JDEmbedding)
        .filter_by(jd_id=jd.id)
        .one_or_none()
    )
    if not jd_emb:
        return None, None

    jd_vec = jd_emb.embedding

    q = (
        session.query(PlacementProfile, CandidateEmbedding)
        .join(CandidateEmbedding, PlacementProfile.candidate_id == CandidateEmbedding.candidate_id)
    )
    rows = q.all()
    if not rows:
        return None, None

    scored = []
    for pp, ce in rows:
        sim = cosine_similarity(jd_vec, ce.embedding)
        scored.append((sim, pp, ce))

    if not scored:
        return None, None

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    vecs = [ce.embedding for _, _, ce in top]
    ideal_vec = list(np.mean(np.array(vecs), axis=0))

    best_sim, best_pp, best_ce = top[0]
    best_placement = (
        session.query(Placement)
        .filter_by(id=best_pp.placement_id)
        .one_or_none()
    )

    return ideal_vec, best_placement

def map_title_to_level(title: str) -> int:
    t = (title or "").lower()
    if "vp" in t or "vice president" in t or "chief" in t or "cmo" in t or "ceo" in t:
        return 4
    if "director" in t:
        return 3
    if "manager" in t or "lead" in t or "supervisor" in t:
        return 2
    return 1

def compute_features_for_candidate(
    jd: JobDescription,
    jd_vec: List[float],
    ideal_vec: Optional[List[float]],
    candidate: Candidate,
    cand_vec: List[float],
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    features["sim_JD"] = cosine_similarity(jd_vec, cand_vec)
    features["sim_ideal"] = cosine_similarity(ideal_vec, cand_vec) if ideal_vec is not None else 0.0

    if candidate.years_experience is not None and jd.min_years_experience > 0:
        ratio = candidate.years_experience / jd.min_years_experience
        features["experience_ok"] = min(1.0, ratio)
    else:
        features["experience_ok"] = 0.5

    jd_must = set((jd.must_have_skills or []))
    cand_skills = set((candidate.skills or []))
    if jd_must:
        overlap = len(jd_must & cand_skills) / len(jd_must)
    else:
        overlap = 0.5
    features["skills_overlap"] = overlap

    text = (candidate.raw_text or "").lower()
    domain_terms = jd.domain or []
    if domain_terms:
        hits = 0
        for term in domain_terms:
            if term.lower() in text:
                hits += 1
        features["domain_match"] = hits / len(domain_terms)
    else:
        features["domain_match"] = 0.5

    jd_level = map_title_to_level(jd.title)
    cand_level = map_title_to_level(candidate.current_title or "")
    diff = abs(jd_level - cand_level)
    features["level_match"] = max(0.0, 1.0 - diff / 3.0)

    features["tenure_score"] = 0.5

    return features

def compute_score(f: Dict[str, float]) -> float:
    placement_component = 0.7 * f["sim_JD"] + 0.3 * f["sim_ideal"]
    heuristics_component = (
        0.15 * f["experience_ok"] +
        0.15 * f["skills_overlap"] +
        0.10 * f["domain_match"] +
        0.10 * f["level_match"] +
        0.05 * f["tenure_score"]
    )
    return placement_component + heuristics_component

def build_explanation(
    jd: JobDescription,
    candidate: Candidate,
    features: Dict[str, float],
    closest_placement: Optional[Placement],
) -> List[str]:
    bullets: List[str] = []

    if candidate.years_experience is not None:
        bullets.append(
            f"{candidate.years_experience:.1f} years experience; "
            f"current role: {candidate.current_title or 'Unknown'} at {candidate.current_company or 'Unknown'}."
        )
    else:
        bullets.append(
            f"Current role: {candidate.current_title or 'Unknown'} at {candidate.current_company or 'Unknown'}."
        )

    jd_must = jd.must_have_skills or []
    cand_skills = candidate.skills or []
    matched_skills = list(set(jd_must) & set(cand_skills))
    if jd_must:
        bullets.append(
            f"Matches {len(matched_skills)}/{len(jd_must)} required skills"
            + (f" ({', '.join(matched_skills[:4])}…)" if matched_skills else ".")
        )

    if features["domain_match"] > 0.4 and jd.domain:
        bullets.append(
            "Domain alignment in: " + ", ".join(jd.domain)
        )

    if closest_placement and features["sim_ideal"] > 0.5:
        bullets.append(
            f"Profile is similar to past hire: {closest_placement.job_title} at {closest_placement.company}."
        )

    return bullets

def score_run(session: Session, run: Run) -> None:
    jd = session.query(JobDescription).filter_by(id=run.jd_id).one()
    jd_emb = session.query(JDEmbedding).filter_by(jd_id=jd.id).one()
    jd_vec = jd_emb.embedding

    ideal_vec, closest_placement_for_label = compute_ideal_embedding_for_jd(session, jd)

    t0 = perf_counter()

    run_candidates = session.query(RunCandidate).filter_by(run_id=run.id).all()
    candidate_ids = [rc.candidate_id for rc in run_candidates]

    candidates = (
        session.query(Candidate)
        .filter(Candidate.id.in_(candidate_ids))
        .all()
    )
    cand_by_id = {c.id: c for c in candidates}

    cand_embs = (
        session.query(CandidateEmbedding)
        .filter(CandidateEmbedding.candidate_id.in_(candidate_ids))
        .all()
    )
    emb_by_cid = {ce.candidate_id: ce.embedding for ce in cand_embs}

    pp_with_embs = (
        session.query(PlacementProfile, CandidateEmbedding)
        .join(CandidateEmbedding, PlacementProfile.candidate_id == CandidateEmbedding.candidate_id)
        .all()
    )

    for rc in run_candidates:
        cand = cand_by_id.get(rc.candidate_id)
        cand_vec = emb_by_cid.get(rc.candidate_id)
        if not cand or not cand_vec:
            continue

        f = compute_features_for_candidate(jd, jd_vec, ideal_vec, cand, cand_vec)

        best_sim = 0.0
        best_pp: Optional[PlacementProfile] = None
        for pp, ce in pp_with_embs:
            sim = cosine_similarity(cand_vec, ce.embedding)
            if sim > best_sim:
                best_sim = sim
                best_pp = pp

        if best_pp:
            rc.placement_similarity = best_sim
            rc.closest_placement_id = best_pp.placement_id
        else:
            rc.placement_similarity = 0.0
            rc.closest_placement_id = None

        score = compute_score(f)
        rc.score = score
        rc.feature_breakdown = f

    run.ranking_time_ms = int((perf_counter() - t0) * 1000)
    session.commit()

