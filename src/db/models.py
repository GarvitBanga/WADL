from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Integer, String, Text, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Placement(Base):
    __tablename__ = "placements"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, index=True)
    company: Mapped[str] = mapped_column(String)
    job_title: Mapped[str] = mapped_column(String)
    position_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    placement_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    date_posted: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    placement_date: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    start_date: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    profiles: Mapped[List["PlacementProfile"]] = relationship(back_populates="placement")

class JobDescription(Base):
    __tablename__ = "jds"
    id: Mapped[int] = mapped_column(primary_key=True)
    raw_text: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String)
    seniority: Mapped[str] = mapped_column(String)
    domain: Mapped[List[str]] = mapped_column(JSON)
    must_have_skills: Mapped[List[str]] = mapped_column(JSON)
    nice_to_have_skills: Mapped[List[str]] = mapped_column(JSON)
    min_years_experience: Mapped[int] = mapped_column(Integer)
    location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    embedding: Mapped["JDEmbedding"] = relationship(back_populates="jd")
    runs: Mapped[List["Run"]] = relationship(back_populates="jd")

class JDEmbedding(Base):
    __tablename__ = "jd_embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    jd_id: Mapped[int] = mapped_column(ForeignKey("jds.id"))
    embedding: Mapped[List[float]] = mapped_column(JSON)
    model_name: Mapped[str] = mapped_column(String)
    jd: Mapped["JobDescription"] = relationship(back_populates="embedding")

class Candidate(Base):
    __tablename__ = "candidates"
    id: Mapped[int] = mapped_column(primary_key=True)
    profile_url: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    headline: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    current_title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    current_company: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    years_experience: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    skills: Mapped[List[str]] = mapped_column(JSON)
    domains: Mapped[List[str]] = mapped_column(JSON)
    experience: Mapped[List[dict]] = mapped_column(JSON)
    education: Mapped[List[dict]] = mapped_column(JSON)
    raw_text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String)
    last_fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    embedding: Mapped["CandidateEmbedding"] = relationship(back_populates="candidate")
    run_associations: Mapped[List["RunCandidate"]] = relationship(back_populates="candidate")

class CandidateEmbedding(Base):
    __tablename__ = "candidate_embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"))
    embedding: Mapped[List[float]] = mapped_column(JSON)
    model_name: Mapped[str] = mapped_column(String)
    candidate: Mapped["Candidate"] = relationship(back_populates="embedding")

class PlacementProfile(Base):
    __tablename__ = "placement_profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    placement_id: Mapped[int] = mapped_column(ForeignKey("placements.id"))
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"))
    placement: Mapped["Placement"] = relationship(back_populates="profiles")
    candidate: Mapped["Candidate"] = relationship()

class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    jd_id: Mapped[int] = mapped_column(ForeignKey("jds.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    urls_found: Mapped[int] = mapped_column(Integer, default=0)
    profiles_parsed: Mapped[int] = mapped_column(Integer, default=0)
    profiles_from_cache: Mapped[int] = mapped_column(Integer, default=0)
    sourcing_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    ranking_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    estimated_llm_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_llm_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    jd: Mapped["JobDescription"] = relationship(back_populates="runs")
    candidates: Mapped[List["RunCandidate"]] = relationship(back_populates="run")
    logs: Mapped[List["AgentLog"]] = relationship(back_populates="run")

class RunCandidate(Base):
    __tablename__ = "run_candidates"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"))
    score: Mapped[float] = mapped_column(Float)
    feature_breakdown: Mapped[Dict[str, Any]] = mapped_column(JSON)
    placement_similarity: Mapped[float] = mapped_column(Float)
    closest_placement_id: Mapped[Optional[int]] = mapped_column(ForeignKey("placements.id"), nullable=True)
    run: Mapped["Run"] = relationship(back_populates="candidates")
    candidate: Mapped["Candidate"] = relationship(back_populates="run_associations")

class AgentLog(Base):
    __tablename__ = "agent_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    agent_name: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)
    reasoning: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    run: Mapped["Run"] = relationship(back_populates="logs")

