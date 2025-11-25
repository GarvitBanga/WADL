import streamlit as st
from textwrap import dedent
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.db.session import SessionLocal
from src.core.llm_client import LLMClient, EmbeddingClient
from src.core.search_client import SearchClient
from src.core.pipeline import run_matching_pipeline_sync
from src.core.scoring import build_explanation
from src.db.models import (
    JobDescription,
    Run,
    RunCandidate,
    Candidate,
    AgentLog,
    Placement,
)
import src.config as config

SAMPLE_JD = dedent("""
Director of Residential Services

We are seeking a Director of Residential Services to oversee behavioral health programs 
for adults with intellectual and developmental disabilities (I/DD). The director will
manage multiple group homes, supervise program managers and clinical staff, ensure
regulatory compliance, and drive quality improvement initiatives.

Requirements:
- 7+ years experience in behavioral health or residential services
- 3+ years in a supervisory or director-level role
- Experience with I/DD populations
- Strong leadership, communication, and crisis management skills

Location: New York, NY (hybrid)
""")

@st.cache_resource
def get_clients():
    llm = LLMClient(model=config.settings.llm_model)
    embedder = EmbeddingClient(model=config.settings.embedding_model)
    search_client = SearchClient()
    return llm, embedder, search_client

def main():
    st.set_page_config(page_title="WADL Agentic Recruiter", layout="wide")
    st.title("Agentic Recruiter")
    st.markdown("**Automatically find and rank top candidates for your job description**")
    st.markdown("---")

    llm, embedder, search_client = get_clients()

    st.sidebar.header("Job Description")
    use_sample = st.sidebar.checkbox("Use sample job description", value=True)
    jd_text = st.sidebar.text_area(
        "Paste your job description:",
        value=SAMPLE_JD if use_sample else "",
        height=280,
    )

    target_profiles = st.sidebar.slider(
        "Number of candidates to find", 
        5, 20, 5, 5,
    )
    
    config.settings.skip_html_fetch = False
    use_browser = False
    
    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("Find Top Candidates", type="primary", use_container_width=True)

    run = None

    if run_btn and jd_text.strip():
        with st.spinner("Running agentic sourcing & ranking..."):
            session = SessionLocal()
            try:
                run = run_matching_pipeline_sync(
                    session=session,
                    raw_jd_text=jd_text.strip(),
                    target_profiles=target_profiles,
                    llm=llm,
                    embedder=embedder,
                    search_client=search_client,
                    use_browser=use_browser,
                )
                st.session_state["last_run_id"] = run.id
            finally:
                session.close()

    if not run and "last_run_id" in st.session_state:
        session = SessionLocal()
        try:
            run = session.query(Run).filter_by(
                id=st.session_state["last_run_id"]
            ).one_or_none()
        finally:
            session.close()

    if not run:
        st.info("Run a search from the sidebar to see results.")
        return

    session = SessionLocal()
    try:
        jd = session.query(JobDescription).filter_by(id=run.jd_id).one()
        logs = (
            session.query(AgentLog)
            .filter(AgentLog.run_id == run.id)
            .order_by(AgentLog.timestamp.asc())
            .all()
        )

        st.subheader("Job Requirements")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Role Details**")
            st.markdown(f"**Title:** {jd.title}")
            st.markdown(f"**Level:** {jd.seniority}")
            st.markdown(f"**Domain:** {', '.join(jd.domain or [])}")
            st.markdown(f"**Experience Required:** {jd.min_years_experience}+ years")
            if jd.location:
                st.markdown(f"**Location:** {jd.location}")

        with c2:
            st.markdown("**Key Skills**")
            if jd.must_have_skills:
                st.markdown(", ".join(jd.must_have_skills[:10]))
            if jd.nice_to_have_skills:
                st.markdown(f"*Nice to have:* {', '.join(jd.nice_to_have_skills[:5])}")

        st.subheader("Search Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Candidates Found", run.profiles_parsed)
        m2.metric("Profiles Analyzed", run.profiles_parsed)
        m3.metric("Total Time", f"{(run.sourcing_time_ms + run.ranking_time_ms) / 1000:.1f}s")

        st.subheader("Top Candidates (Ranked by Match Quality)")

        q = (
            session.query(RunCandidate, Candidate, Placement)
            .join(Candidate, RunCandidate.candidate_id == Candidate.id)
            .outerjoin(Placement, RunCandidate.closest_placement_id == Placement.id)
            .filter(RunCandidate.run_id == run.id)
            .order_by(RunCandidate.score.desc())
        )
        rows = q.all()
        if not rows:
            st.warning("No candidates for this run.")
            return

        table_rows = []
        for rc, cand, placement in rows[:50]:
            placement_label = ""
            if placement:
                placement_label = f"{placement.job_title} @ {placement.company}"
            table_rows.append({
                "Rank": len(table_rows) + 1,
                "Name": cand.name,
                "Current Role": cand.current_title or "",
                "Company": cand.current_company or "",
                "Match Score": f"{round(rc.score * 100, 1)}%",
                "Similar to Past Hire": placement_label if placement_label else "—",
            })

        st.dataframe(table_rows, hide_index=True)

        names = [row[1].name for row in rows[:50]]
        selected_name = st.selectbox("View detailed profile:", options=names, index=0)

        selected = next((row for row in rows if row[1].name == selected_name), None)
        if selected:
            rc, cand, placement = selected
            features = rc.feature_breakdown or {}
            explanation_bullets = build_explanation(jd, cand, features, placement)

            st.markdown(f"### {cand.name}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Current Role:** {cand.current_title or 'N/A'}")
                st.markdown(f"**Company:** {cand.current_company or 'N/A'}")
            with col2:
                st.markdown(f"**Match Score:** {round(rc.score * 100, 1)}%")
                if placement:
                    st.markdown(f"**Similar to:** {placement.job_title} @ {placement.company}")
            
            st.markdown(f"**LinkedIn:** [{cand.profile_url}]({cand.profile_url})")
            
            st.markdown("---")
            st.markdown("### Why This Candidate is a Strong Match")
            for b in explanation_bullets:
                st.markdown(f"- {b}")

            st.markdown("---")
            st.markdown("### Profile Details")

            if cand.headline:
                st.markdown(f"**Headline:** {cand.headline}")

            if cand.experience:
                st.markdown("**Career History (Extracted):**")
                for idx, role in enumerate(cand.experience[:5], 1):
                    dates = f"{role.get('start_date', '')} – {role.get('end_date', '')}" if role.get('start_date') or role.get('end_date') else ""
                    current = " (Current)" if role.get('is_current') else ""
                    st.markdown(
                        f"{idx}. **{role.get('title', 'N/A')}** at {role.get('company', 'N/A')} {dates}{current}"
                    )
                    if role.get('description'):
                        st.markdown(f"   *{role.get('description', '')}*")
                    if role.get('location'):
                        st.markdown(f"   Location: {role.get('location', '')}")
                    st.markdown("")
            else:
                st.info("No experience data extracted")

            if cand.education:
                st.markdown("**Education (Extracted):**")
                for idx, ed in enumerate(cand.education[:5], 1):
                    degree_part = f"{ed.get('degree', '') or ''} in {ed.get('field', '') or ''}" if ed.get('degree') or ed.get('field') else "Degree"
                    year_part = f"({ed.get('start_year', '')}–{ed.get('end_year', '')})" if ed.get('start_year') or ed.get('end_year') else ""
                    st.markdown(f"{idx}. {degree_part} from {ed.get('school', 'N/A')} {year_part}")
            else:
                st.info("No education data extracted")

            if cand.skills:
                st.markdown(f"**Skills (Extracted):** {', '.join(cand.skills[:15])}")
                if len(cand.skills) > 15:
                    st.markdown(f"*... and {len(cand.skills) - 15} more*")

            if cand.domains:
                st.markdown(f"**Domains (Extracted):** {', '.join(cand.domains)}")

            st.markdown("---")
            with st.expander("Technical Details"):
                st.markdown("**Feature Breakdown:**")
                st.json(features)
                
                extracted_data = {
                    "headline": cand.headline,
                    "years_experience": cand.years_experience,
                    "skills": cand.skills,
                    "domains": cand.domains,
                    "experience": cand.experience,
                    "education": cand.education,
                }
                st.markdown("**Extracted Data:**")
                st.json(extracted_data)
    finally:
        session.close()

if __name__ == "__main__":
    main()

