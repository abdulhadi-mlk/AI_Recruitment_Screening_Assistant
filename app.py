import html as html_lib
import os
import re
import sys
from typing import Dict

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pdf_parser import extract_text_from_pdf
from src.preprocessing import preprocess_text
from src.scoring import analyze_resume
from src.skill_extraction import get_jd_skills
from src.skill_ontology import flattened_skill_map, full_skill_weights_map, skill_to_category_map
from src.utils import build_explanation, group_skills_by_category

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
        :root {
          --paper: #EFF1EC;
          --paper-dim: #E4E7E0;
          --ink: #12181B;
          --ink-soft: #4B5760;
          --steel: #1E2C3B;
          --mint: #4FA593;
          --mint-dim: #DCEAE6;
          --green: #2E9E5B;
          --amber: #D9922E;
          --line: #CDD2C7;
        }
        html, body, [data-testid="stAppViewContainer"] {
          background: var(--paper);
          color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { display: none; }
        .stAppDeployButton, .stDownloadButton, #MainMenu { display: none !important; }
        .block-container {
          padding-top: 0;
          padding-bottom: 2rem;
          max-width: 1400px;
        }
        .app-shell { padding: 0 0 2rem 0; }
        .hero-card {
          background: linear-gradient(135deg, var(--steel) 0%, #27394a 100%);
          border-radius: 18px;
          padding: 1.4rem 1.4rem 1.2rem;
          color: white;
          box-shadow: 0 14px 40px rgba(30,44,59,0.18);
          border: 1px solid rgba(255,255,255,0.08);
          margin-bottom: 1.1rem;
          overflow: hidden;
          position: relative;
        }
        .hero-title {
          font-family: 'Space Grotesk', sans-serif;
          font-size: 1.85rem;
          font-weight: 700;
          margin: 0 0 0.25rem;
        }
        .hero-subtitle {
          color: var(--mint-dim);
          font-family: 'Inter', sans-serif;
          font-size: 0.95rem;
          margin: 0;
        }
        .hero-scan {
          margin-top: 1rem;
          display: grid;
          grid-template-columns: 1.3fr 0.7fr;
          gap: 1rem;
          align-items: center;
        }
        .scan-visual {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 16px;
          padding: 1rem;
          position: relative;
          min-height: 220px;
          overflow: hidden;
        }
        .scan-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.8rem;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.8rem;
          color: rgba(255,255,255,0.76);
        }
        .scan-lines {
          display: flex;
          flex-direction: column;
          gap: 0.45rem;
        }
        .scan-bar {
          height: 10px;
          border-radius: 999px;
          background: var(--paper-dim);
          opacity: 0.8;
        }
        .scan-line {
          position: absolute;
          left: 8%;
          right: 8%;
          height: 2px;
          border-radius: 999px;
          background: var(--mint);
          box-shadow: 0 0 16px rgba(79,165,147,0.8);
          animation: scan-line 3s ease-in-out infinite;
        }
        .scan-chip {
          position: absolute;
          display: inline-block;
          padding: 0.25rem 0.55rem;
          border-radius: 999px;
          background: rgba(255,255,255,0.12);
          color: white;
          border: 1px solid rgba(255,255,255,0.16);
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          opacity: 0;
          animation: chip-fade 3.6s ease-in-out infinite;
        }
        .scan-chip.chip-a { top: 18%; left: 12%; animation-delay: 0.2s; }
        .scan-chip.chip-b { top: 35%; right: 14%; animation-delay: 0.9s; }
        .scan-chip.chip-c { top: 58%; left: 18%; animation-delay: 1.6s; }
        .scan-chip.chip-d { bottom: 18%; right: 20%; animation-delay: 2.3s; }
.gauge {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          justify-content: flex-start;
          gap: 0.6rem;
        }
        .gauge-label {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.75rem;
          color: rgba(255,255,255,0.75);
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }
        .gauge-bar-container {
          width: 100%;
          height: 24px;
          background: rgba(255,255,255,0.12);
          border: 1px solid rgba(255,255,255,0.18);
          border-radius: 12px;
          overflow: hidden;
          display: flex;
          align-items: center;
          padding: 0 0.5rem;
          box-sizing: border-box;
        }
        .gauge-bar-fill {
          height: 16px;
          border-radius: 8px;
          background: linear-gradient(90deg, var(--mint) 0%, var(--green) 100%);
          transition: width 0.6s ease;
          box-shadow: 0 0 12px rgba(79,165,147,0.6);
        }
        .gauge-value {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.9rem;
          font-weight: 700;
          color: white;
          margin-left: 0.5rem;
          min-width: 3.5rem;
          text-align: right;
        }
        .gauge-caption {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          color: rgba(255,255,255,0.74);
          text-transform: uppercase;
          letter-spacing: 0.08em;
          margin-top: 0.6rem;
          text-align: center;
        }
        .section-card {
          background: white;
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 1rem;
          box-shadow: 0 10px 24px rgba(18,24,27,0.05);
          margin-bottom: 1rem;
        }
        .widget-label {
          font-family: 'Inter', sans-serif;
          font-size: 0.95rem;
          color: var(--ink-soft);
          margin-bottom: 0.45rem;
          font-weight: 600;
        }
        .stTextArea > div > div > textarea,
        .stTextInput > div > div > input,
        .stFileUploader > div > div {
          font-family: 'Inter', sans-serif !important;
          border-radius: 12px !important;
          border: 1px solid var(--line) !important;
          background: white !important;
          color: var(--ink) !important;
        }
        .stTextArea textarea { min-height: 220px !important; }
        .stButton > button {
          width: 100%;
          border-radius: 999px !important;
          background: var(--steel) !important;
          color: white !important;
          border: none !important;
          padding: 0.7rem 1rem !important;
          font-family: 'Space Grotesk', sans-serif !important;
          font-weight: 700 !important;
          box-shadow: 0 10px 18px rgba(30,44,59,0.16) !important;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton > button:hover {
          transform: translateY(-2px);
          box-shadow: 0 14px 24px rgba(30,44,59,0.18) !important;
        }
        .results-shell { display: flex; flex-direction: column; gap: 0.85rem; }
        .metric-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.8rem; }
        .metric-card {
          background: white;
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 0.8rem 1rem;
          box-shadow: 0 8px 16px rgba(18,24,27,0.04);
        }
        .metric-card.primary {
          background: var(--steel);
          color: white;
        }
        .metric-label {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          color: var(--ink-soft);
          text-transform: uppercase;
          letter-spacing: 0.08em;
          margin-bottom: 0.35rem;
        }
        .metric-card.primary .metric-label { color: rgba(255,255,255,0.75); }
        .metric-value {
          font-family: 'Space Grotesk', sans-serif;
          font-size: 1.4rem;
          color: var(--steel);
          font-weight: 700;
        }
        .metric-card.primary .metric-value { color: white; }
        .progress-track {
          width: 100%;
          height: 10px;
          border-radius: 999px;
          background: var(--paper-dim);
          overflow: hidden;
          box-shadow: inset 0 1px 2px rgba(18,24,27,0.06);
        }
        .progress-fill {
          height: 100%;
          border-radius: 999px;
          background: linear-gradient(90deg, var(--mint) 0%, var(--green) 100%);
          transition: width 0.7s ease;
        }
        .chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .chip {
          display: inline-flex;
          align-items: center;
          padding: 0.32rem 0.6rem;
          border-radius: 999px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.74rem;
          font-weight: 600;
        }
        .chip.match {
          background: var(--mint-dim);
          color: var(--green);
          border: 1px solid rgba(46,158,91,0.3);
        }
        .chip.miss {
          background: #FFF1E0;
          color: var(--amber);
          border: 1px solid rgba(217,146,46,0.26);
        }
        .chip.neutral {
          background: #F5F6F3;
          color: var(--ink-soft);
          border: 1px solid var(--line);
        }
        .section-title {
          font-family: 'Space Grotesk', sans-serif;
          color: var(--steel);
          font-size: 1.1rem;
          margin: 0 0 0.7rem;
        }
        .breakdown-list { display: flex; flex-direction: column; gap: 0.55rem; margin-top: 0.25rem; }
        .breakdown-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.7rem 0.8rem;
          border: 1px solid var(--line);
          border-radius: 12px;
          background: #FCFCFA;
        }
        .breakdown-row.final { background: var(--mint-dim); }
        .breakdown-label { font-family: 'Inter', sans-serif; color: var(--ink-soft); font-size: 0.92rem; }
        .breakdown-value { font-family: 'IBM Plex Mono', monospace; font-weight: 700; color: var(--steel); }
        .category-block { margin-top: 0.45rem; }
        .category-label {
          font-family: 'IBM Plex Mono', monospace;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-size: 0.72rem;
          color: var(--ink-soft);
          margin-bottom: 0.35rem;
        }
        .explanation-box {
          background: var(--paper-dim);
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 0.85rem 1rem;
          color: var(--ink-soft);
          line-height: 1.6;
          font-family: 'Inter', sans-serif;
        }
        @keyframes scan-line {
          0% { top: 8%; }
          50% { top: 92%; }
          100% { top: 8%; }
        }
        @keyframes chip-fade {
          0% { opacity: 0; transform: translateY(6px); }
          20%, 70% { opacity: 1; transform: translateY(0); }
          100% { opacity: 0; transform: translateY(-6px); }
        }
        @media (max-width: 900px) {
          .hero-scan { grid-template-columns: 1fr; }
          .metric-grid { grid-template-columns: 1fr; }
        }
        @media (prefers-reduced-motion: reduce) {
          .scan-line, .scan-chip { animation: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _escape(text: object) -> str:
    return html_lib.escape(str(text))


def _build_hero_html(analysis: Dict[str, object] | None = None) -> str:
    if analysis is None:
        analysis = st.session_state.get("analysis")

    score = int(round(float(analysis.get("combined_score", 0) or 0))) if analysis else 87
    score = max(0, min(100, score))

    return f"""
    <div class='hero-card'>
      <div class='hero-title'>AI Recruitment Screening Assistant</div>
      <p class='hero-subtitle'>Scan resumes quickly against job requirements with a polished, explainable workflow.</p>
      <div class='hero-scan'>
        <div class='scan-visual'>
          <div class='scan-header'>
            <span>Resume Preview</span>
            <span>01 / 03</span>
          </div>
          <div class='scan-line'></div>
          <div class='scan-chip chip-a'>Python</div>
          <div class='scan-chip chip-b'>NLP</div>
          <div class='scan-chip chip-c'>SQL</div>
          <div class='scan-chip chip-d'>Git</div>
          <div class='scan-lines'>
            <div class='scan-bar' style='width: 58%'></div>
            <div class='scan-bar' style='width: 82%'></div>
            <div class='scan-bar' style='width: 61%'></div>
            <div class='scan-bar' style='width: 73%'></div>
          </div>
        </div>
        <div class='gauge'>
          <div class='gauge-label'>Candidate Score</div>
          <div style='display: flex; align-items: center; width: 100%; gap: 0.5rem;'>
            <div class='gauge-bar-container' style='flex: 1;'>
              <div class='gauge-bar-fill' style='width: {score}%'></div>
            </div>
            <div class='gauge-value'>{score}%</div>
          </div>
          <div class='gauge-caption'>Scanning resume against job description</div>
        </div>
      </div>
    </div>
    """


def _build_results_html(analysis: Dict[str, object]) -> str:
    score = int(round(float(analysis.get("combined_score", 0) or 0)))
    matched_required = analysis.get("matched_required_skills", []) or []
    missing_required = analysis.get("missing_required_skills", []) or []
    matched_preferred = analysis.get("matched_preferred_skills", []) or []
    missing_preferred = analysis.get("missing_preferred_skills", []) or []
    weighted_score = analysis.get("weighted_skill_score", 0)
    tfidf_similarity_score = analysis.get("tfidf_similarity_score", 0)
    combined_score = analysis.get("combined_score", 0)
    extracted_skills = analysis.get("extracted_skills", []) or []
    grouped_skills = group_skills_by_category(extracted_skills, skill_to_category_map)

    def chip_markup(skills: list[str], kind: str) -> str:
        if not skills:
            return ""
        chips = "".join(f"<span class='chip {kind}'>{_escape(skill)}</span>" for skill in skills)
        return f"<div class='chip-row'>{chips}</div>"

    matched_markup = chip_markup(list(matched_required) + list(matched_preferred), "match")
    missing_required_markup = chip_markup(list(missing_required), "miss")
    missing_preferred_markup = chip_markup(list(missing_preferred), "miss")

    skill_category_markup = ""
    for category, skills in grouped_skills.items():
        if not skills:
            continue
        skill_items = "".join(f"<span class='chip neutral'>{_escape(skill)}</span>" for skill in skills)
        skill_category_markup += (
            f"<div class='category-block'><div class='category-label'>{_escape(category)}</div>"
            f"<div class='chip-row'>{skill_items}</div></div>"
        )

    return f"""
    <div class='results-shell'>
      <div class='metric-grid'>
        <div class='metric-card primary'>
          <div class='metric-label'>Overall Match</div>
          <div class='metric-value'>{score}/100</div>
        </div>
        <div class='metric-card'>
          <div class='metric-label'>Required Skills</div>
          <div class='metric-value'>{len(matched_required)}/{len(analysis.get('required_skills', []))}</div>
        </div>
        <div class='metric-card'>
          <div class='metric-label'>Preferred Skills</div>
          <div class='metric-value'>{len(matched_preferred)}/{len(analysis.get('preferred_skills', []))}</div>
        </div>
      </div>
      <div class='progress-track'>
        <div class='progress-fill' style='width:{max(6, score)}%'></div>
      </div>
      <div class='section-card'>
        <div class='section-title'>Matched Skills</div>
        {matched_markup or "<div class='explanation-box'>No matching skills were found.</div>"}
      </div>
      <div class='section-card'>
        <div class='section-title'>Missing Required Skills</div>
        {missing_required_markup or "<div class='explanation-box'>No missing required skills.</div>"}
      </div>
      <div class='section-card'>
        <div class='section-title'>Missing Preferred Skills</div>
        {missing_preferred_markup or "<div class='explanation-box'>No missing preferred skills.</div>"}
      </div>
      <div class='section-card'>
        <div class='section-title'>Score Breakdown</div>
        <div class='breakdown-list'>
          <div class='breakdown-row'><span class='breakdown-label'>Weighted skill score</span><span class='breakdown-value'>{weighted_score}</span></div>
          <div class='breakdown-row'><span class='breakdown-label'>TF-IDF similarity</span><span class='breakdown-value'>{tfidf_similarity_score}</span></div>
          <div class='breakdown-row final'><span class='breakdown-label'>Final combined score</span><span class='breakdown-value'>{combined_score}</span></div>
        </div>
      </div>
      <div class='section-card'>
        <div class='section-title'>Resume Skills by Category</div>
        {skill_category_markup or "<div class='explanation-box'>No extracted skills were found.</div>"}
      </div>
      <div class='section-card'>
        <div class='section-title'>Why this score?</div>
        <div class='explanation-box'>{_escape(build_explanation(analysis))}</div>
      </div>
    </div>
    """


def _extract_candidate_name(resume_text: str, fallback: str) -> str:
    """Return the person's name from the resume header, with a filename fallback."""
    fallback_name = os.path.splitext(os.path.basename(fallback))[0].replace("_", " ")
    if not resume_text:
        return fallback_name

    lines = [re.sub(r"\s+", " ", line).strip(" -|•") for line in resume_text.splitlines()]
    lines = [line for line in lines if line]
    ignored_terms = {
        "resume", "curriculum vitae", "contact", "summary", "profile", "objective",
        "experience", "education", "skills", "projects", "certifications", "references",
        "email", "phone", "linkedin", "github", "website", "address",
    }

    def is_name(candidate: str) -> bool:
        words = candidate.replace(".", "").split()
        if not 2 <= len(words) <= 5 or len(candidate) > 60:
            return False
        lowered = candidate.lower()
        if any(term in lowered for term in ignored_terms) or "@" in candidate or any(char.isdigit() for char in candidate):
            return False
        # Resume names are normally title-cased or all capitals, unlike headings/sentences.
        return all(word[:1].isupper() or word.isupper() for word in words if word)

    # Prefer an explicitly labelled name, e.g. "Name: Ayesha Khan".
    for line in lines[:12]:
        match = re.match(r"^(?:candidate\s*)?name\s*[:\-]\s*(.+)$", line, flags=re.IGNORECASE)
        if match and is_name(match.group(1).strip()):
            return match.group(1).strip()

    # The name is conventionally in the first few lines of a resume.
    for line in lines[:8]:
        if is_name(line):
            return line

    return fallback_name


def assign_recommendation_category(score: float, thresholds: Dict[str, int] | None = None) -> str:
    thresholds = thresholds or {"Strong Match": 85, "Good Match": 70, "Partial Match": 50}
    if score >= thresholds["Strong Match"]:
        return "Strong Match"
    if score >= thresholds["Good Match"]:
        return "Good Match"
    if score >= thresholds["Partial Match"]:
        return "Partial Match"
    return "Weak Match"


def build_ranked_dataframe(batch_results: list[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for index, result in enumerate(batch_results, start=1):
        matched_required_skills = result.get("matched_required_skills", []) or []
        matched_preferred_skills = result.get("matched_preferred_skills", []) or []
        rows.append(
            {
                "candidate": result.get("candidate_id", result.get("filename", f"Candidate {index}")),
                "filename": result.get("filename", f"Candidate {index}"),
                "combined_score": round(float(result.get("combined_score", 0) or 0), 2),
                "skills_matched": len(matched_required_skills) + len(matched_preferred_skills),
                "required_skills_matched": len(matched_required_skills),
                "preferred_skills_matched": len(matched_preferred_skills),
                "recommendation_category": result.get("recommendation_category", "Weak Match"),
            }
        )

    ranked_df = pd.DataFrame(rows)
    if ranked_df.empty:
        return pd.DataFrame(
            columns=["rank", "candidate", "filename", "combined_score", "skills_matched", "required_skills_matched", "preferred_skills_matched", "recommendation_category"]
        )

    ranked_df = ranked_df.sort_values(by="combined_score", ascending=False).reset_index(drop=True)
    ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))
    return ranked_df


def render_sidebar() -> Dict[str, object]:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='widget-label'>Upload Resumes (PDF)</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    st.markdown("<div class='widget-label'>Paste Job Description</div>", unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste Job Description",
        value=(
            "Job Title: AI/ML Intern\n\n"
            "Responsibilities:\n"
            "- Develop machine learning models and data pipelines\n"
            "- Work with Python, SQL, pandas, NumPy, and scikit-learn\n"
            "- Support NLP and deep learning projects\n\n"
            "REQUIRED SKILLS:\n"
            "- Python\n"
            "- SQL\n"
            "- Machine Learning\n"
            "- Pandas\n"
            "- NumPy\n"
            "- Scikit-learn\n"
            "- TensorFlow\n"
            "- PyTorch\n"
            "- Git\n\n"
            "PREFERRED SKILLS:\n"
            "- NLP\n"
            "- Deep Learning\n"
            "- Matplotlib\n"
            "- Seaborn\n"
            "- Docker\n"
            "- AWS"
        ),
        height=320,
        key="job_description",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return {"job_description": job_description, "uploaded_files": uploaded_files}


def show_results(analysis: Dict[str, object]) -> None:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Analysis Results</div>", unsafe_allow_html=True)
    st.markdown(_build_results_html(analysis), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_custom_css()
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "ranked_df" not in st.session_state:
        st.session_state.ranked_df = pd.DataFrame(columns=["rank", "candidate", "filename", "combined_score", "required_matches_total", "preferred_matches_total", "recommendation_category"])
    if "selected_candidate_index" not in st.session_state:
        st.session_state.selected_candidate_index = None
    if "selected_candidate_for_hero" not in st.session_state:
        st.session_state.selected_candidate_for_hero = None

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    # Display hero with selected candidate score if available
    selected_candidate_for_hero = st.session_state.get("selected_candidate_for_hero")
    if selected_candidate_for_hero:
        st.markdown(_build_hero_html(selected_candidate_for_hero), unsafe_allow_html=True)
    else:
        st.markdown(_build_hero_html(st.session_state.get("analysis")), unsafe_allow_html=True)

    input_col, results_col = st.columns([1.05, 0.95], gap="large")
    analysis = st.session_state.get("analysis")

    with input_col:
        settings = render_sidebar()
        job_description = settings.get("job_description", "")
        uploaded_files = settings.get("uploaded_files") or []

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        if st.button("Analyze Resumes", type="primary"):
            if not job_description.strip():
                st.error("Please paste a job description before analyzing.")
            elif not uploaded_files:
                st.error("Please upload at least one PDF resume before analyzing.")
            else:
                batch_results: list[Dict[str, object]] = []
                for uploaded_file in uploaded_files:
                    try:
                        resume_text = extract_text_from_pdf(uploaded_file)
                    except ValueError as exc:
                        st.caption(f"⚠️ {uploaded_file.name}: {exc}")
                        continue

                    try:
                        analysis = analyze_resume(
                            job_description=job_description,
                            resume_text=resume_text,
                            skill_map=flattened_skill_map,
                            preprocessor_func=preprocess_text,
                            skill_weights_map=full_skill_weights_map,
                        )
                    except Exception as exc:
                        st.caption(f"⚠️ {uploaded_file.name}: {exc}")
                        continue

                    candidate_id = _extract_candidate_name(resume_text, uploaded_file.name)
                    result = {
                        "filename": uploaded_file.name,
                        "candidate_id": candidate_id,
                        "combined_score": round(float(analysis.get("combined_score", 0) or 0), 2),
                        "weighted_skill_score": analysis.get("weighted_skill_score", 0),
                        "tfidf_similarity_score": analysis.get("tfidf_similarity_score", 0),
                        "matched_required_skills": analysis.get("matched_required_skills", []) or [],
                        "matched_preferred_skills": analysis.get("matched_preferred_skills", []) or [],
                        "missing_required_skills": analysis.get("missing_required_skills", []) or [],
                        "missing_preferred_skills": analysis.get("missing_preferred_skills", []) or [],
                        "required_skills": analysis.get("required_skills", []) or [],
                        "preferred_skills": analysis.get("preferred_skills", []) or [],
                        "extracted_skills": analysis.get("extracted_skills", []) or [],
                        "explanation": build_explanation(analysis),
                    }
                    result["recommendation_category"] = assign_recommendation_category(result["combined_score"])
                    batch_results.append(result)

                if batch_results:
                    ranked_df = build_ranked_dataframe(batch_results)
                    st.session_state.batch_results = batch_results
                    st.session_state.ranked_df = ranked_df
                    st.session_state.analysis = batch_results[0]
                    # Reset hero/selection so the newly analyzed batch drives the hero card
                    st.session_state.selected_candidate_for_hero = batch_results[0]
                    st.session_state.selected_candidate_index = 0
                else:
                    st.session_state.batch_results = []
                    st.session_state.ranked_df = pd.DataFrame(columns=["rank", "candidate", "filename", "combined_score", "skills_matched", "required_skills_matched", "preferred_skills_matched", "recommendation_category"])
                    st.session_state.analysis = None
                    st.session_state.selected_candidate_for_hero = None
                    st.session_state.selected_candidate_index = None
        st.markdown("</div>", unsafe_allow_html=True)

    with results_col:
        batch_results = st.session_state.get("batch_results", [])
        ranked_df = st.session_state.get("ranked_df")

        if batch_results:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Ranked Candidates</div>", unsafe_allow_html=True)
            if ranked_df is not None and not ranked_df.empty:
                categories = sorted(ranked_df["recommendation_category"].dropna().unique().tolist())
                selected_categories = st.multiselect("Filter by recommendation category", options=categories, default=categories)
                min_score = st.slider("Minimum combined score", min_value=0, max_value=100, value=0)
                filtered_df = ranked_df[
                    (ranked_df["recommendation_category"].isin(selected_categories))
                    & (ranked_df["combined_score"] >= min_score)
                ].copy()

                if filtered_df.empty:
                    st.info("No candidates match the selected filters.")
                else:
                    display_df = filtered_df[["rank", "candidate", "skills_matched", "required_skills_matched", "preferred_skills_matched", "combined_score"]].copy()
                    display_df.columns = ["Rank", "Name", "Skills Matched", "Required Skills Matched", "Preferred Skills Matched", "Combined Score"]
                    st.dataframe(display_df, use_container_width=True)

                    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download as CSV",
                        data=csv_data,
                        file_name="ranked_candidates.csv",
                        mime="text/csv",
                    )

                    selected_index = st.selectbox(
                        "Select candidate",
                        options=list(range(len(filtered_df))),
                        format_func=lambda idx: f"#{int(filtered_df.iloc[idx]['rank'])} — {filtered_df.iloc[idx]['candidate']} — {int(filtered_df.iloc[idx]['combined_score'])}/100",
                        key="candidate_selectbox",
                    )
                    selected_row = filtered_df.iloc[selected_index]
                    selected_candidate = next(
                        (
                            candidate
                            for candidate in batch_results
                            if candidate.get("filename") == selected_row["filename"] and candidate.get("candidate_id") == selected_row["candidate"]
                        ),
                        None,
                    )
                    if selected_candidate is not None:
                        st.session_state.selected_candidate_index = selected_index
                        # Only trigger a rerun the moment the selection actually changes,
                        # so the hero card picks up the new candidate without looping forever.
                        previous_candidate = st.session_state.get("selected_candidate_for_hero")
                        candidate_changed = (
                            previous_candidate is None
                            or previous_candidate.get("filename") != selected_candidate.get("filename")
                            or previous_candidate.get("candidate_id") != selected_candidate.get("candidate_id")
                        )
                        st.session_state.selected_candidate_for_hero = selected_candidate
                        if candidate_changed:
                            st.rerun()

                        show_results(
                            {
                                "required_skills": selected_candidate.get("required_skills", []),
                                "preferred_skills": selected_candidate.get("preferred_skills", []),
                                "extracted_skills": selected_candidate.get("extracted_skills", []),
                                "matched_required_skills": selected_candidate.get("matched_required_skills", []),
                                "missing_required_skills": selected_candidate.get("missing_required_skills", []),
                                "matched_preferred_skills": selected_candidate.get("matched_preferred_skills", []),
                                "missing_preferred_skills": selected_candidate.get("missing_preferred_skills", []),
                                "weighted_skill_score": selected_candidate.get("weighted_skill_score", 0),
                                "tfidf_similarity_score": selected_candidate.get("tfidf_similarity_score", 0),
                                "combined_score": selected_candidate.get("combined_score", 0),
                                "explanation": selected_candidate.get("explanation", ""),
                            }
                        )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class='section-card'>
                  <div class='section-title'>Results Preview</div>
                  <div class='explanation-box'>Your ranked shortlist will appear here once you upload resumes and press Analyze Resumes.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
