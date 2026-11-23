"""Automated Resume Screening Tool — Interactive Streamlit Dashboard.

This application provides a professional graphical interface for screening,
scoring, ranking, and auditing candidate resumes against target job descriptions.

Features:
    - Target job description selection and structured requirements inspection
    - Flexible resume input (built-in synthetic dataset or custom PDF/DOCX/TXT upload)
    - Resilient batch processing with per-candidate error isolation
    - Summary analytics metrics and distribution visualizations
    - Interactive candidate ranking table and in-depth candidate audit views
    - Exportable CSV and self-contained HTML screening reports
    - Responsible AI governance and transparent human-in-the-loop screening notices
"""

import os
import tempfile

# Prevent Matplotlib directory permission warnings on Windows
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Import core backend modules
from src.job_parser import get_standardized_job_profile, load_job_description
from src.matcher import match_candidates
from src.ranking import get_ranking_summary, rank_candidates
from src.report_generator import generate_csv_report, generate_html_report
from src.resume_parser import extract_resume_text
from src.score_calculator import score_candidates

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parent
JOB_DESCRIPTIONS_DIR = PROJECT_ROOT / "data" / "job_descriptions"
SAMPLE_RESUMES_DIR = PROJECT_ROOT / "resumes" / "sample_resumes"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def discover_job_descriptions(directory: Path = JOB_DESCRIPTIONS_DIR) -> Dict[str, Dict[str, Path]]:
    """Discover available job descriptions and requirements files in the data directory.

    Args:
        directory: Directory path to scan for job descriptions.

    Returns:
        Mapping of display title -> {'txt': Path, 'json': Optional[Path]}.
    """
    jobs = {}
    if not directory.exists() or not directory.is_dir():
        return jobs

    for txt_file in sorted(directory.glob("*.txt")):
        stem = txt_file.stem
        # Look for matching requirements JSON
        json_candidate_1 = directory / f"{stem}_requirements.json"
        json_candidate_2 = directory / f"{stem}.json"

        json_path = None
        if json_candidate_1.exists():
            json_path = json_candidate_1
        elif json_candidate_2.exists():
            json_path = json_candidate_2

        # Format display name
        display_name = stem.replace("_", " ").title()
        jobs[display_name] = {
            "txt": txt_file,
            "json": json_path,
        }

    return jobs


def run_screening_pipeline(
    job_txt_path: Path,
    job_json_path: Optional[Path],
    source_type: str,
    uploaded_files: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end resume screening and ranking pipeline.

    Processes resumes with strict per-file error isolation so corrupted or
    unsupported files do not crash the batch.

    Args:
        job_txt_path: Path to plain text job description.
        job_json_path: Path to requirements JSON.
        source_type: 'sample' for built-in dataset, 'upload' for uploaded files.
        uploaded_files: List of Streamlit UploadedFile objects (if source_type == 'upload').

    Returns:
        Dictionary containing:
        - 'ranked_candidates': List of ranked candidate dicts
        - 'summary': Cohort summary dictionary
        - 'job_profile': Loaded standardized job profile
        - 'failed_files': List of dicts describing any parsing failures
    """
    # 1. Load standardized job profile
    job_profile = get_standardized_job_profile(
        description_path=job_txt_path,
        requirements_path=job_json_path,
    )

    valid_resume_paths: List[Path] = []
    resume_texts: Dict[str, str] = {}
    failed_files: List[Dict[str, str]] = []

    # 2. Gather and extract resume documents
    if source_type == "sample":
        sample_files = sorted(SAMPLE_RESUMES_DIR.glob("CAND-*.txt"))
        if not sample_files:
            raise FileNotFoundError(f"No sample resumes found in {SAMPLE_RESUMES_DIR}.")

        for f_path in sample_files:
            try:
                text = extract_resume_text(f_path)
                valid_resume_paths.append(f_path)
                resume_texts[f_path.stem] = text
            except Exception as exc:
                failed_files.append({"filename": f_path.name, "error": str(exc)})

    elif source_type == "upload":
        if not uploaded_files:
            raise ValueError("Please upload at least one candidate resume file (PDF, DOCX, or TXT).")

        # Use temporary directory for safe file handling
        temp_dir = tempfile.mkdtemp(prefix="resume_screening_")
        for u_file in uploaded_files:
            target_path = Path(temp_dir) / u_file.name
            try:
                # Write uploaded bytes to disk for extraction
                with open(target_path, "wb") as f_out:
                    f_out.write(u_file.getbuffer())

                text = extract_resume_text(target_path)
                valid_resume_paths.append(target_path)
                resume_texts[target_path.stem] = text
            except Exception as exc:
                failed_files.append({"filename": u_file.name, "error": str(exc)})

    if not valid_resume_paths:
        return {
            "ranked_candidates": [],
            "summary": get_ranking_summary([]),
            "job_profile": job_profile,
            "failed_files": failed_files,
        }

    # 3. Match candidates against job requirements
    match_results = match_candidates(valid_resume_paths, job_profile)

    # 4. Score candidates using transparent weighted rubrics
    scored_candidates = score_candidates(match_results, resume_texts=resume_texts)

    # 5. Rank candidates deterministically
    ranked_candidates = rank_candidates(scored_candidates)

    # 6. Generate ranking summary analytics
    summary = get_ranking_summary(ranked_candidates)

    return {
        "ranked_candidates": ranked_candidates,
        "summary": summary,
        "job_profile": job_profile,
        "failed_files": failed_files,
    }


def _safe_pyplot(fig):
    """Render Matplotlib figure via Streamlit if in active app runtime, then close."""
    try:
        from streamlit.runtime import exists
        if exists():
            st.pyplot(fig)
    except Exception:
        pass
    finally:
        plt.close(fig)


def render_score_distribution_chart(ranked_candidates: List[Dict[str, Any]]):
    """Render a clean Matplotlib bar chart of candidate scores."""
    if not ranked_candidates:
        return

    cids = [c["candidate_id"] for c in ranked_candidates]
    scores = [c["overall_score"] for c in ranked_candidates]
    colors = []
    for c in ranked_candidates:
        cat = c.get("review_category", "")
        if "Priority" in cat:
            colors.append("#10b981")  # Green
        elif "Review" in cat and "Additional" not in cat:
            colors.append("#3b82f6")  # Blue
        else:
            colors.append("#94a3b8")  # Slate

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(cids, scores, color=colors, width=0.55, edgecolor="#334155", linewidth=0.8)

    # Threshold guidelines
    ax.axhline(80, color="#10b981", linestyle="--", linewidth=1.2, label="Priority Review Tier (≥80)")
    ax.axhline(60, color="#3b82f6", linestyle="--", linewidth=1.2, label="Review Tier (≥60)")

    ax.set_ylabel("Overall Screening Score (0-100)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("Candidate Screening Score Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    # Add score labels on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.8,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    _safe_pyplot(fig)


def render_category_donut_chart(category_counts: Dict[str, int]):
    """Render a clean Matplotlib donut chart of review categories."""
    if not category_counts or sum(category_counts.values()) == 0:
        return

    labels = list(category_counts.keys())
    counts = list(category_counts.values())

    color_map = {
        "Priority Review": "#10b981",
        "Review": "#3b82f6",
        "Additional Review": "#94a3b8",
    }
    colors = [color_map.get(lbl, "#cbd5e1") for lbl in labels]

    fig, ax = plt.subplots(figsize=(5, 3.8))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )

    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(9)

    ax.set_title("Candidate Review Tier Proportions", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    _safe_pyplot(fig)


def render_component_breakdown_chart(ranked_candidates: List[Dict[str, Any]]):
    """Render a stacked bar chart comparing score contributions for top candidates."""
    if not ranked_candidates:
        return

    top_candidates = ranked_candidates[:5]  # Top 5 candidates
    cids = [c["candidate_id"] for c in top_candidates]

    req_pts = [c.get("weighted_contributions", {}).get("required_skills", 0.0) for c in top_candidates]
    pref_pts = [c.get("weighted_contributions", {}).get("preferred_skills", 0.0) for c in top_candidates]
    exp_pts = [c.get("weighted_contributions", {}).get("experience", 0.0) for c in top_candidates]
    sim_pts = [c.get("weighted_contributions", {}).get("text_similarity", 0.0) for c in top_candidates]
    edu_pts = [c.get("weighted_contributions", {}).get("education", 0.0) for c in top_candidates]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(cids))
    width = 0.50

    p1 = ax.bar(x, req_pts, width, label="Req Skills (35%)", color="#2563eb")
    p2 = ax.bar(x, pref_pts, width, bottom=req_pts, label="Pref Skills (15%)", color="#60a5fa")
    bottom_exp = np.array(req_pts) + np.array(pref_pts)
    p3 = ax.bar(x, exp_pts, width, bottom=bottom_exp, label="Experience (20%)", color="#10b981")
    bottom_sim = bottom_exp + np.array(exp_pts)
    p4 = ax.bar(x, sim_pts, width, bottom=bottom_sim, label="Text Sim (20%)", color="#f59e0b")
    bottom_edu = bottom_sim + np.array(sim_pts)
    p5 = ax.bar(x, edu_pts, width, bottom=bottom_edu, label="Education (10%)", color="#8b5cf6")

    ax.set_ylabel("Weighted Points (Sum = Overall Score)", fontsize=9.5, fontweight="bold")
    ax.set_title("Component Contribution Comparison (Top Candidates)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cids, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    _safe_pyplot(fig)


def main():
    """Main Streamlit application layout and orchestration."""
    # -------------------------------------------------------------------------
    # A. Page Configuration
    # -------------------------------------------------------------------------
    st.set_page_config(
        page_title="Automated Resume Screening Tool",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling enhancements
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 4px;
        }
        .sub-header {
            font-size: 1.0rem;
            color: #475569;
            margin-bottom: 20px;
        }
        .ai-banner {
            background-color: #eff6ff;
            border-left: 5px solid #2563eb;
            border-radius: 6px;
            padding: 12px 18px;
            margin-bottom: 24px;
            font-size: 0.88rem;
            color: #1e3a8a;
            line-height: 1.5;
        }
        .stMetric {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .skill-tag-match {
            display: inline-block;
            background-color: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
            border-radius: 9999px;
            padding: 2px 10px;
            font-size: 0.80rem;
            font-weight: 600;
            margin: 2px 4px 2px 0;
        }
        .skill-tag-missing {
            display: inline-block;
            background-color: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
            border-radius: 9999px;
            padding: 2px 10px;
            font-size: 0.80rem;
            font-weight: 600;
            margin: 2px 4px 2px 0;
        }
        .skill-tag-pref {
            display: inline-block;
            background-color: #eff6ff;
            color: #1e40af;
            border: 1px solid #bfdbfe;
            border-radius: 9999px;
            padding: 2px 10px;
            font-size: 0.80rem;
            font-weight: 600;
            margin: 2px 4px 2px 0;
        }
        .badge-priority {
            background-color: #d1fae5;
            color: #065f46;
            padding: 3px 10px;
            border-radius: 9999px;
            font-weight: 700;
        }
        .badge-review {
            background-color: #dbeafe;
            color: #1e40af;
            padding: 3px 10px;
            border-radius: 9999px;
            font-weight: 700;
        }
        .badge-additional {
            background-color: #f1f5f9;
            color: #475569;
            padding: 3px 10px;
            border-radius: 9999px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # B. Header & Responsible AI Notice
    # -------------------------------------------------------------------------
    st.markdown('<div class="main-header">🎯 Automated Resume Screening Tool</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Transparent, explainable resume-to-job matching, rubric evaluation, '
        'and deterministic candidate ranking.</div>',
        unsafe_allow_html=True,
    )

    # Mandatory Responsible AI Disclaimer
    st.markdown(
        """
        <div class="ai-banner">
            <strong>⚖️ Responsible AI Screening Notice:</strong><br>
            Scores indicate alignment with the selected job description and configured screening criteria.
            They are not predictions of job performance or hiring decisions. This tool is intended to support
            human review. No candidate should be automatically rejected based on these scores.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # C. Sidebar Controls
    # -------------------------------------------------------------------------
    st.sidebar.title("Screening Controls")

    # 1. Job Description Discovery
    available_jobs = discover_job_descriptions()
    if not available_jobs:
        st.sidebar.error("No job description (.txt) files found in data/job_descriptions.")
        return

    job_choice = st.sidebar.selectbox("1. Target Job Description", list(available_jobs.keys()))
    selected_job_info = available_jobs[job_choice]

    # Display target job details in sidebar
    with st.sidebar.expander("📋 View Job Requirements", expanded=False):
        try:
            profile_peek = get_standardized_job_profile(
                description_path=selected_job_info["txt"],
                requirements_path=selected_job_info["json"],
            )
            st.markdown(f"**Title:** {profile_peek.get('job_title', 'Not specified')}")
            st.markdown(f"**Experience:** {profile_peek.get('minimum_experience_years', 0)}–{profile_peek.get('preferred_experience_years', 1)} years")
            st.markdown(f"**Required Skills:** {', '.join(profile_peek.get('required_skills', []))}")
            st.markdown(f"**Preferred Skills:** {', '.join(profile_peek.get('preferred_skills', []))}")
            st.markdown(f"**Education:** {profile_peek.get('education_preferred', 'Not specified')}")
        except Exception as exc:
            st.error(f"Failed to inspect job requirements: {exc}")

    st.sidebar.markdown("---")

    # 2. Resume Source Selection
    source_option = st.sidebar.radio(
        "2. Candidate Resumes Source",
        ["Sample Dataset (8 Synthetic Resumes)", "Upload Custom Resumes"],
        index=0,
    )

    uploaded_files = None
    if source_option == "Upload Custom Resumes":
        uploaded_files = st.sidebar.file_uploader(
            "Upload Resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Select one or more candidate resume documents to screen.",
        )
        if uploaded_files:
            st.sidebar.success(f"{len(uploaded_files)} file(s) staged for screening.")

    st.sidebar.markdown("---")

    # 3. Execution Action Button
    run_screening = st.sidebar.button("🚀 Run Candidate Screening", type="primary", use_container_width=True)

    # Initialize Session State
    if "screening_results" not in st.session_state:
        st.session_state["screening_results"] = None
    if "active_job" not in st.session_state:
        st.session_state["active_job"] = None

    # Invalidate results if user changed the target job
    if st.session_state["active_job"] != job_choice and st.session_state["screening_results"] is not None:
        st.session_state["screening_results"] = None
        st.session_state["active_job"] = job_choice

    # -------------------------------------------------------------------------
    # D. Execute Pipeline
    # -------------------------------------------------------------------------
    if run_screening:
        with st.spinner("Processing resumes, calculating match scores, and ranking candidates..."):
            try:
                src_type = "sample" if "Sample Dataset" in source_option else "upload"
                pipeline_output = run_screening_pipeline(
                    job_txt_path=selected_job_info["txt"],
                    job_json_path=selected_job_info["json"],
                    source_type=src_type,
                    uploaded_files=uploaded_files,
                )
                st.session_state["screening_results"] = pipeline_output
                st.session_state["active_job"] = job_choice
                st.success("Screening successfully executed!")
            except Exception as exc:
                st.error(f"Screening pipeline error: {exc}")
                return

    # -------------------------------------------------------------------------
    # E. Display Dashboard Results
    # -------------------------------------------------------------------------
    results = st.session_state["screening_results"]

    if not results:
        st.info("👈 Select a target job description and resume source in the sidebar, then click **Run Candidate Screening**.")
        return

    ranked_candidates = results["ranked_candidates"]
    summary = results["summary"]
    job_profile = results["job_profile"]
    failed_files = results["failed_files"]

    # 1. Processing Errors / Warnings (if any)
    if failed_files:
        st.warning(f"⚠️ {len(failed_files)} file(s) encountered parsing errors and were excluded from scoring:")
        failed_df = pd.DataFrame(failed_files)
        st.dataframe(failed_df, use_container_width=True, hide_index=True)

    if not ranked_candidates:
        st.error("No valid candidate resumes could be scored. Please check your uploaded files.")
        return

    # 2. Summary Metric Cards
    st.markdown("### 📊 Cohort Screening Summary")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Total Candidates", summary.get("total_candidates", len(ranked_candidates)))
    with m_col2:
        high = summary.get("highest_score")
        st.metric("Highest Score", f"{high:.2f}" if high is not None else "N/A")
    with m_col3:
        low = summary.get("lowest_score")
        st.metric("Lowest Score", f"{low:.2f}" if low is not None else "N/A")
    with m_col4:
        avg = summary.get("mean_score")
        st.metric("Cohort Mean", f"{avg:.2f}" if avg is not None else "N/A")
    with m_col5:
        med = summary.get("median_score")
        st.metric("Cohort Median", f"{med:.2f}" if med is not None else "N/A")

    # Review Category Distribution Pills
    cat_counts = summary.get("review_category_counts", {})
    p_cnt = cat_counts.get("Priority Review", 0)
    r_cnt = cat_counts.get("Review", 0)
    a_cnt = cat_counts.get("Additional Review", 0)

    st.markdown(
        f"""
        <div style="display: flex; gap: 16px; margin: 12px 0 24px 0; flex-wrap: wrap;">
            <span class="badge-priority">Priority Review (≥80): {p_cnt}</span>
            <span class="badge-review">Review (60–79): {r_cnt}</span>
            <span class="badge-additional">Additional Review (&lt;60): {a_cnt}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # F. Ranked Candidate Overview Table
    # -------------------------------------------------------------------------
    st.markdown("### 🏆 Candidate Screening Leaderboard")

    table_data = []
    for c in ranked_candidates:
        table_data.append({
            "Rank": c["rank"],
            "Candidate ID": c["candidate_id"],
            "Overall Score": f"{c['overall_score']:.2f}",
            "Review Category": c["review_category"],
            "Required Match %": f"{c['required_skill_match_percentage']:.1f}%",
            "Preferred Match %": f"{c.get('preferred_skill_match_percentage', c.get('preferred_skill_score', 0.0)):.1f}%",
            "Skill Coverage %": f"{c['overall_skill_coverage_percentage']:.1f}%",
            "Text Sim %": f"{c['text_similarity_percentage']:.1f}%",
            "Exp Score": f"{c.get('experience_score', 0.0):.1f}",
            "Edu Score": f"{c.get('education_score', 0.0):.1f}",
        })

    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # G. Candidate Detail View
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔍 Candidate Profile & Evaluation Audit")

    candidate_options = [
        f"Rank {c['rank']} | {c['candidate_id']} ({c['review_category']} — {c['overall_score']:.2f}/100)"
        for c in ranked_candidates
    ]
    selected_option = st.selectbox("Select Candidate for In-Depth Review", candidate_options)

    # Find the selected candidate record
    sel_index = candidate_options.index(selected_option)
    cand = ranked_candidates[sel_index]

    # Candidate Detail Card
    detail_box = st.container()
    with detail_box:
        d_col1, d_col2 = st.columns([3, 1])
        with d_col1:
            st.markdown(f"#### Candidate: `{cand['candidate_id']}`")
            st.markdown(f"**Review Tier:** `{cand['review_category']}` &nbsp;|&nbsp; **Display Order:** #{cand.get('display_order', cand['rank'])}")
        with d_col2:
            st.metric("Overall Score", f"{cand['overall_score']:.2f} / 100")

        # Score Contributions Breakdown
        st.markdown("**Weighted Component Contributions:**")
        wc = cand.get("weighted_contributions", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Req Skills (35%)", f"+{wc.get('required_skills', 0.0):.2f} pts")
        with c2:
            st.metric("Pref Skills (15%)", f"+{wc.get('preferred_skills', 0.0):.2f} pts")
        with c3:
            st.metric("Experience (20%)", f"+{wc.get('experience', 0.0):.2f} pts")
        with c4:
            st.metric("Text Sim (20%)", f"+{wc.get('text_similarity', 0.0):.2f} pts")
        with c5:
            st.metric("Education (10%)", f"+{wc.get('education', 0.0):.2f} pts")

        # Skills Match & Gaps
        sk_col1, sk_col2 = st.columns(2)
        with sk_col1:
            st.markdown("##### Required Skills Audit (Target: Core Competencies)")
            matched_req = cand.get("matched_required_skills") or cand.get("raw_match_result", {}).get("matched_required_skills", [])
            missing_req = cand.get("missing_required_skills") or cand.get("raw_match_result", {}).get("missing_required_skills", [])

            if matched_req:
                st.markdown("**Matched:** " + "".join(f'<span class="skill-tag-match">✓ {s}</span>' for s in matched_req), unsafe_allow_html=True)
            else:
                st.markdown("**Matched:** _None_")

            if missing_req:
                st.markdown("**Missing:** " + "".join(f'<span class="skill-tag-missing">✗ {s}</span>' for s in missing_req), unsafe_allow_html=True)
            else:
                st.markdown("**Missing:** _None (100% Core Match)_")

        with sk_col2:
            st.markdown("##### Preferred Skills Audit (Bonus Capabilities)")
            matched_pref = cand.get("matched_preferred_skills") or cand.get("raw_match_result", {}).get("matched_preferred_skills", [])
            missing_pref = cand.get("missing_preferred_skills") or cand.get("raw_match_result", {}).get("missing_preferred_skills", [])

            if matched_pref:
                st.markdown("**Matched:** " + "".join(f'<span class="skill-tag-pref">★ {s}</span>' for s in matched_pref), unsafe_allow_html=True)
            else:
                st.markdown("**Matched:** _None_")

            if missing_pref:
                st.markdown("**Unmatched:** " + ", ".join(missing_pref))
            else:
                st.markdown("**Unmatched:** _None_")

        # Qualitative Rubric Explanations
        st.markdown("##### Narrative Evaluation Notes")
        st.info(cand.get("score_explanation", "No score explanation provided."))

    # -------------------------------------------------------------------------
    # H. Visualizations
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📈 Screening Analytics & Visualizations")

    tab_scores, tab_categories, tab_compare = st.tabs([
        "Score Distribution",
        "Category Breakdown",
        "Top Candidate Component Comparison",
    ])

    with tab_scores:
        render_score_distribution_chart(ranked_candidates)

    with tab_categories:
        render_category_donut_chart(cat_counts)

    with tab_compare:
        render_component_breakdown_chart(ranked_candidates)

    # -------------------------------------------------------------------------
    # I. Export & Download Reports
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 💾 Export Screening Reports")
    st.write("Generate and download official screening reports in spreadsheet (CSV) or standalone interactive HTML format.")

    with tempfile.TemporaryDirectory() as report_tmp_dir:
        job_title = job_profile.get("job_title", job_choice)
        csv_path = generate_csv_report(ranked_candidates, output_dir=report_tmp_dir)
        html_path = generate_html_report(
            ranked_candidates,
            output_dir=report_tmp_dir,
            job_title=job_title,
            summary=summary,
        )

        with open(csv_path, "rb") as f_csv:
            csv_bytes = f_csv.read()
        with open(html_path, "rb") as f_html:
            html_bytes = f_html.read()

        dl_col1, dl_col2, _ = st.columns([1, 1, 2])
        with dl_col1:
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_bytes,
                file_name="candidate_ranking_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="📄 Download HTML Report",
                data=html_bytes,
                file_name="candidate_ranking_report.html",
                mime="text/html",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
