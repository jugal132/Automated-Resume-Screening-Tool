"""Report Generation Module for Automated Resume Screening Tool.

This module converts scored and ranked candidate records into professional,
exportable CSV and standalone HTML reports.

Features:
    - CSV Report: Spreadsheet-compatible (UTF-8 with BOM), one row per candidate,
      including ranks, matching percentages, rubric evaluations, skill lists,
      and contribution breakdowns.
    - HTML Report: Standalone, responsive, beautifully styled, fully offline
      (no external CDN or JS dependencies). Includes summary metric cards,
      review category distribution, candidate ranking table, per-candidate skill
      breakdowns, and a Responsible AI governance disclaimer.
    - Strict Security: HTML escaping on all dynamic candidate data and metadata to
      preclude XSS/injection attacks.
    - Non-Mutating: Does not calculate, alter, or invent candidate scores.
    - Safe Handling: Gracefully handles empty candidate cohorts and missing optional fields.
"""

from datetime import datetime, timezone
import html
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from src.ranking import get_ranking_summary


# Standard spreadsheet column headers for CSV reports
CSV_HEADERS = [
    "Candidate ID",
    "Rank",
    "Display Order",
    "Overall Score",
    "Review Category",
    "Required Skill Match %",
    "Preferred Skill Match %",
    "Skill Coverage %",
    "Experience Score",
    "Education Score",
    "Text Similarity %",
    "Matched Required Skills",
    "Missing Required Skills",
    "Matched Preferred Skills",
    "Missing Preferred Skills",
    "Required Skills Contribution",
    "Preferred Skills Contribution",
    "Experience Contribution",
    "Text Similarity Contribution",
    "Education Contribution",
    "Score Explanation",
]


def _format_float(val: Any) -> str:
    """Format numeric values to 2 decimal places, or return 'N/A'."""
    if val is None:
        return "N/A"
    try:
        f_val = float(val)
        return f"{f_val:.2f}"
    except (ValueError, TypeError):
        return str(val)


def _format_skills(skills: Any) -> str:
    """Format skill lists into a consistent semicolon-separated string."""
    if not skills:
        return "None"
    if isinstance(skills, (list, tuple, set)):
        items = [str(s).strip() for s in skills if str(s).strip()]
        return "; ".join(items) if items else "None"
    return str(skills)


def _get_metric(candidate: Dict[str, Any], metric_type: str) -> Any:
    """Extract a matching metric from either top-level or nested raw_match_result."""
    raw = candidate.get("raw_match_result", {}) if isinstance(candidate.get("raw_match_result"), dict) else {}

    if metric_type == "required_skill":
        for k in ["required_skill_match_percentage", "required_skill_score"]:
            if k in candidate:
                return candidate[k]
        return raw.get("required_skill_match_percentage")

    elif metric_type == "preferred_skill":
        for k in ["preferred_skill_match_percentage", "preferred_skill_score"]:
            if k in candidate:
                return candidate[k]
        return raw.get("preferred_skill_match_percentage")

    elif metric_type == "coverage":
        for k in ["overall_skill_coverage_percentage", "skill_coverage_percentage"]:
            if k in candidate:
                return candidate[k]
        return raw.get("overall_skill_coverage_percentage")

    elif metric_type == "similarity":
        for k in ["text_similarity_percentage", "text_similarity_score"]:
            if k in candidate:
                return candidate[k]
        return raw.get("text_similarity_percentage")

    return None


def _get_skill_list(candidate: Dict[str, Any], key: str) -> List[str]:
    """Extract a list of skills from top-level or nested raw_match_result."""
    if key in candidate and isinstance(candidate[key], (list, tuple, set)):
        return list(candidate[key])
    raw = candidate.get("raw_match_result", {})
    if isinstance(raw, dict) and key in raw and isinstance(raw[key], (list, tuple, set)):
        return list(raw[key])
    return []


def _resolve_output_path(
    output_path: Optional[Union[str, Path]],
    output_dir: Optional[Union[str, Path]],
    filename: str,
) -> Path:
    """Resolve destination file path and ensure target directory exists.

    Args:
        output_path: Explicit target file path (takes precedence if provided).
        output_dir: Target directory path (used with filename if output_path is None).
        filename: Fallback filename.

    Returns:
        Resolved Path object.

    Raises:
        OSError: If destination directory cannot be created.
    """
    if output_path is not None:
        target = Path(output_path)
    else:
        directory = Path(output_dir) if output_dir is not None else Path("outputs")
        target = directory / filename

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise OSError(f"Failed to create output directory '{target.parent}': {exc}") from exc

    return target


def validate_report_candidates(ranked_candidates: Any) -> None:
    """Validate candidate cohort input for report generation.

    Args:
        ranked_candidates: Candidate list to validate.

    Raises:
        TypeError: If ranked_candidates is not a list or items are not dicts.
        ValueError: If candidate_id is missing or empty.
    """
    if not isinstance(ranked_candidates, list):
        raise TypeError(
            f"ranked_candidates must be a list of dicts, got {type(ranked_candidates).__name__}."
        )

    for idx, cand in enumerate(ranked_candidates):
        if not isinstance(cand, dict):
            raise TypeError(
                f"Candidate at index {idx} must be a dict, got {type(cand).__name__}."
            )
        cid = cand.get("candidate_id")
        if cid is None or not str(cid).strip():
            raise ValueError(
                f"Candidate at index {idx} is missing a valid non-empty 'candidate_id'."
            )


def generate_csv_report(
    ranked_candidates: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "candidate_ranking_report.csv",
) -> Path:
    """Generate a spreadsheet-friendly CSV report for ranked candidates.

    Encoding: UTF-8 with BOM (utf-8-sig) for native Microsoft Excel compatibility.

    Args:
        ranked_candidates: List of ranked and scored candidate dictionaries.
        output_path: Optional exact output file path.
        output_dir: Optional output directory (defaults to 'outputs').
        filename: Filename if output_path is not specified.

    Returns:
        Path to the generated CSV file.

    Raises:
        TypeError: If ranked_candidates is not a list of dicts.
        ValueError: If a candidate has invalid or missing candidate_id.
        OSError: If file writing or directory creation fails.
    """
    validate_report_candidates(ranked_candidates)
    target_path = _resolve_output_path(output_path, output_dir, filename)

    import csv

    try:
        with open(target_path, mode="w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADERS)

            for cand in ranked_candidates:
                cid = str(cand.get("candidate_id", "Unknown")).strip()
                rank = cand.get("rank", "N/A")
                order = cand.get("display_order", "N/A")
                overall = _format_float(cand.get("overall_score"))
                cat = str(cand.get("review_category", "N/A"))

                req_pct = _format_float(_get_metric(cand, "required_skill"))
                pref_pct = _format_float(_get_metric(cand, "preferred_skill"))
                cov_pct = _format_float(_get_metric(cand, "coverage"))
                exp_score = _format_float(cand.get("experience_score"))
                edu_score = _format_float(cand.get("education_score"))
                sim_pct = _format_float(_get_metric(cand, "similarity"))

                matched_req = _format_skills(_get_skill_list(cand, "matched_required_skills"))
                missing_req = _format_skills(_get_skill_list(cand, "missing_required_skills"))
                matched_pref = _format_skills(_get_skill_list(cand, "matched_preferred_skills"))
                missing_pref = _format_skills(_get_skill_list(cand, "missing_preferred_skills"))

                contributions = (
                    cand.get("weighted_contributions", {})
                    if isinstance(cand.get("weighted_contributions"), dict)
                    else {}
                )
                contrib_req = _format_float(contributions.get("required_skills"))
                contrib_pref = _format_float(contributions.get("preferred_skills"))
                contrib_exp = _format_float(contributions.get("experience"))
                contrib_sim = _format_float(contributions.get("text_similarity"))
                contrib_edu = _format_float(contributions.get("education"))

                explanation = str(cand.get("score_explanation", "N/A"))

                row = [
                    cid,
                    rank,
                    order,
                    overall,
                    cat,
                    req_pct,
                    pref_pct,
                    cov_pct,
                    exp_score,
                    edu_score,
                    sim_pct,
                    matched_req,
                    missing_req,
                    matched_pref,
                    missing_pref,
                    contrib_req,
                    contrib_pref,
                    contrib_exp,
                    contrib_sim,
                    contrib_edu,
                    explanation,
                ]
                writer.writerow(row)
    except Exception as exc:
        raise OSError(f"Failed to write CSV report to '{target_path}': {exc}") from exc

    return target_path


def _get_badge_class(category: str) -> str:
    """Return CSS class for review category badge."""
    cat_lower = str(category).lower()
    if "priority" in cat_lower:
        return "badge-priority"
    elif "additional" in cat_lower:
        return "badge-additional"
    elif "review" in cat_lower:
        return "badge-review"
    return "badge-neutral"


def generate_html_report(
    ranked_candidates: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: str = "candidate_ranking_report.html",
    job_title: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate a standalone, self-contained, responsive HTML report.

    Security:
        All candidate-supplied texts, IDs, job titles, and explanations are
        strictly escaped using html.escape(..., quote=True).

    Args:
        ranked_candidates: List of ranked and scored candidate dictionaries.
        output_path: Optional exact output file path.
        output_dir: Optional output directory (defaults to 'outputs').
        filename: Filename if output_path is not specified.
        job_title: Optional target role title for the report header.
        summary: Optional pre-computed ranking summary dictionary. If None,
                 computed via src.ranking.get_ranking_summary.

    Returns:
        Path to the generated HTML file.

    Raises:
        TypeError: If ranked_candidates is not a list of dicts or summary is not a dict.
        ValueError: If a candidate has invalid or missing candidate_id.
        OSError: If file writing fails.
    """
    validate_report_candidates(ranked_candidates)
    if summary is not None and not isinstance(summary, dict):
        raise TypeError(f"summary must be a dict, got {type(summary).__name__}.")

    target_path = _resolve_output_path(output_path, output_dir, filename)

    # 1. Resolve summary metrics safely
    if summary is None:
        cohort_summary = get_ranking_summary(ranked_candidates)
    else:
        cohort_summary = summary

    total_candidates = cohort_summary.get("total_candidates", len(ranked_candidates))
    highest_score = cohort_summary.get("highest_score", cohort_summary.get("highest_overall_score"))
    lowest_score = cohort_summary.get("lowest_score", cohort_summary.get("lowest_overall_score"))
    mean_score = cohort_summary.get("mean_score", cohort_summary.get("mean_overall_score"))
    median_score = cohort_summary.get("median_score", cohort_summary.get("median_overall_score"))

    cat_counts = (
        cohort_summary.get("review_category_counts")
        or cohort_summary.get("category_counts")
        or {}
    )

    # 2. Format Header Metadata
    gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    esc_job_title = html.escape(str(job_title), quote=True) if job_title else None
    report_title = "Automated Resume Screening Report"

    # 3. Build Table Rows
    table_rows_html = []
    if not ranked_candidates:
        table_rows_html.append(
            '<tr><td colspan="10" class="empty-state">No candidate records available.</td></tr>'
        )
    else:
        for cand in ranked_candidates:
            cid = html.escape(str(cand.get("candidate_id", "Unknown")).strip(), quote=True)
            rank = html.escape(str(cand.get("rank", "N/A")), quote=True)
            overall = html.escape(_format_float(cand.get("overall_score")), quote=True)
            cat = str(cand.get("review_category", "N/A"))
            badge_cls = _get_badge_class(cat)
            esc_cat = html.escape(cat, quote=True)

            req_pct = html.escape(_format_float(_get_metric(cand, "required_skill")), quote=True)
            pref_pct = html.escape(_format_float(_get_metric(cand, "preferred_skill")), quote=True)
            cov_pct = html.escape(_format_float(_get_metric(cand, "coverage")), quote=True)
            exp_score = html.escape(_format_float(cand.get("experience_score")), quote=True)
            edu_score = html.escape(_format_float(cand.get("education_score")), quote=True)
            sim_pct = html.escape(_format_float(_get_metric(cand, "similarity")), quote=True)

            row_html = f"""
            <tr>
                <td class="text-center font-bold">{rank}</td>
                <td class="font-bold">{cid}</td>
                <td class="text-center score-cell">{overall}</td>
                <td><span class="badge {badge_cls}">{esc_cat}</span></td>
                <td class="text-center">{req_pct}%</td>
                <td class="text-center">{pref_pct}%</td>
                <td class="text-center">{cov_pct}%</td>
                <td class="text-center">{sim_pct}%</td>
                <td class="text-center">{exp_score}</td>
                <td class="text-center">{edu_score}</td>
            </tr>"""
            table_rows_html.append(row_html)

    # 4. Build Candidate Detail Cards
    detail_cards_html = []
    if not ranked_candidates:
        detail_cards_html.append(
            '<div class="card"><p class="empty-state">No candidate profiles to display.</p></div>'
        )
    else:
        for cand in ranked_candidates:
            cid = html.escape(str(cand.get("candidate_id", "Unknown")).strip(), quote=True)
            rank = html.escape(str(cand.get("rank", "N/A")), quote=True)
            overall = html.escape(_format_float(cand.get("overall_score")), quote=True)
            cat = str(cand.get("review_category", "N/A"))
            badge_cls = _get_badge_class(cat)
            esc_cat = html.escape(cat, quote=True)
            explanation = html.escape(str(cand.get("score_explanation", "No score explanation provided.")), quote=True)

            matched_req = _get_skill_list(cand, "matched_required_skills")
            missing_req = _get_skill_list(cand, "missing_required_skills")
            matched_pref = _get_skill_list(cand, "matched_preferred_skills")
            missing_pref = _get_skill_list(cand, "missing_preferred_skills")

            def _render_badges(items: List[str], css_class: str, empty_text: str = "None") -> str:
                if not items:
                    return f'<span class="text-muted font-italic">{html.escape(empty_text)}</span>'
                return "".join(
                    f'<span class="badge {css_class}">{html.escape(str(s), quote=True)}</span>'
                    for s in items
                )

            matched_req_badges = _render_badges(matched_req, "badge-skill-match")
            missing_req_badges = _render_badges(missing_req, "badge-skill-missing")
            matched_pref_badges = _render_badges(matched_pref, "badge-skill-pref")
            missing_pref_badges = _render_badges(missing_pref, "badge-neutral")

            # Weighted Contributions
            wc = (
                cand.get("weighted_contributions", {})
                if isinstance(cand.get("weighted_contributions"), dict)
                else {}
            )
            c_req = html.escape(_format_float(wc.get("required_skills")), quote=True)
            c_pref = html.escape(_format_float(wc.get("preferred_skills")), quote=True)
            c_exp = html.escape(_format_float(wc.get("experience")), quote=True)
            c_sim = html.escape(_format_float(wc.get("text_similarity")), quote=True)
            c_edu = html.escape(_format_float(wc.get("education")), quote=True)

            card_html = f"""
            <div class="candidate-card" id="candidate-{cid}">
                <div class="card-header">
                    <div class="candidate-title">
                        <span class="rank-pill">Rank {rank}</span>
                        <h3>{cid}</h3>
                        <span class="badge {badge_cls}">{esc_cat}</span>
                    </div>
                    <div class="score-display">
                        <span class="score-num">{overall}</span>
                        <span class="score-sub">/ 100</span>
                    </div>
                </div>
                <div class="card-body">
                    <div class="contrib-grid">
                        <div class="contrib-item">
                            <span class="contrib-label">Req Skills (35%)</span>
                            <span class="contrib-val">+{c_req} pts</span>
                        </div>
                        <div class="contrib-item">
                            <span class="contrib-label">Pref Skills (15%)</span>
                            <span class="contrib-val">+{c_pref} pts</span>
                        </div>
                        <div class="contrib-item">
                            <span class="contrib-label">Experience (20%)</span>
                            <span class="contrib-val">+{c_exp} pts</span>
                        </div>
                        <div class="contrib-item">
                            <span class="contrib-label">Text Sim (20%)</span>
                            <span class="contrib-val">+{c_sim} pts</span>
                        </div>
                        <div class="contrib-item">
                            <span class="contrib-label">Education (10%)</span>
                            <span class="contrib-val">+{c_edu} pts</span>
                        </div>
                    </div>
                    <div class="skills-section">
                        <div class="skill-group">
                            <strong>Matched Required Skills:</strong>
                            <div class="badge-container">{matched_req_badges}</div>
                        </div>
                        <div class="skill-group">
                            <strong>Missing Required Skills:</strong>
                            <div class="badge-container">{missing_req_badges}</div>
                        </div>
                        <div class="skill-group">
                            <strong>Matched Preferred Skills:</strong>
                            <div class="badge-container">{matched_pref_badges}</div>
                        </div>
                        <div class="skill-group">
                            <strong>Missing Preferred Skills:</strong>
                            <div class="badge-container">{missing_pref_badges}</div>
                        </div>
                    </div>
                    <div class="explanation-box">
                        <strong>Evaluation Note:</strong> {explanation}
                    </div>
                </div>
            </div>"""
            detail_cards_html.append(card_html)

    # 5. Build Category Distribution Badges
    cat_distribution_pills = []
    for cat_name, count in sorted(cat_counts.items()):
        esc_name = html.escape(str(cat_name), quote=True)
        b_class = _get_badge_class(cat_name)
        pct = (count / total_candidates * 100) if total_candidates > 0 else 0.0
        cat_distribution_pills.append(
            f'<div class="cat-pill-item">'
            f'<span class="badge {b_class}">{esc_name}</span>'
            f'<span class="cat-pill-count">{count} ({pct:.1f}%)</span>'
            f'</div>'
        )
    cat_distribution_html = "".join(cat_distribution_pills) if cat_distribution_pills else '<span class="text-muted">None</span>'

    # 6. Compose Full Self-Contained HTML Document
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --primary: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.5;
            color: var(--text-main);
            background-color: var(--bg-color);
            padding: 24px 16px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .header h1 {{
            font-size: 1.85rem;
            color: #0f172a;
            margin-bottom: 6px;
        }}
        .header .meta {{
            color: var(--text-muted);
            font-size: 0.92rem;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
        }}
        .header .meta span strong {{
            color: var(--text-main);
        }}
        .section-title {{
            font-size: 1.25rem;
            color: #0f172a;
            margin: 28px 0 14px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        /* Summary Grid */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}
        .metric-label {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 1.85rem;
            font-weight: 700;
            color: #0f172a;
        }}
        .distribution-box {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 24px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 20px;
        }}
        .distribution-title {{
            font-size: 0.95rem;
            font-weight: 600;
            color: #334155;
        }}
        .cat-pills-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
        }}
        .cat-pill-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .cat-pill-count {{
            font-size: 0.88rem;
            font-weight: 600;
            color: #475569;
        }}
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        .badge-priority {{
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }}
        .badge-review {{
            background-color: #dbeafe;
            color: #1e40af;
            border: 1px solid #bfdbfe;
        }}
        .badge-additional {{
            background-color: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
        }}
        .badge-neutral {{
            background-color: #f8fafc;
            color: #64748b;
            border: 1px solid #e2e8f0;
        }}
        .badge-skill-match {{
            background-color: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
        }}
        .badge-skill-missing {{
            background-color: #fef2f2;
            color: #b91c1c;
            border: 1px solid #fecaca;
        }}
        .badge-skill-pref {{
            background-color: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
        }}
        .rank-pill {{
            background-color: #3b82f6;
            color: #ffffff;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
        }}
        /* Overview Table */
        .table-responsive {{
            overflow-x: auto;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 28px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.90rem;
            text-align: left;
        }}
        thead {{
            background-color: #f1f5f9;
            border-bottom: 2px solid var(--border-color);
        }}
        th {{
            padding: 12px 14px;
            font-weight: 600;
            color: #334155;
            white-space: nowrap;
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
        }}
        tbody tr:hover {{
            background-color: #f8fafc;
        }}
        .text-center {{
            text-align: center;
        }}
        .font-bold {{
            font-weight: 600;
        }}
        .score-cell {{
            font-weight: 700;
            color: #0f172a;
        }}
        .empty-state {{
            text-align: center;
            color: var(--text-muted);
            padding: 32px;
            font-style: italic;
        }}
        /* Candidate Cards */
        .candidate-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            margin-bottom: 18px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        .card-header {{
            background-color: #f8fafc;
            border-bottom: 1px solid var(--border-color);
            padding: 14px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .candidate-title {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .candidate-title h3 {{
            font-size: 1.15rem;
            color: #0f172a;
        }}
        .score-display {{
            display: flex;
            align-items: baseline;
            gap: 4px;
        }}
        .score-num {{
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
        }}
        .score-sub {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}
        .card-body {{
            padding: 18px 20px;
        }}
        .contrib-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
            margin-bottom: 18px;
            background: #f8fafc;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #edf2f7;
        }}
        .contrib-item {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        .contrib-label {{
            font-size: 0.74rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }}
        .contrib-val {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #1e293b;
        }}
        .skills-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 14px;
            margin-bottom: 16px;
        }}
        .skill-group strong {{
            font-size: 0.85rem;
            color: #475569;
            display: block;
            margin-bottom: 6px;
        }}
        .badge-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .explanation-box {{
            background-color: #f1f5f9;
            border-left: 4px solid #64748b;
            padding: 12px 16px;
            border-radius: 0 6px 6px 0;
            font-size: 0.88rem;
            color: #334155;
            line-height: 1.5;
        }}
        /* Responsible AI Box */
        .ai-disclaimer {{
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            border-left: 5px solid #2563eb;
            border-radius: 8px;
            padding: 18px 22px;
            margin-top: 36px;
            margin-bottom: 24px;
        }}
        .ai-disclaimer h3 {{
            font-size: 1.05rem;
            color: #1e40af;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .ai-disclaimer p {{
            font-size: 0.88rem;
            color: #1e3a8a;
            line-height: 1.55;
        }}
        .footer {{
            text-align: center;
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-top: 24px;
            padding: 12px;
        }}
        @media (max-width: 768px) {{
            .summary-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .skills-section {{
                grid-template-columns: 1fr;
            }}
            .header .meta {{
                flex-direction: column;
                align-items: flex-start;
                gap: 6px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <h1>{report_title}</h1>
            <div class="meta">
                {f"<span><strong>Target Role:</strong> {esc_job_title}</span>" if esc_job_title else ""}
                <span><strong>Generated:</strong> {gen_time}</span>
                <span><strong>Candidates Evaluated:</strong> {total_candidates}</span>
            </div>
        </header>

        <!-- Cohort Summary Metrics -->
        <h2 class="section-title">Cohort Screening Summary</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-label">Total Candidates</div>
                <div class="metric-value">{total_candidates}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Highest Score</div>
                <div class="metric-value">{_format_float(highest_score)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Lowest Score</div>
                <div class="metric-value">{_format_float(lowest_score)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cohort Mean</div>
                <div class="metric-value">{_format_float(mean_score)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cohort Median</div>
                <div class="metric-value">{_format_float(median_score)}</div>
            </div>
        </div>

        <!-- Review Category Distribution -->
        <div class="distribution-box">
            <div class="distribution-title">Review Category Distribution:</div>
            <div class="cat-pills-container">
                {cat_distribution_html}
            </div>
        </div>

        <!-- Ranked Overview Table -->
        <h2 class="section-title">Ranked Candidate Overview</h2>
        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th class="text-center">Rank</th>
                        <th>Candidate ID</th>
                        <th class="text-center">Overall Score</th>
                        <th>Review Tier</th>
                        <th class="text-center">Req Skills %</th>
                        <th class="text-center">Pref Skills %</th>
                        <th class="text-center">Coverage %</th>
                        <th class="text-center">Text Sim %</th>
                        <th class="text-center">Exp Score</th>
                        <th class="text-center">Edu Score</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows_html)}
                </tbody>
            </table>
        </div>

        <!-- Detailed Candidate Breakdowns -->
        <h2 class="section-title">Detailed Candidate Profiles & Skill Audits</h2>
        <div class="candidate-profiles">
            {"".join(detail_cards_html)}
        </div>

        <!-- Responsible AI Governance Disclaimer -->
        <aside class="ai-disclaimer" role="note">
            <h3>Responsible AI & Human-in-the-Loop Governance Notice</h3>
            <p>
                This screening report is generated by an automated candidate matching tool to assist human
                hiring teams in triaging large applicant pools. Match percentages, rubric evaluations, and
                rank positions reflect keyword and semantic alignment with the published job specification.
                They do <strong>not</strong> constitute automated hiring or rejection decisions, nor do they
                predict on-the-job performance. All applicants remain eligible for holistic recruiter evaluation.
            </p>
        </aside>

        <!-- Footer -->
        <footer class="footer">
            <p>Automated Resume Screening Tool &bull; Phase 12 Report Generator &bull; Standalone Offline Report</p>
        </footer>
    </div>
</body>
</html>"""

    try:
        with open(target_path, mode="w", encoding="utf-8") as html_file:
            html_file.write(html_content)
    except Exception as exc:
        raise OSError(f"Failed to write HTML report to '{target_path}': {exc}") from exc

    return target_path


def generate_reports(
    ranked_candidates: List[Dict[str, Any]],
    output_dir: Union[str, Path] = "outputs",
    job_title: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    csv_filename: str = "candidate_ranking_report.csv",
    html_filename: str = "candidate_ranking_report.html",
) -> Dict[str, Path]:
    """Batch generate both CSV and HTML screening reports in the designated directory.

    Args:
        ranked_candidates: List of ranked and scored candidate dictionaries.
        output_dir: Target directory path (default: 'outputs').
        job_title: Optional job role title for header metadata.
        summary: Optional pre-computed ranking summary dictionary.
        csv_filename: Output CSV filename.
        html_filename: Output HTML filename.

    Returns:
        Dictionary mapping report type to generated Path:
        {'csv': Path(...), 'html': Path(...)}

    Raises:
        TypeError: If ranked_candidates is not a list or summary is not a dict.
        ValueError: If a candidate has invalid or missing candidate_id.
        OSError: If report generation fails.
    """
    validate_report_candidates(ranked_candidates)

    csv_path = generate_csv_report(
        ranked_candidates=ranked_candidates,
        output_dir=output_dir,
        filename=csv_filename,
    )

    html_path = generate_html_report(
        ranked_candidates=ranked_candidates,
        output_dir=output_dir,
        filename=html_filename,
        job_title=job_title,
        summary=summary,
    )

    return {
        "csv": csv_path,
        "html": html_path,
    }
