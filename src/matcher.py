"""Resume-to-job matching engine.

Computes canonical skill matching, percentage coverage, and TF-IDF cosine similarity
between candidate resumes and job profiles without predictive scoring or ranking.
"""

from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.job_parser import get_standardized_job_profile
from src.resume_parser import extract_resume_text
from src.skill_extractor import extract_skills, normalize_skill_name
from src.text_cleaner import clean_text


def calculate_required_skill_match(matched_count: int, total_count: int) -> float:
    """Calculate the match percentage for required skills.

    Args:
        matched_count: Number of required skills satisfied by candidate.
        total_count: Total number of required skills specified by job.

    Returns:
        Percentage as a float between 0.0 and 100.0. Returns 100.0 if total_count is 0.

    Raises:
        TypeError: If arguments are not integers.
        ValueError: If arguments are negative or matched_count > total_count.
    """
    if not isinstance(matched_count, int) or not isinstance(total_count, int):
        raise TypeError("Counts must be integers.")
    if matched_count < 0 or total_count < 0:
        raise ValueError("Skill counts cannot be negative.")
    if matched_count > total_count:
        raise ValueError(f"Matched count ({matched_count}) cannot exceed total count ({total_count}).")

    if total_count == 0:
        return 100.0

    return round((matched_count / total_count) * 100.0, 2)


def calculate_preferred_skill_match(matched_count: int, total_count: int) -> float:
    """Calculate the match percentage for preferred skills.

    Args:
        matched_count: Number of preferred skills satisfied by candidate.
        total_count: Total number of preferred skills specified by job.

    Returns:
        Percentage as a float between 0.0 and 100.0. Returns 100.0 if total_count is 0.

    Raises:
        TypeError: If arguments are not integers.
        ValueError: If arguments are negative or matched_count > total_count.
    """
    if not isinstance(matched_count, int) or not isinstance(total_count, int):
        raise TypeError("Counts must be integers.")
    if matched_count < 0 or total_count < 0:
        raise ValueError("Skill counts cannot be negative.")
    if matched_count > total_count:
        raise ValueError(f"Matched count ({matched_count}) cannot exceed total count ({total_count}).")

    if total_count == 0:
        return 100.0

    return round((matched_count / total_count) * 100.0, 2)


def calculate_skill_coverage(
    matched_required: int, total_required: int, matched_preferred: int, total_preferred: int
) -> float:
    """Calculate overall skill coverage percentage across both required and preferred skills.

    Formula:
        (total matched required + total matched preferred) /
        (total required + total preferred) * 100

    Args:
        matched_required: Number of matched required skills.
        total_required: Total required skills.
        matched_preferred: Number of matched preferred skills.
        total_preferred: Total preferred skills.

    Returns:
        Overall coverage percentage between 0.0 and 100.0. Returns 100.0 if total is 0.

    Raises:
        TypeError: If any count is not an integer.
        ValueError: If any count is negative or matched exceeds total.
    """
    for val, name in [
        (matched_required, "matched_required"),
        (total_required, "total_required"),
        (matched_preferred, "matched_preferred"),
        (total_preferred, "total_preferred"),
    ]:
        if not isinstance(val, int):
            raise TypeError(f"'{name}' must be an integer, got {type(val).__name__}.")
        if val < 0:
            raise ValueError(f"'{name}' cannot be negative: {val}")

    if matched_required > total_required:
        raise ValueError(f"matched_required ({matched_required}) cannot exceed total_required ({total_required}).")
    if matched_preferred > total_preferred:
        raise ValueError(f"matched_preferred ({matched_preferred}) cannot exceed total_preferred ({total_preferred}).")

    total_skills = total_required + total_preferred
    if total_skills == 0:
        return 100.0

    matched_total = matched_required + matched_preferred
    return round((matched_total / total_skills) * 100.0, 2)


def match_skills(candidate_skills: Iterable[str], job_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compare candidate skills with job profile requirements and compute match metrics.

    Normalizes candidate skills using canonical aliases, avoids duplicate counts,
    and performs case-insensitive comparisons against job profile skill requirements.

    Args:
        candidate_skills: Iterable of candidate skill strings or aliases.
        job_profile: Standardized job profile dictionary containing 'required_skills'
                     and 'preferred_skills'.

    Returns:
        Dictionary containing:
        - matched_required_skills (List[str])
        - missing_required_skills (List[str])
        - matched_preferred_skills (List[str])
        - missing_preferred_skills (List[str])
        - total_required_skills (int)
        - total_preferred_skills (int)
        - required_skill_match_percentage (float)
        - preferred_skill_match_percentage (float)
        - overall_skill_coverage_percentage (float)

    Raises:
        TypeError: If candidate_skills is not iterable or job_profile is not a dict.
        ValueError: If job_profile does not contain expected skill lists.
    """
    if not isinstance(job_profile, dict):
        raise TypeError(f"Expected job_profile as dict, got {type(job_profile).__name__}.")
    if not isinstance(candidate_skills, Iterable) or isinstance(candidate_skills, (str, bytes)):
        raise TypeError("candidate_skills must be an iterable of strings.")

    # 1. Normalize and canonicalize candidate skills
    normalized_candidate_set = set()
    for skill in candidate_skills:
        if not isinstance(skill, str):
            raise TypeError(f"All candidate skills must be strings, got {type(skill).__name__}.")
        cleaned = skill.strip()
        if cleaned:
            canonical = normalize_skill_name(cleaned)
            normalized_candidate_set.add(canonical.lower())

    # 2. Extract job profile requirements
    required_skills = job_profile.get("required_skills", [])
    preferred_skills = job_profile.get("preferred_skills", [])

    if not isinstance(required_skills, list) or not isinstance(preferred_skills, list):
        raise ValueError("job_profile 'required_skills' and 'preferred_skills' must be lists.")

    # 3. Match required skills
    matched_required = []
    missing_required = []
    for req in required_skills:
        if req.lower() in normalized_candidate_set:
            matched_required.append(req)
        else:
            missing_required.append(req)

    # 4. Match preferred skills
    matched_preferred = []
    missing_preferred = []
    for pref in preferred_skills:
        if pref.lower() in normalized_candidate_set:
            matched_preferred.append(pref)
        else:
            missing_preferred.append(pref)

    # 5. Compute metrics
    total_req = len(required_skills)
    total_pref = len(preferred_skills)

    req_percentage = calculate_required_skill_match(len(matched_required), total_req)
    pref_percentage = calculate_preferred_skill_match(len(matched_preferred), total_pref)
    overall_coverage = calculate_skill_coverage(len(matched_required), total_req, len(matched_preferred), total_pref)

    return {
        "matched_required_skills": matched_required,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred,
        "missing_preferred_skills": missing_preferred,
        "total_required_skills": total_req,
        "total_preferred_skills": total_pref,
        "required_skill_match_percentage": req_percentage,
        "preferred_skill_match_percentage": pref_percentage,
        "overall_skill_coverage_percentage": overall_coverage,
    }


def calculate_text_similarity(resume_text: str, job_description: str) -> Dict[str, float]:
    """Calculate TF-IDF cosine similarity between resume text and job description.

    Fits a TfidfVectorizer on the cleaned texts together in the same vector space.
    Gracefully handles empty strings, whitespace, stopword-only documents, and
    short texts without crashing.

    Args:
        resume_text: Raw or cleaned resume text.
        job_description: Raw or cleaned job description text.

    Returns:
        Dictionary containing:
        - similarity_score (float): Value between 0.0 and 1.0.
        - similarity_percentage (float): Value between 0.0 and 100.0.

    Raises:
        TypeError: If either input is not a string.
    """
    if not isinstance(resume_text, str):
        raise TypeError(f"resume_text must be str, got {type(resume_text).__name__}.")
    if not isinstance(job_description, str):
        raise TypeError(f"job_description must be str, got {type(job_description).__name__}.")

    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_description)

    # If either text has no usable tokens, similarity is 0.0
    if not clean_resume or not clean_job:
        return {"similarity_score": 0.0, "similarity_percentage": 0.0}

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_job])
        raw_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        # Clamp to [0.0, 1.0] to safeguard against minor floating-point errors
        clamped_score = max(0.0, min(1.0, float(raw_score)))
    except ValueError:
        # Occurs if documents contain no usable tokens (e.g. empty vocabulary)
        clamped_score = 0.0
    except Exception:
        clamped_score = 0.0

    return {
        "similarity_score": round(clamped_score, 4),
        "similarity_percentage": round(clamped_score * 100.0, 2),
    }


def _extract_candidate_id(resume_text: str, fallback_id: Optional[str] = None) -> str:
    """Extract candidate ID from resume text or fallback to provided ID."""
    match = re.search(r"CANDIDATE ID:\s*([A-Za-z0-9_-]+)", resume_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return fallback_id if fallback_id is not None else "UNKNOWN"


def match_candidate(
    resume_text: str,
    job_profile: Dict[str, Any],
    candidate_id: Optional[str] = None,
    candidate_skills: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Match a single candidate resume against a job profile.

    Combines skill extraction/matching with TF-IDF cosine similarity into a
    structured match result. Does not compute weighted composite scores or rank.

    Args:
        resume_text: Candidate resume raw text string.
        job_profile: Standardized job profile dictionary.
        candidate_id: Optional explicit candidate ID. Extracted from text if omitted.
        candidate_skills: Optional pre-extracted skills list. Extracted if omitted.

    Returns:
        Structured match result dictionary containing:
        - candidate_id (str)
        - matched_required_skills (List[str])
        - missing_required_skills (List[str])
        - matched_preferred_skills (List[str])
        - missing_preferred_skills (List[str])
        - total_required_skills (int)
        - total_preferred_skills (int)
        - required_skill_match_percentage (float)
        - preferred_skill_match_percentage (float)
        - overall_skill_coverage_percentage (float)
        - text_similarity_score (float)
        - text_similarity_percentage (float)
    """
    if not isinstance(resume_text, str):
        raise TypeError(f"resume_text must be str, got {type(resume_text).__name__}.")
    if not isinstance(job_profile, dict):
        raise TypeError(f"job_profile must be dict, got {type(job_profile).__name__}.")

    effective_id = _extract_candidate_id(resume_text, candidate_id)

    # Extract skills if not provided
    if candidate_skills is None:
        extracted = extract_skills(resume_text)
    else:
        extracted = list(candidate_skills)

    # 1. Skill matching
    skill_results = match_skills(extracted, job_profile)

    # 2. Text similarity
    job_desc = job_profile.get("job_description", "")
    similarity_results = calculate_text_similarity(resume_text, job_desc)

    return {
        "candidate_id": effective_id,
        "matched_required_skills": skill_results["matched_required_skills"],
        "missing_required_skills": skill_results["missing_required_skills"],
        "matched_preferred_skills": skill_results["matched_preferred_skills"],
        "missing_preferred_skills": skill_results["missing_preferred_skills"],
        "total_required_skills": skill_results["total_required_skills"],
        "total_preferred_skills": skill_results["total_preferred_skills"],
        "required_skill_match_percentage": skill_results["required_skill_match_percentage"],
        "preferred_skill_match_percentage": skill_results["preferred_skill_match_percentage"],
        "overall_skill_coverage_percentage": skill_results["overall_skill_coverage_percentage"],
        "text_similarity_score": similarity_results["similarity_score"],
        "text_similarity_percentage": similarity_results["similarity_percentage"],
    }


def match_candidates(
    candidates: List[Union[str, Path]],
    job_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Batch match multiple candidate resumes against a job profile.

    Supports candidate inputs as either file paths (Path or str pointing to .txt/.pdf/.docx)
    or raw resume text strings. Handles candidate-level errors gracefully without
    corrupting the entire batch.

    Args:
        candidates: List of file paths or raw resume text strings.
        job_profile: Optional job profile. Defaults to standardized Junior Python profile.

    Returns:
        List of candidate match result dictionaries.

    Raises:
        TypeError: If candidates is not a list or job_profile is invalid.
    """
    if not isinstance(candidates, list):
        raise TypeError(f"candidates must be a list, got {type(candidates).__name__}.")

    profile = job_profile if job_profile is not None else get_standardized_job_profile()

    results: List[Dict[str, Any]] = []

    for item in candidates:
        candidate_entry = None
        # Check if item represents an existing file
        is_file_path = False
        if isinstance(item, (str, Path)):
            try:
                p = Path(item)
                if p.exists() and p.is_file():
                    is_file_path = True
            except Exception:
                is_file_path = False

        if is_file_path:
            file_path = Path(item)
            cand_id = file_path.stem
            try:
                resume_text = extract_resume_text(file_path)
                candidate_entry = match_candidate(resume_text, profile, candidate_id=cand_id)
            except Exception as exc:
                candidate_entry = {
                    "candidate_id": cand_id,
                    "error": str(exc),
                    "status": "failed",
                }
        else:
            # Item is raw resume text
            if not isinstance(item, str):
                candidate_entry = {
                    "candidate_id": "UNKNOWN",
                    "error": f"Invalid candidate item type: {type(item).__name__}",
                    "status": "failed",
                }
            else:
                try:
                    candidate_entry = match_candidate(item, profile)
                except Exception as exc:
                    candidate_entry = {
                        "candidate_id": "UNKNOWN",
                        "error": str(exc),
                        "status": "failed",
                    }

        if candidate_entry is not None:
            results.append(candidate_entry)

    return results
