"""Job description and structured requirements parsing module.

Loads, validates, and normalizes job description text and structured requirements JSON,
integrating with the skill normalizer to produce standardized job profiles for
candidate matching and scoring.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from src.skill_extractor import normalize_skill_name

# Default file paths
DEFAULT_JOB_DESCRIPTION_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "job_descriptions" / "junior_python_developer.txt"
)
DEFAULT_JOB_REQUIREMENTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "job_descriptions"
    / "junior_python_developer_requirements.json"
)

# Mandatory fields expected in the requirements JSON schema
MANDATORY_REQUIREMENTS_FIELDS = [
    "job_title",
    "required_skills",
    "preferred_skills",
    "minimum_experience_years",
    "preferred_experience_years",
    "education_preferred",
    "experience_evidence",
]


def load_job_description(file_path: Optional[Union[str, Path]] = None) -> str:
    """Load and validate the plain text job description file.

    Args:
        file_path: Optional path to job description file. Defaults to DEFAULT_JOB_DESCRIPTION_PATH.

    Returns:
        The stripped raw job description text.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file path is a directory, unreadable, or contains empty/whitespace-only text.
        TypeError: If file_path is not a valid str or Path.
    """
    if file_path is not None and not isinstance(file_path, (str, Path)):
        raise TypeError(f"Expected file_path as str or Path, got {type(file_path).__name__}.")

    path = Path(file_path) if file_path is not None else DEFAULT_JOB_DESCRIPTION_PATH

    if not path.exists():
        raise FileNotFoundError(f"Job description file not found at: '{path}'")

    if not path.is_file():
        raise ValueError(f"Job description path is not a regular file: '{path}'")

    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")
    except Exception as exc:
        raise ValueError(f"Failed to read job description '{path}': {exc}") from exc

    cleaned = content.strip()
    if not cleaned:
        raise ValueError(f"Job description file '{path}' is empty or contains only whitespace.")

    return cleaned


def _normalize_skill_list(skills_list: Any, field_name: str) -> List[str]:
    """Validate, normalize aliases, and deduplicate a list of skills.

    Args:
        skills_list: Raw list of skill strings.
        field_name: Field name for descriptive error messaging.

    Returns:
        Deduplicated list of canonical skill names preserving insertion order.

    Raises:
        TypeError: If skills_list is not a list or contains non-string items.
        ValueError: If any skill item is an empty string.
    """
    if not isinstance(skills_list, list):
        raise TypeError(f"'{field_name}' must be a list, got {type(skills_list).__name__}.")

    seen_lower = set()
    normalized: List[str] = []

    for item in skills_list:
        if not isinstance(item, str):
            raise TypeError(f"All items in '{field_name}' must be strings, got {type(item).__name__}.")

        cleaned_item = item.strip()
        if not cleaned_item:
            raise ValueError(f"Found empty or whitespace-only skill in '{field_name}'.")

        canonical = normalize_skill_name(cleaned_item)
        canonical_lower = canonical.lower()

        if canonical_lower not in seen_lower:
            seen_lower.add(canonical_lower)
            normalized.append(canonical)

    return normalized


def parse_job_requirements(raw_data: Any) -> Dict[str, Any]:
    """Validate, normalize, and parse raw job requirements dictionary.

    Performs schema validation, type checks, alias normalization, deduplication,
    experience constraints verification, and detects overlaps between required
    and preferred skills.

    Args:
        raw_data: Dictionary loaded from job requirements JSON.

    Returns:
        Validated and normalized requirements dictionary.

    Raises:
        TypeError: If raw_data is not a dictionary or contains invalid data types.
        ValueError: If required fields are missing, invalid, or skills overlap.
    """
    if not isinstance(raw_data, dict):
        raise TypeError(f"Job requirements must be a dictionary, got {type(raw_data).__name__}.")

    # 1. Verify mandatory fields presence
    missing_fields = [f for f in MANDATORY_REQUIREMENTS_FIELDS if f not in raw_data]
    if missing_fields:
        raise ValueError(f"Missing mandatory requirements fields: {', '.join(missing_fields)}")

    # 2. Validate job_title
    job_title = raw_data["job_title"]
    if not isinstance(job_title, str) or not job_title.strip():
        raise ValueError("Field 'job_title' must be a non-empty string.")

    # 3. Validate and normalize skills
    required_skills = _normalize_skill_list(raw_data["required_skills"], "required_skills")
    preferred_skills = _normalize_skill_list(raw_data["preferred_skills"], "preferred_skills")

    # 4. Check for overlap between required and preferred skills
    req_set = {s.lower() for s in required_skills}
    pref_set = {s.lower() for s in preferred_skills}
    overlap = req_set.intersection(pref_set)
    if overlap:
        overlapping_skills = [s for s in preferred_skills if s.lower() in overlap]
        raise ValueError(
            f"Skills cannot be both required and preferred. Overlapping skills: {overlapping_skills}"
        )

    # 5. Validate experience fields
    min_exp = raw_data["minimum_experience_years"]
    pref_exp = raw_data["preferred_experience_years"]

    if not isinstance(min_exp, (int, float)) or isinstance(min_exp, bool):
        raise TypeError(f"'minimum_experience_years' must be numeric, got {type(min_exp).__name__}.")
    if not isinstance(pref_exp, (int, float)) or isinstance(pref_exp, bool):
        raise TypeError(f"'preferred_experience_years' must be numeric, got {type(pref_exp).__name__}.")

    if min_exp < 0:
        raise ValueError(f"'minimum_experience_years' cannot be negative: {min_exp}")
    if pref_exp < min_exp:
        raise ValueError(
            f"'preferred_experience_years' ({pref_exp}) cannot be less than "
            f"'minimum_experience_years' ({min_exp})."
        )

    # 6. Validate education_preferred
    edu_pref = raw_data["education_preferred"]
    if not isinstance(edu_pref, list):
        raise TypeError(f"'education_preferred' must be a list, got {type(edu_pref).__name__}.")
    for edu in edu_pref:
        if not isinstance(edu, str) or not edu.strip():
            raise TypeError("All items in 'education_preferred' must be non-empty strings.")

    # 7. Validate experience_evidence
    exp_evidence = raw_data["experience_evidence"]
    if not isinstance(exp_evidence, list):
        raise TypeError(f"'experience_evidence' must be a list, got {type(exp_evidence).__name__}.")
    for ev in exp_evidence:
        if not isinstance(ev, str) or not ev.strip():
            raise TypeError("All items in 'experience_evidence' must be non-empty strings.")

    return {
        "job_title": job_title.strip(),
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "minimum_experience_years": min_exp,
        "preferred_experience_years": pref_exp,
        "education_preferred": [e.strip() for e in edu_pref],
        "experience_evidence": [ev.strip() for ev in exp_evidence],
    }


def load_job_requirements(file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load, validate, and parse structured job requirements from a JSON file.

    Args:
        file_path: Optional path to JSON file. Defaults to DEFAULT_JOB_REQUIREMENTS_PATH.

    Returns:
        Validated requirements dictionary.

    Raises:
        FileNotFoundError: If the requirements file does not exist.
        ValueError: If JSON syntax is invalid or schema validation fails.
        TypeError: If file_path is of invalid type.
    """
    if file_path is not None and not isinstance(file_path, (str, Path)):
        raise TypeError(f"Expected file_path as str or Path, got {type(file_path).__name__}.")

    path = Path(file_path) if file_path is not None else DEFAULT_JOB_REQUIREMENTS_PATH

    if not path.exists():
        raise FileNotFoundError(f"Job requirements file not found at: '{path}'")

    if not path.is_file():
        raise ValueError(f"Job requirements path is not a regular file: '{path}'")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in requirements file '{path}': {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read requirements file '{path}': {exc}") from exc

    return parse_job_requirements(data)


def _extract_responsibilities(description_text: str) -> List[str]:
    """Extract bulleted responsibility statements from job description text.

    Args:
        description_text: Raw or cleaned job description text.

    Returns:
        List of responsibility statements, or empty list if section not found.
    """
    match = re.search(r"RESPONSIBILITIES\s*\n(.*?)(?=\n[A-Z\s]{4,}|\Z)", description_text, re.DOTALL)
    if not match:
        return []

    lines = match.group(1).splitlines()
    responsibilities = []
    for line in lines:
        cleaned = line.strip().lstrip("-*•▪ ").strip()
        if cleaned:
            responsibilities.append(cleaned)

    return responsibilities


def get_standardized_job_profile(
    description_path: Optional[Union[str, Path]] = None,
    requirements_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Assemble a standardized job profile dictionary from description and requirements.

    Combines the plain text description and validated requirements into a clean,
    unified profile dictionary suitable for downstream matching and scoring.

    Args:
        description_path: Optional path to job description TXT file.
        requirements_path: Optional path to job requirements JSON file.

    Returns:
        Standardized job profile dictionary containing:
        - job_title (str)
        - job_description (str)
        - required_skills (List[str])
        - preferred_skills (List[str])
        - minimum_experience (Union[int, float])
        - preferred_experience (Union[int, float])
        - education (List[str])
        - responsibilities (List[str])
        - experience_evidence (List[str])
    """
    description_text = load_job_description(description_path)
    requirements_data = load_job_requirements(requirements_path)
    responsibilities = _extract_responsibilities(description_text)

    return {
        "job_title": requirements_data["job_title"],
        "job_description": description_text,
        "required_skills": requirements_data["required_skills"],
        "preferred_skills": requirements_data["preferred_skills"],
        "minimum_experience": requirements_data["minimum_experience_years"],
        "preferred_experience": requirements_data["preferred_experience_years"],
        "education": requirements_data["education_preferred"],
        "responsibilities": responsibilities,
        "experience_evidence": requirements_data["experience_evidence"],
    }
