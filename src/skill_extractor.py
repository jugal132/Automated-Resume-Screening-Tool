"""Technical skill extraction and alias normalization module.

Identifies technical competencies, programming languages, databases, tools,
and libraries from resume text, mapping aliases to standardized canonical skill names
using boundary-aware pattern matching.
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

from src.text_cleaner import clean_text

# Default location for the skill database JSON file
DEFAULT_SKILLS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "skills_database" / "skills.json"


def load_skill_database(db_path: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, List[str]]]:
    """Load and validate the structured skill dictionary from JSON.

    Args:
        db_path: Optional custom path to skills.json. Defaults to DEFAULT_SKILLS_DB_PATH.

    Returns:
        A nested dictionary mapping Category -> {CanonicalSkill: [Aliases]}.

    Raises:
        FileNotFoundError: If the skill database file cannot be found.
        ValueError: If JSON syntax is invalid or database structure does not conform to schema.
    """
    path = Path(db_path) if db_path is not None else DEFAULT_SKILLS_DB_PATH

    if not path.exists():
        raise FileNotFoundError(f"Skill database file not found at: '{path}'")

    if not path.is_file():
        raise ValueError(f"Skill database path is not a regular file: '{path}'")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in skill database '{path}': {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read skill database '{path}': {exc}") from exc

    # Validate schema structure
    if not isinstance(data, dict) or not data:
        raise ValueError("Invalid skill database schema: expected non-empty root dictionary.")

    for category, skills in data.items():
        if not isinstance(skills, dict):
            raise ValueError(f"Invalid schema for category '{category}': expected dictionary of skills.")
        for canonical, aliases in skills.items():
            if not isinstance(aliases, list):
                raise ValueError(
                    f"Invalid schema for skill '{canonical}' in category '{category}': expected list of aliases."
                )

    return data


def _build_boundary_pattern(term: str) -> re.Pattern:
    """Build a boundary-aware regular expression pattern for a skill or alias term.

    Handles special boundary conditions for terms containing punctuation
    such as 'C++', 'C#', '.NET', 'Node.js', isolated 'C', etc.

    Args:
        term: The skill or alias text.

    Returns:
        Compiled case-insensitive regex pattern.
    """
    raw = term.strip().lower()

    if raw == "c++":
        # Matches c++ with leading word boundary, not followed by additional +
        return re.compile(r"\bc\+\+(?!\+)", re.IGNORECASE)
    elif raw == "c#":
        # Matches c# not followed by # or alphanumeric char
        return re.compile(r"\bc#(?![#\w])", re.IGNORECASE)
    elif raw == ".net":
        # Matches .net preceded by start of string or whitespace
        return re.compile(r"(?:^|(?<=[\s,.;:()\[\]]))\.net\b", re.IGNORECASE)
    elif raw == "c":
        # Strict boundary for single-letter language C to prevent false positives
        # in words like 'developer', 'communication', or terms like 'c++'/'c#'
        return re.compile(r"(?<![\w+#])c(?![\w+#])", re.IGNORECASE)
    else:
        # Standard phrase or single token
        escaped = re.escape(raw)
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def normalize_skill_name(skill_name: str, db: Optional[Dict[str, Dict[str, List[str]]]] = None) -> str:
    """Normalize a skill alias or variant to its canonical skill name.

    Performs case-insensitive lookup across all canonical skills and aliases.
    If the name matches an existing canonical skill or alias, the canonical name
    is returned. If no match is found, the stripped original name is returned.

    Args:
        skill_name: The raw skill name or alias to normalize.
        db: Optional loaded skill database. If omitted, the default database is loaded.

    Returns:
        The canonical skill name string, or the stripped original if unknown.

    Raises:
        TypeError: If skill_name is not a string.
    """
    if not isinstance(skill_name, str):
        raise TypeError(f"Expected str for skill_name, got {type(skill_name).__name__}.")

    cleaned = skill_name.strip()
    if not cleaned:
        return ""

    database = db if db is not None else load_skill_database()
    target_lower = cleaned.lower()

    # Search canonical names and aliases
    for _, skills in database.items():
        for canonical, aliases in skills.items():
            if canonical.lower() == target_lower:
                return canonical
            for alias in aliases:
                if alias.lower() == target_lower:
                    return canonical

    return cleaned


def extract_skills(text: str, db: Optional[Dict[str, Dict[str, List[str]]]] = None) -> List[str]:
    """Extract technical skills from resume text, mapping aliases to canonical names.

    Normalizes text via clean_text() to preserve technical tokens, applies
    boundary-aware regex matching for all canonical skills and aliases, and
    deduplicates matches to return a unique list of detected canonical skills.

    Args:
        text: Input resume or document text.
        db: Optional preloaded skill database.

    Returns:
        Sorted list of unique detected canonical skill names.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str for text, got {type(text).__name__}.")

    if not text.strip():
        return []

    database = db if db is not None else load_skill_database()

    # Preprocess text through text_cleaner to ensure standardized representation
    cleaned = clean_text(text)
    if not cleaned:
        return []

    detected_canonical: set = set()

    # Iterate through categories and skills
    for _, skills in database.items():
        for canonical, aliases in skills.items():
            # Check canonical name
            canonical_pattern = _build_boundary_pattern(canonical)
            if canonical_pattern.search(cleaned):
                detected_canonical.add(canonical)
                continue

            # Check aliases if canonical was not already matched
            for alias in aliases:
                alias_pattern = _build_boundary_pattern(alias)
                if alias_pattern.search(cleaned):
                    detected_canonical.add(canonical)
                    break

    return sorted(list(detected_canonical))


def get_skill_categories(
    text: str, db: Optional[Dict[str, Dict[str, List[str]]]] = None
) -> Dict[str, List[str]]:
    """Extract skills from text and group detected canonical skills by category.

    Args:
        text: Input resume text.
        db: Optional preloaded skill database.

    Returns:
        Dictionary mapping Category -> list of detected canonical skills.
        Only categories containing at least one detected skill are included.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str for text, got {type(text).__name__}.")

    database = db if db is not None else load_skill_database()
    detected_skills = set(extract_skills(text, db=database))

    result: Dict[str, List[str]] = {}
    for category, skills in database.items():
        category_matches = [canonical for canonical in skills if canonical in detected_skills]
        if category_matches:
            result[category] = sorted(category_matches)

    return result
