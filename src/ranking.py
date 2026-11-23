"""Candidate Ranking Engine for Automated Resume Screening Tool.

This module provides deterministic ranking and sorting of already-scored
candidate profiles based on overall screening score and transparent tie-breaking
hierarchies.

Ranking Order:
    1. Overall screening score (descending by default, configurable)
    2. Required skill match percentage (descending)
    3. Overall skill coverage percentage (descending)
    4. Text similarity percentage (descending)
    5. Candidate ID (ascending lexicographical tie-breaker)

Tie-Handling Methodology:
    - Ranking Method: Standard Competition Ranking ("1224" ranking, default).
      Candidates with identical overall scores share the same rank position
      (e.g., two top candidates both receive Rank 1, and the next candidate
      receives Rank 3). This avoids misleading recruiters into treating
      candidates with identical primary scores as having different overall tiers.
    - Display Order: The sequential ordering (1, 2, 3...) after deterministic
      tie-breakers have resolved ordering in the sorted list.
    - Supported ranking methods: 'competition', 'dense' (1, 1, 2), 'sequential' (1, 2, 3, 4).

Responsible AI Note:
    Candidate ranking is an organizational triage mechanism for human recruitment
    workflows. It does NOT predict candidate job success, nor does it automate
    rejection decisions. All candidates remain available for holistic evaluation.
"""

from collections import Counter
import copy
import math
from statistics import mean, median
from typing import Any, Dict, List, Optional, Set, Tuple


def _validate_numeric_metric(
    candidate: Dict[str, Any],
    candidate_id: str,
    primary_keys: List[str],
    fallback_paths: List[str],
    label: str,
) -> float:
    """Extract and validate a numeric percentage metric (0.0 - 100.0).

    Checks primary candidate keys first, followed by nested raw_match_result paths.

    Args:
        candidate: Candidate dictionary.
        candidate_id: Candidate identifier for error reporting.
        primary_keys: Candidate top-level keys to check.
        fallback_paths: Keys inside candidate['raw_match_result'] to check.
        label: Human-readable label for error messages.

    Returns:
        Validated float value between 0.0 and 100.0.

    Raises:
        TypeError: If the value is non-numeric or boolean.
        ValueError: If the key is missing or outside [0.0, 100.0].
    """
    val = None
    for k in primary_keys:
        if k in candidate:
            val = candidate[k]
            break

    if val is None and "raw_match_result" in candidate and isinstance(candidate["raw_match_result"], dict):
        raw_match = candidate["raw_match_result"]
        for k in fallback_paths:
            if k in raw_match:
                val = raw_match[k]
                break

    if val is None:
        raise ValueError(
            f"Candidate '{candidate_id}' is missing required field for {label} "
            f"(checked keys: {primary_keys + fallback_paths})."
        )

    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(
            f"Candidate '{candidate_id}' has non-numeric {label}: {val!r} "
            f"(type: {type(val).__name__}). Must be a float or integer."
        )

    f_val = float(val)
    if not (0.0 <= f_val <= 100.0):
        raise ValueError(
            f"Candidate '{candidate_id}' has {label} {f_val} outside valid range [0, 100]."
        )

    return f_val


def validate_candidate_record(
    candidate: Dict[str, Any],
    seen_ids: Optional[Set[str]] = None,
) -> Tuple[str, float, float, float, float]:
    """Validate a single candidate record for ranking.

    Ensures that mandatory fields are present, candidate IDs are unique non-empty
    strings, and all score/metric values are valid numbers within [0, 100].

    Args:
        candidate: Candidate dictionary.
        seen_ids: Optional set of already encountered candidate IDs to check duplicates.

    Returns:
        Tuple of (candidate_id, overall_score, required_match_pct, coverage_pct, text_sim_pct).

    Raises:
        TypeError: If candidate is not a dict or fields contain invalid types.
        ValueError: If fields are missing, invalid, out of range, or duplicate ID found.
    """
    if not isinstance(candidate, dict):
        raise TypeError(f"Candidate record must be a dict, got {type(candidate).__name__}.")

    cid = candidate.get("candidate_id")
    if cid is None or not isinstance(cid, str) or not cid.strip():
        raise ValueError(
            f"Candidate record is missing a valid non-empty 'candidate_id': received {cid!r}."
        )
    candidate_id = cid.strip()

    if seen_ids is not None:
        if candidate_id in seen_ids:
            raise ValueError(
                f"Duplicate candidate ID detected: '{candidate_id}'. Candidate IDs must be unique."
            )
        seen_ids.add(candidate_id)

    # 1. Overall Score validation
    if "overall_score" not in candidate:
        raise ValueError(f"Candidate '{candidate_id}' is missing mandatory 'overall_score' field.")

    overall = candidate["overall_score"]
    if isinstance(overall, bool) or not isinstance(overall, (int, float)):
        raise TypeError(
            f"Candidate '{candidate_id}' has non-numeric 'overall_score': {overall!r} "
            f"(type: {type(overall).__name__}). Must be numeric."
        )
    overall_score = float(overall)
    if not (0.0 <= overall_score <= 100.0):
        raise ValueError(
            f"Candidate '{candidate_id}' has 'overall_score' {overall_score} outside [0, 100]."
        )

    # 2. Required skill percentage validation
    req_pct = _validate_numeric_metric(
        candidate=candidate,
        candidate_id=candidate_id,
        primary_keys=["required_skill_match_percentage", "required_skill_score"],
        fallback_paths=["required_skill_match_percentage"],
        label="required skill match percentage",
    )

    # 3. Overall skill coverage percentage validation
    cov_pct = _validate_numeric_metric(
        candidate=candidate,
        candidate_id=candidate_id,
        primary_keys=["overall_skill_coverage_percentage", "skill_coverage_percentage"],
        fallback_paths=["overall_skill_coverage_percentage"],
        label="overall skill coverage percentage",
    )

    # 4. Text similarity percentage validation
    sim_pct = _validate_numeric_metric(
        candidate=candidate,
        candidate_id=candidate_id,
        primary_keys=["text_similarity_percentage", "text_similarity_score"],
        fallback_paths=["text_similarity_percentage"],
        label="text similarity percentage",
    )

    return candidate_id, overall_score, req_pct, cov_pct, sim_pct


def rank_candidates(
    candidates: List[Dict[str, Any]],
    descending: bool = True,
    ranking_method: str = "competition",
) -> List[Dict[str, Any]]:
    """Rank scored candidates deterministically using scores and tie-breaking rules.

    The ranking process executes without modifying input objects or calculating/altering
    underlying candidate scores.

    Tie-breaking Order:
        1. Overall score (descending by default, or ascending if descending=False)
        2. Required skill match percentage (descending)
        3. Overall skill coverage percentage (descending)
        4. Text similarity percentage (descending)
        5. Candidate ID (ascending lexicographical order)

    Rank Assignment (based on overall score):
        - 'competition' (default): 1, 1, 3 (standard competition ranking).
        - 'dense': 1, 1, 2.
        - 'sequential': 1, 2, 3, 4 (strictly by display order).

    Args:
        candidates: List of scored candidate dictionaries.
        descending: True for highest overall score first (default), False for ascending.
        ranking_method: Ranking method for ties: 'competition', 'dense', or 'sequential'.

    Returns:
        A new list of deep-copied candidate dictionaries with added fields:
        - 'rank' (int): Rank position based on ranking_method.
        - 'display_order' (int): 1-indexed sequential position in the sorted list.
        - 'is_tied' (bool): True if candidate shares overall score with another.
        - 'required_skill_match_percentage' (float)
        - 'overall_skill_coverage_percentage' (float)
        - 'text_similarity_percentage' (float)

    Raises:
        TypeError: If inputs are of incorrect type.
        ValueError: If candidate data is invalid, out of range, or contains duplicate IDs.
    """
    if not isinstance(candidates, list):
        raise TypeError(f"candidates must be a list of dicts, got {type(candidates).__name__}.")

    if not isinstance(descending, bool):
        raise TypeError(f"descending must be a boolean, got {type(descending).__name__}.")

    valid_methods = {"competition", "dense", "sequential"}
    if ranking_method not in valid_methods:
        raise ValueError(
            f"Unsupported ranking_method '{ranking_method}'. Must be one of: {sorted(valid_methods)}."
        )

    if not candidates:
        return []

    # 1. Validate all candidates and extract tie-break metrics
    seen_ids: Set[str] = set()
    validated_tuples: List[Tuple[Dict[str, Any], str, float, float, float, float]] = []

    for cand in candidates:
        cid, overall, req_pct, cov_pct, sim_pct = validate_candidate_record(cand, seen_ids)
        cand_copy = copy.deepcopy(cand)
        validated_tuples.append((cand_copy, cid, overall, req_pct, cov_pct, sim_pct))

    # 2. Sort deterministically using composite tie-breaking key
    # Primary: overall_score (descending if descending=True, else ascending)
    # Secondary: required_skill_match_percentage (descending)
    # Tertiary: overall_skill_coverage_percentage (descending)
    # Quaternary: text_similarity_percentage (descending)
    # Quinary: candidate_id (ascending)
    def _sort_key(item: Tuple[Dict[str, Any], str, float, float, float, float]):
        _, cid, overall, req_pct, cov_pct, sim_pct = item
        primary = -overall if descending else overall
        return (
            primary,
            -req_pct,
            -cov_pct,
            -sim_pct,
            cid,
        )

    sorted_tuples = sorted(validated_tuples, key=_sort_key)

    # 3. Identify tied overall scores across candidates
    score_rounded_counts = Counter(round(item[2], 2) for item in sorted_tuples)

    # 4. Assign ranks based on chosen ranking method
    ranked_results: List[Dict[str, Any]] = []
    prev_overall: Optional[float] = None
    prev_rank: int = 1

    for idx, (cand_copy, cid, overall, req_pct, cov_pct, sim_pct) in enumerate(sorted_tuples):
        display_order = idx + 1
        is_tied = score_rounded_counts[round(overall, 2)] > 1

        if idx == 0:
            assigned_rank = 1
        else:
            # Check if tied with previous candidate on primary overall score
            if math.isclose(overall, prev_overall, abs_tol=1e-5):
                if ranking_method == "sequential":
                    assigned_rank = display_order
                else:
                    assigned_rank = prev_rank
            else:
                if ranking_method == "competition":
                    assigned_rank = display_order  # 1-based index (e.g. 1, 1, 3)
                elif ranking_method == "dense":
                    assigned_rank = prev_rank + 1  # (e.g. 1, 1, 2)
                else:  # sequential
                    assigned_rank = display_order

        prev_overall = overall
        prev_rank = assigned_rank

        # Attach rank attributes without altering original score fields
        cand_copy["rank"] = assigned_rank
        cand_copy["display_order"] = display_order
        cand_copy["is_tied"] = is_tied

        # Expose canonical matching metrics at top-level for transparent access
        cand_copy["required_skill_match_percentage"] = req_pct
        cand_copy["overall_skill_coverage_percentage"] = cov_pct
        cand_copy["text_similarity_percentage"] = sim_pct

        ranked_results.append(cand_copy)

    return ranked_results


def get_ranking_summary(ranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics and distribution metrics for a ranked candidate set.

    Safely handles empty candidate lists without raising errors.

    Args:
        ranked_candidates: List of ranked candidate dictionaries produced by rank_candidates().

    Returns:
        Dictionary containing summary statistics:
        - total_candidates (int)
        - highest_score (Optional[float])
        - lowest_score (Optional[float])
        - mean_score (Optional[float])
        - median_score (Optional[float])
        - review_category_counts (Dict[str, int])
        - tied_score_count (int): Count of distinct overall scores that are tied.
        - tied_candidate_count (int): Total count of candidates involved in score ties.
        - ranking_method_used (Optional[str])

    Raises:
        TypeError: If ranked_candidates is not a list.
    """
    if not isinstance(ranked_candidates, list):
        raise TypeError(
            f"ranked_candidates must be a list, got {type(ranked_candidates).__name__}."
        )

    if not ranked_candidates:
        return {
            "total_candidates": 0,
            "highest_score": None,
            "lowest_score": None,
            "mean_score": None,
            "median_score": None,
            "review_category_counts": {},
            "category_counts": {},
            "tied_score_count": 0,
            "tied_candidate_count": 0,
            "highest_overall_score": None,
            "lowest_overall_score": None,
            "mean_overall_score": None,
            "median_overall_score": None,
        }

    scores = [float(c["overall_score"]) for c in ranked_candidates]
    categories = [c.get("review_category", "Unknown") for c in ranked_candidates]

    category_counts = dict(Counter(categories))

    # Identify ties on overall score (rounded to 2 decimal places)
    rounded_scores = [round(s, 2) for s in scores]
    score_counts = Counter(rounded_scores)
    tied_scores = {score: count for score, count in score_counts.items() if count > 1}

    highest = round(max(scores), 2)
    lowest = round(min(scores), 2)
    mean_val = round(mean(scores), 2)
    median_val = round(median(scores), 2)
    tied_score_count = len(tied_scores)
    tied_candidate_count = sum(tied_scores.values())

    return {
        "total_candidates": len(ranked_candidates),
        "highest_score": highest,
        "lowest_score": lowest,
        "mean_score": mean_val,
        "median_score": median_val,
        "review_category_counts": category_counts,
        "category_counts": category_counts,  # Alias for convenience
        "tied_score_count": tied_score_count,
        "tied_candidate_count": tied_candidate_count,
        # Extended explicit field aliases
        "highest_overall_score": highest,
        "lowest_overall_score": lowest,
        "mean_overall_score": mean_val,
        "median_overall_score": median_val,
    }
