"""Unit tests for the Candidate Ranking Engine (src/ranking.py).

Covers overall score ordering, deterministic tie-breakers (required skills, skill coverage,
text similarity, candidate ID), competition/dense/sequential rank assignments, input validation,
non-mutation guarantees, ranking summaries, and integration with the Phase 10 synthetic dataset.
"""

import copy
from pathlib import Path
import unittest

from src.job_parser import get_standardized_job_profile
from src.matcher import match_candidates
from src.ranking import (
    get_ranking_summary,
    rank_candidates,
    validate_candidate_record,
)
from src.resume_parser import extract_resume_text
from src.score_calculator import score_candidates


class TestCandidateRanking(unittest.TestCase):
    """Test suite for candidate ranking, tie-breaking, and validation."""

    def setUp(self):
        """Create sample scored candidate fixtures for test isolation."""
        self.sample_candidate_a = {
            "candidate_id": "CAND-001",
            "overall_score": 85.0,
            "required_skill_match_percentage": 100.0,
            "overall_skill_coverage_percentage": 90.0,
            "text_similarity_percentage": 60.0,
            "review_category": "Priority Review",
            "score_explanation": "Solid candidate with strong Python skills.",
        }
        self.sample_candidate_b = {
            "candidate_id": "CAND-002",
            "overall_score": 75.0,
            "required_skill_match_percentage": 75.0,
            "overall_skill_coverage_percentage": 70.0,
            "text_similarity_percentage": 50.0,
            "review_category": "Review",
            "score_explanation": "Good backend experience.",
        }
        self.sample_candidate_c = {
            "candidate_id": "CAND-003",
            "overall_score": 55.0,
            "required_skill_match_percentage": 50.0,
            "overall_skill_coverage_percentage": 40.0,
            "text_similarity_percentage": 35.0,
            "review_category": "Additional Review",
            "score_explanation": "Requires additional review.",
        }

    # -------------------------------------------------------------------------
    # 1. Primary Ranking & Ordering
    # -------------------------------------------------------------------------

    def test_ranking_by_overall_score(self):
        """Verify ranking candidates correctly orders by overall score descending."""
        candidates = [self.sample_candidate_c, self.sample_candidate_a, self.sample_candidate_b]
        ranked = rank_candidates(candidates)

        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["candidate_id"], "CAND-001")
        self.assertEqual(ranked[1]["candidate_id"], "CAND-002")
        self.assertEqual(ranked[2]["candidate_id"], "CAND-003")

        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["rank"], 2)
        self.assertEqual(ranked[2]["rank"], 3)

    def test_descending_ordering(self):
        """Verify explicit descending=True order (highest score first)."""
        candidates = [self.sample_candidate_b, self.sample_candidate_a]
        ranked = rank_candidates(candidates, descending=True)

        self.assertEqual(ranked[0]["candidate_id"], "CAND-001")
        self.assertGreater(ranked[0]["overall_score"], ranked[1]["overall_score"])

    def test_ascending_ordering(self):
        """Verify ascending=True order (lowest score first)."""
        candidates = [self.sample_candidate_a, self.sample_candidate_c, self.sample_candidate_b]
        ranked = rank_candidates(candidates, descending=False)

        self.assertEqual(ranked[0]["candidate_id"], "CAND-003")
        self.assertEqual(ranked[1]["candidate_id"], "CAND-002")
        self.assertEqual(ranked[2]["candidate_id"], "CAND-001")
        self.assertLess(ranked[0]["overall_score"], ranked[1]["overall_score"])
        self.assertLess(ranked[1]["overall_score"], ranked[2]["overall_score"])

    # -------------------------------------------------------------------------
    # 2. Deterministic Tie-Breaking Hierarchies
    # -------------------------------------------------------------------------

    def test_required_skill_match_tie_break(self):
        """Verify tie-breaker 1: equal score broken by higher required skill match."""
        cand1 = copy.deepcopy(self.sample_candidate_a)
        cand2 = copy.deepcopy(self.sample_candidate_b)
        cand1["overall_score"] = 80.0
        cand2["overall_score"] = 80.0
        cand1["required_skill_match_percentage"] = 100.0
        cand2["required_skill_match_percentage"] = 75.0

        ranked = rank_candidates([cand2, cand1])

        # cand1 wins display order due to higher required skill match
        self.assertEqual(ranked[0]["candidate_id"], "CAND-001")
        self.assertEqual(ranked[0]["display_order"], 1)
        self.assertEqual(ranked[1]["candidate_id"], "CAND-002")
        self.assertEqual(ranked[1]["display_order"], 2)

        # In competition ranking, equal overall score shares rank 1
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["rank"], 1)

    def test_overall_skill_coverage_tie_break(self):
        """Verify tie-breaker 2: equal score and required skills broken by overall coverage."""
        cand1 = copy.deepcopy(self.sample_candidate_a)
        cand2 = copy.deepcopy(self.sample_candidate_b)
        cand1["overall_score"] = 80.0
        cand2["overall_score"] = 80.0
        cand1["required_skill_match_percentage"] = 100.0
        cand2["required_skill_match_percentage"] = 100.0
        cand1["overall_skill_coverage_percentage"] = 90.0
        cand2["overall_skill_coverage_percentage"] = 75.0

        ranked = rank_candidates([cand2, cand1])

        self.assertEqual(ranked[0]["candidate_id"], "CAND-001")
        self.assertEqual(ranked[0]["display_order"], 1)
        self.assertEqual(ranked[1]["candidate_id"], "CAND-002")
        self.assertEqual(ranked[1]["display_order"], 2)

    def test_tfidf_tie_break(self):
        """Verify tie-breaker 3: equal score, required, and coverage broken by text similarity."""
        cand1 = copy.deepcopy(self.sample_candidate_a)
        cand2 = copy.deepcopy(self.sample_candidate_b)
        cand1["overall_score"] = 80.0
        cand2["overall_score"] = 80.0
        cand1["required_skill_match_percentage"] = 100.0
        cand2["required_skill_match_percentage"] = 100.0
        cand1["overall_skill_coverage_percentage"] = 85.0
        cand2["overall_skill_coverage_percentage"] = 85.0
        cand1["text_similarity_percentage"] = 65.0
        cand2["text_similarity_percentage"] = 45.0

        ranked = rank_candidates([cand2, cand1])

        self.assertEqual(ranked[0]["candidate_id"], "CAND-001")
        self.assertEqual(ranked[0]["display_order"], 1)
        self.assertEqual(ranked[1]["candidate_id"], "CAND-002")
        self.assertEqual(ranked[1]["display_order"], 2)

    def test_candidate_id_tie_break(self):
        """Verify tie-breaker 4: all metrics equal broken deterministically by ID ascending."""
        cand1 = copy.deepcopy(self.sample_candidate_a)
        cand2 = copy.deepcopy(self.sample_candidate_b)
        cand1["candidate_id"] = "CAND-002"
        cand2["candidate_id"] = "CAND-001"
        cand1["overall_score"] = 80.0
        cand2["overall_score"] = 80.0
        cand1["required_skill_match_percentage"] = 100.0
        cand2["required_skill_match_percentage"] = 100.0
        cand1["overall_skill_coverage_percentage"] = 85.0
        cand2["overall_skill_coverage_percentage"] = 85.0
        cand1["text_similarity_percentage"] = 50.0
        cand2["text_similarity_percentage"] = 50.0

        ranked = rank_candidates([cand1, cand2])

        # CAND-001 comes before CAND-002 ascending
        self.assertEqual(ranked[0]["candidate_id"], "CAND-001")
        self.assertEqual(ranked[1]["candidate_id"], "CAND-002")

    # -------------------------------------------------------------------------
    # 3. Competition / Dense / Sequential Rank Assignments
    # -------------------------------------------------------------------------

    def test_equal_overall_score_rank_handling(self):
        """Verify competition ranking assigns 1, 1, 3 for a two-way tie at the top."""
        cand1 = copy.deepcopy(self.sample_candidate_a)
        cand2 = copy.deepcopy(self.sample_candidate_b)
        cand3 = copy.deepcopy(self.sample_candidate_c)
        cand1["overall_score"] = 80.0
        cand2["overall_score"] = 80.0
        cand3["overall_score"] = 70.0

        ranked = rank_candidates([cand1, cand2, cand3], ranking_method="competition")

        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["rank"], 1)
        self.assertEqual(ranked[2]["rank"], 3)
        self.assertTrue(ranked[0]["is_tied"])
        self.assertTrue(ranked[1]["is_tied"])
        self.assertFalse(ranked[2]["is_tied"])

    def test_multiple_tie_cases(self):
        """Verify handling of multiple tie clusters across a candidate cohort."""
        # 2 candidates at 90.0, 3 candidates at 75.0, 1 candidate at 60.0
        data = [
            {"candidate_id": "CAND-01", "overall_score": 90.0, "required_skill_match_percentage": 100.0, "overall_skill_coverage_percentage": 90.0, "text_similarity_percentage": 50.0},
            {"candidate_id": "CAND-02", "overall_score": 90.0, "required_skill_match_percentage": 80.0, "overall_skill_coverage_percentage": 85.0, "text_similarity_percentage": 50.0},
            {"candidate_id": "CAND-03", "overall_score": 75.0, "required_skill_match_percentage": 100.0, "overall_skill_coverage_percentage": 80.0, "text_similarity_percentage": 40.0},
            {"candidate_id": "CAND-04", "overall_score": 75.0, "required_skill_match_percentage": 75.0, "overall_skill_coverage_percentage": 70.0, "text_similarity_percentage": 40.0},
            {"candidate_id": "CAND-05", "overall_score": 75.0, "required_skill_match_percentage": 50.0, "overall_skill_coverage_percentage": 60.0, "text_similarity_percentage": 40.0},
            {"candidate_id": "CAND-06", "overall_score": 60.0, "required_skill_match_percentage": 50.0, "overall_skill_coverage_percentage": 50.0, "text_similarity_percentage": 30.0},
        ]

        competition_ranked = rank_candidates(data, ranking_method="competition")
        ranks = [c["rank"] for c in competition_ranked]
        # Competition: 1, 1, 3, 3, 3, 6
        self.assertEqual(ranks, [1, 1, 3, 3, 3, 6])

        dense_ranked = rank_candidates(data, ranking_method="dense")
        dense_ranks = [c["rank"] for c in dense_ranked]
        # Dense: 1, 1, 2, 2, 2, 3
        self.assertEqual(dense_ranks, [1, 1, 2, 2, 2, 3])

        sequential_ranked = rank_candidates(data, ranking_method="sequential")
        sequential_ranks = [c["rank"] for c in sequential_ranked]
        # Sequential: 1, 2, 3, 4, 5, 6
        self.assertEqual(sequential_ranks, [1, 2, 3, 4, 5, 6])

    # -------------------------------------------------------------------------
    # 4. Input Validation & Error Handling
    # -------------------------------------------------------------------------

    def test_empty_candidate_list(self):
        """Verify that an empty candidate list returns an empty list safely."""
        self.assertEqual(rank_candidates([]), [])
        summary = get_ranking_summary([])
        self.assertEqual(summary["total_candidates"], 0)
        self.assertIsNone(summary["highest_score"])

    def test_invalid_input_type(self):
        """Verify TypeError for non-list candidates or non-dict candidate items."""
        with self.assertRaises(TypeError):
            rank_candidates("not a list")  # type: ignore
        with self.assertRaises(TypeError):
            rank_candidates([1, 2, 3])  # type: ignore

    def test_missing_candidate_id(self):
        """Verify ValueError when candidate_id is missing or empty."""
        cand = copy.deepcopy(self.sample_candidate_a)
        del cand["candidate_id"]
        with self.assertRaises(ValueError):
            rank_candidates([cand])

        cand["candidate_id"] = "   "
        with self.assertRaises(ValueError):
            rank_candidates([cand])

    def test_invalid_overall_score(self):
        """Verify TypeError when overall_score is non-numeric, None, or boolean."""
        cand = copy.deepcopy(self.sample_candidate_a)
        cand["overall_score"] = "eighty"
        with self.assertRaises(TypeError):
            rank_candidates([cand])

        cand["overall_score"] = True
        with self.assertRaises(TypeError):
            rank_candidates([cand])

        cand["overall_score"] = None
        with self.assertRaises(TypeError):
            rank_candidates([cand])

    def test_score_outside_0_to_100(self):
        """Verify ValueError when overall_score is outside [0, 100]."""
        cand = copy.deepcopy(self.sample_candidate_a)
        cand["overall_score"] = -5.0
        with self.assertRaises(ValueError):
            rank_candidates([cand])

        cand["overall_score"] = 105.0
        with self.assertRaises(ValueError):
            rank_candidates([cand])

    def test_invalid_matching_percentages(self):
        """Verify validation of required, coverage, and similarity percentages."""
        # Test out-of-range required skills
        cand = copy.deepcopy(self.sample_candidate_a)
        cand["required_skill_match_percentage"] = 150.0
        with self.assertRaises(ValueError):
            rank_candidates([cand])

        # Test out-of-range coverage
        cand = copy.deepcopy(self.sample_candidate_a)
        cand["overall_skill_coverage_percentage"] = -1.0
        with self.assertRaises(ValueError):
            rank_candidates([cand])

        # Test boolean similarity
        cand = copy.deepcopy(self.sample_candidate_a)
        cand["text_similarity_percentage"] = False
        with self.assertRaises(TypeError):
            rank_candidates([cand])

        # Test missing metric without fallback
        del cand["text_similarity_percentage"]
        with self.assertRaises(ValueError):
            rank_candidates([cand])

    def test_duplicate_candidate_ids(self):
        """Verify ValueError when duplicate candidate IDs are present."""
        cand1 = copy.deepcopy(self.sample_candidate_a)
        cand2 = copy.deepcopy(self.sample_candidate_b)
        cand2["candidate_id"] = "CAND-001"  # duplicate of cand1

        with self.assertRaises(ValueError) as ctx:
            rank_candidates([cand1, cand2])
        self.assertIn("Duplicate candidate ID", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 5. Immutability & Preservation
    # -------------------------------------------------------------------------

    def test_original_inputs_remain_unchanged(self):
        """Verify that ranking returns new copies and does not mutate input dictionaries."""
        cand_orig = copy.deepcopy(self.sample_candidate_a)
        input_list = [self.sample_candidate_a]

        ranked = rank_candidates(input_list)

        # Ranked list should have new fields
        self.assertIn("rank", ranked[0])
        self.assertIn("display_order", ranked[0])

        # Input candidate must NOT have been mutated
        self.assertNotIn("rank", self.sample_candidate_a)
        self.assertNotIn("display_order", self.sample_candidate_a)
        self.assertEqual(self.sample_candidate_a, cand_orig)

    def test_all_candidate_fields_preserved(self):
        """Verify all original fields (scores, explanations, reviews) are preserved intact."""
        ranked = rank_candidates([self.sample_candidate_a])
        result = ranked[0]

        self.assertEqual(result["candidate_id"], "CAND-001")
        self.assertEqual(result["overall_score"], 85.0)
        self.assertEqual(result["review_category"], "Priority Review")
        self.assertEqual(result["score_explanation"], "Solid candidate with strong Python skills.")
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["display_order"], 1)

    # -------------------------------------------------------------------------
    # 6. Ranking Summary Calculations
    # -------------------------------------------------------------------------

    def test_ranking_summary_calculations(self):
        """Verify ranking summary statistics across multiple candidates."""
        candidates = [
            self.sample_candidate_a,  # 85.0, Priority Review
            self.sample_candidate_b,  # 75.0, Review
            self.sample_candidate_c,  # 55.0, Additional Review
        ]
        ranked = rank_candidates(candidates)
        summary = get_ranking_summary(ranked)

        self.assertEqual(summary["total_candidates"], 3)
        self.assertEqual(summary["highest_score"], 85.0)
        self.assertEqual(summary["lowest_score"], 55.0)
        self.assertEqual(summary["mean_score"], 71.67)
        self.assertEqual(summary["median_score"], 75.0)
        self.assertEqual(summary["review_category_counts"]["Priority Review"], 1)
        self.assertEqual(summary["review_category_counts"]["Review"], 1)
        self.assertEqual(summary["review_category_counts"]["Additional Review"], 1)
        self.assertEqual(summary["tied_score_count"], 0)
        self.assertEqual(summary["tied_candidate_count"], 0)

    # -------------------------------------------------------------------------
    # 7. Integration with Phase 10 Scored Candidates
    # -------------------------------------------------------------------------

    def test_rank_all_8_synthetic_candidates(self):
        """End-to-end integration test ranking all 8 synthetic resumes from Phase 10."""
        resume_dir = Path("resumes/sample_resumes")
        resume_files = sorted(resume_dir.glob("CAND-*.txt"))
        self.assertEqual(len(resume_files), 8)

        # 1. Match candidates against standardized profile
        profile = get_standardized_job_profile()
        matches = match_candidates(resume_files, profile)
        self.assertEqual(len(matches), 8)

        # 2. Score candidates
        resume_texts = {f.stem: extract_resume_text(f) for f in resume_files}
        scores = score_candidates(matches, resume_texts=resume_texts)
        self.assertEqual(len(scores), 8)

        # 3. Rank candidates
        ranked = rank_candidates(scores)
        self.assertEqual(len(ranked), 8)

        # Verify ranking order matches expectations
        # Expected top candidate is CAND-001 (Priority Review)
        self.assertEqual(ranked[0]["candidate_id"], "CAND-001")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["review_category"], "Priority Review")
        self.assertAlmostEqual(ranked[0]["overall_score"], 88.83, places=1)

        # Verify CAND-008 is at the bottom
        self.assertEqual(ranked[-1]["candidate_id"], "CAND-008")
        self.assertEqual(ranked[-1]["rank"], 8)
        self.assertEqual(ranked[-1]["review_category"], "Additional Review")
        self.assertAlmostEqual(ranked[-1]["overall_score"], 24.84, places=1)

        # Verify descending order across all ranks
        for i in range(len(ranked) - 1):
            self.assertGreaterEqual(ranked[i]["overall_score"], ranked[i + 1]["overall_score"])

        # Verify summary on synthetic candidates
        summary = get_ranking_summary(ranked)
        self.assertEqual(summary["total_candidates"], 8)
        self.assertEqual(summary["highest_score"], 88.83)
        self.assertEqual(summary["lowest_score"], 24.84)
        self.assertEqual(summary["review_category_counts"]["Priority Review"], 1)
        self.assertEqual(summary["review_category_counts"]["Review"], 3)
        self.assertEqual(summary["review_category_counts"]["Additional Review"], 4)


if __name__ == "__main__":
    unittest.main()
