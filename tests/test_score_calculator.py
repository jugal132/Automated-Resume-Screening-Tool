"""Unit tests for the transparent weighted score calculator (src/score_calculator.py).

Covers skill sub-scores, experience rubrics, education rubrics, overall score synthesis,
weight validation, contribution sums, review category thresholds, and end-to-end batch scoring.
"""

from pathlib import Path
import unittest

from src.job_parser import get_standardized_job_profile
from src.matcher import match_candidate, match_candidates
from src.resume_parser import extract_resume_text
from src.score_calculator import (
    DEFAULT_WEIGHTS,
    calculate_education_score,
    calculate_experience_score,
    calculate_overall_score,
    calculate_skill_score,
    get_review_category,
    score_candidate,
    score_candidates,
    validate_review_categories,
    validate_weights,
)


class TestScoreCalculator(unittest.TestCase):
    """Test suite for the score calculator module."""

    # -------------------------------------------------------------------------
    # 1. Skill Component Calculations
    # -------------------------------------------------------------------------

    def test_calculate_skill_score_exact(self):
        """Verify skill score calculation with 70% required and 30% preferred."""
        # 100% required, 50% preferred -> 0.7*100 + 0.3*50 = 70 + 15 = 85.0
        score = calculate_skill_score(100.0, 50.0, required_weight=0.70, preferred_weight=0.30)
        self.assertEqual(score, 85.0)

    def test_calculate_skill_score_boundaries(self):
        """Verify skill score at 0% and 100% boundaries."""
        self.assertEqual(calculate_skill_score(0.0, 0.0), 0.0)
        self.assertEqual(calculate_skill_score(100.0, 100.0), 100.0)

    def test_calculate_skill_score_invalid_inputs(self):
        """Verify that out-of-range percentages and invalid weights raise errors."""
        with self.assertRaises(ValueError):
            calculate_skill_score(-10.0, 50.0)
        with self.assertRaises(ValueError):
            calculate_skill_score(110.0, 50.0)
        with self.assertRaises(ValueError):
            calculate_skill_score(50.0, 50.0, required_weight=0.8, preferred_weight=0.3)  # Sum != 1.0
        with self.assertRaises(TypeError):
            calculate_skill_score("100", 50.0)

    # -------------------------------------------------------------------------
    # 2. Weight Validation Tests
    # -------------------------------------------------------------------------

    def test_validate_weights_valid(self):
        """Verify that default weights validate successfully."""
        validate_weights(DEFAULT_WEIGHTS)

    def test_validate_weights_invalid_sum(self):
        """Verify that weights not summing to 1.0 raise ValueError."""
        invalid_w = dict(DEFAULT_WEIGHTS)
        invalid_w["required_skills"] = 0.50  # Total 1.15
        with self.assertRaises(ValueError):
            validate_weights(invalid_w)

    def test_validate_weights_negative_or_missing(self):
        """Verify negative weight or missing keys raise ValueError."""
        neg_w = dict(DEFAULT_WEIGHTS)
        neg_w["experience"] = -0.20
        with self.assertRaises(ValueError):
            validate_weights(neg_w)

        missing_w = {"required_skills": 0.5, "preferred_skills": 0.5}
        with self.assertRaises(ValueError):
            validate_weights(missing_w)

    # -------------------------------------------------------------------------
    # 3. Experience Scoring Rubric Tests
    # -------------------------------------------------------------------------

    def test_calculate_experience_score_explicit_tiers(self):
        """Verify experience scoring across predefined rubric tiers."""
        # Tier 1: Professional >= 1 year -> 100.0
        score, expl = calculate_experience_score(years=1.0, evidence_type="PROFESSIONAL")
        self.assertEqual(score, 100.0)
        self.assertIn(">= 1 year", expl)

        # Tier 2: 6-month software internship -> 85.0
        score, expl = calculate_experience_score(years=0.5, evidence_type="INTERNSHIP_MID")
        self.assertEqual(score, 85.0)

        # Tier 3: 3-month summer internship / adjacent -> 75.0
        score, expl = calculate_experience_score(years=0.25, evidence_type="INTERNSHIP_SHORT")
        self.assertEqual(score, 75.0)

        # Tier 4: Apprenticeship / Academic TA -> 70.0
        score, expl = calculate_experience_score(evidence_type="ACADEMIC_TA")
        self.assertEqual(score, 70.0)

        # Tier 5: Personal projects / coursework -> 50.0
        score, expl = calculate_experience_score(evidence_type="PERSONAL_PROJECTS")
        self.assertEqual(score, 50.0)

        # Tier 6: Non-technical administration -> 30.0
        score, expl = calculate_experience_score(evidence_type="NON_TECHNICAL")
        self.assertEqual(score, 30.0)

    def test_calculate_experience_score_unknown(self):
        """Verify that unknown/missing experience receives neutral 50.0 baseline score."""
        score, expl = calculate_experience_score(evidence_type="UNKNOWN")
        self.assertEqual(score, 50.0)
        self.assertIn("neutral", expl.lower())

        score_none, expl_none = calculate_experience_score(resume_text="")
        self.assertEqual(score_none, 50.0)

    # -------------------------------------------------------------------------
    # 4. Education Scoring Rubric Tests
    # -------------------------------------------------------------------------

    def test_calculate_education_score_explicit_tiers(self):
        """Verify education scoring across explicit degree tiers."""
        # Tier 1: Bachelor of Science in Computer Science -> 100.0
        score, expl = calculate_education_score(degree="Bachelor of Science", field="Computer Science")
        self.assertEqual(score, 100.0)
        self.assertIn("preferred discipline", expl.lower())

        # Tier 2: Associate in Web Development & IT -> 80.0
        score, expl = calculate_education_score(degree="Associate Degree", field="Information Technology")
        self.assertEqual(score, 80.0)

        # Tier 3: Adjacent Information Systems -> 80.0
        score, expl = calculate_education_score(degree="Bachelor", field="Information Systems")
        self.assertEqual(score, 80.0)

        # Tier 4: Quantitative STEM (Math/Physics) -> 75.0
        score, expl = calculate_education_score(degree="Bachelor", field="Mathematics")
        self.assertEqual(score, 75.0)

        # Tier 5: Non-technical degree (Communications) -> 40.0
        score, expl = calculate_education_score(degree="Bachelor of Arts", field="Communications")
        self.assertEqual(score, 40.0)

    def test_calculate_education_score_missing(self):
        """Verify that missing education is treated neutrally (50.0) per non-exclusionary policy."""
        score, expl = calculate_education_score(resume_text="")
        self.assertEqual(score, 50.0)
        self.assertIn("non-exclusionary", expl.lower())

    # -------------------------------------------------------------------------
    # 5. Overall Weighted Score & Contribution Tests
    # -------------------------------------------------------------------------

    def test_calculate_overall_score_exact_formula(self):
        """Verify that calculate_overall_score implements the exact 5-component formula."""
        # 100 req (35pts) + 50 pref (7.5pts) + 80 exp (16pts) + 40 sim (8pts) + 100 edu (10pts)
        # Total = 35 + 7.5 + 16 + 8 + 10 = 76.5
        overall, contribs = calculate_overall_score(
            required_match_pct=100.0,
            preferred_match_pct=50.0,
            experience_score=80.0,
            text_similarity_pct=40.0,
            education_score=100.0,
        )
        self.assertAlmostEqual(overall, 76.5, places=2)
        self.assertAlmostEqual(contribs["required_skills"], 35.0, places=2)
        self.assertAlmostEqual(contribs["preferred_skills"], 7.5, places=2)
        self.assertAlmostEqual(contribs["experience"], 16.0, places=2)
        self.assertAlmostEqual(contribs["text_similarity"], 8.0, places=2)
        self.assertAlmostEqual(contribs["education"], 10.0, places=2)

    def test_contributions_sum_to_overall_score(self):
        """Verify that the sum of weighted contributions equals the overall score."""
        overall, contribs = calculate_overall_score(
            required_match_pct=75.0,
            preferred_match_pct=33.33,
            experience_score=85.0,
            text_similarity_pct=52.4,
            education_score=80.0,
        )
        total_contribs = sum(contribs.values())
        self.assertAlmostEqual(overall, total_contribs, places=3)

    def test_overall_score_clamped(self):
        """Verify that overall scores are clamped between 0 and 100."""
        overall_min, _ = calculate_overall_score(0, 0, 0, 0, 0)
        self.assertEqual(overall_min, 0.0)
        overall_max, _ = calculate_overall_score(100, 100, 100, 100, 100)
        self.assertEqual(overall_max, 100.0)

    # -------------------------------------------------------------------------
    # 6. Review Category Threshold Tests
    # -------------------------------------------------------------------------

    def test_get_review_category(self):
        """Verify organizational review category assignments."""
        self.assertEqual(get_review_category(85.0), "Priority Review")
        self.assertEqual(get_review_category(80.0), "Priority Review")
        self.assertEqual(get_review_category(79.99), "Review")
        self.assertEqual(get_review_category(60.0), "Review")
        self.assertEqual(get_review_category(59.99), "Additional Review")
        self.assertEqual(get_review_category(0.0), "Additional Review")

    def test_validate_review_categories(self):
        """Verify validation of review category thresholds."""
        with self.assertRaises(ValueError):
            # Not in descending order
            validate_review_categories([("Review", 50.0), ("Priority", 80.0)])
        with self.assertRaises(ValueError):
            # Out of 0-100 bounds
            validate_review_categories([("Priority", 120.0)])

    # -------------------------------------------------------------------------
    # 7. Single Candidate Scoring Integration
    # -------------------------------------------------------------------------

    def test_score_candidate_full_structure(self):
        """Verify that score_candidate returns all required explainability fields."""
        match_result = {
            "candidate_id": "CAND-001",
            "required_skill_match_percentage": 100.0,
            "preferred_skill_match_percentage": 87.5,
            "text_similarity_percentage": 53.56,
        }
        resume_text = "Bachelor of Science in Computer Science. Python Developer Intern (June 2023 - May 2024)."
        result = score_candidate(match_result, resume_text=resume_text)

        expected_fields = [
            "candidate_id",
            "required_skill_score",
            "preferred_skill_score",
            "skill_component_score",
            "experience_score",
            "experience_explanation",
            "education_score",
            "education_explanation",
            "text_similarity_score",
            "component_weights",
            "weighted_contributions",
            "overall_score",
            "review_category",
            "score_explanation",
        ]
        for field in expected_fields:
            self.assertIn(field, result, f"Missing field '{field}' in score result.")

        self.assertEqual(result["candidate_id"], "CAND-001")
        self.assertEqual(result["review_category"], "Priority Review")
        self.assertGreater(result["overall_score"], 80.0)

    # -------------------------------------------------------------------------
    # 8. Batch Candidate Scoring & Synthetic Dataset Integration
    # -------------------------------------------------------------------------

    def test_score_candidates_all_synthetic_resumes(self):
        """Verify full scoring pipeline on all 8 synthetic resumes."""
        resume_dir = Path("resumes/sample_resumes")
        resume_files = sorted(resume_dir.glob("CAND-*.txt"))
        self.assertEqual(len(resume_files), 8)

        profile = get_standardized_job_profile()
        matches = match_candidates(resume_files, profile)
        self.assertEqual(len(matches), 8)

        resume_texts = {f.stem: extract_resume_text(f) for f in resume_files}
        scores = score_candidates(matches, resume_texts=resume_texts)
        self.assertEqual(len(scores), 8)

        # Confirm CAND-001 is in Priority Review
        cand1 = next(s for s in scores if s["candidate_id"] == "CAND-001")
        self.assertEqual(cand1["review_category"], "Priority Review")
        self.assertAlmostEqual(cand1["overall_score"], 88.84, places=1)

        # Confirm CAND-008 is in Additional Review
        cand8 = next(s for s in scores if s["candidate_id"] == "CAND-008")
        self.assertEqual(cand8["review_category"], "Additional Review")
        self.assertLess(cand8["overall_score"], 35.0)

        # Confirm all scores have non-empty explanations
        for s in scores:
            self.assertTrue(len(s["score_explanation"]) > 50)
            self.assertEqual(
                round(sum(s["weighted_contributions"].values()), 2),
                s["overall_score"],
            )


if __name__ == "__main__":
    unittest.main()
