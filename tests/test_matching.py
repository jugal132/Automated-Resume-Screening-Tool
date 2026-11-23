"""Unit tests for the resume-to-job matching engine (src/matcher.py).

Covers skill matching, percentage calculation, TF-IDF cosine similarity,
candidate match aggregation, batch candidate matching, and integration tests
against the synthetic resume dataset.
"""

from pathlib import Path
import tempfile
import unittest

from src.job_parser import get_standardized_job_profile
from src.matcher import (
    calculate_preferred_skill_match,
    calculate_required_skill_match,
    calculate_skill_coverage,
    calculate_text_similarity,
    match_candidate,
    match_candidates,
    match_skills,
)


class TestMatchingEngine(unittest.TestCase):
    """Test suite for skill matching and TF-IDF similarity."""

    def setUp(self):
        """Create sample job profile and temporary directory for tests."""
        self.sample_profile = {
            "job_title": "Junior Python Developer",
            "job_description": "We are seeking a Junior Python Developer with SQL, Pandas, Git, FastAPI, and Flask.",
            "required_skills": ["Python", "SQL", "Pandas", "Git"],
            "preferred_skills": ["FastAPI", "Flask", "PostgreSQL", "Docker"],
            "minimum_experience": 0,
            "preferred_experience": 1,
            "education": ["Computer Science"],
            "responsibilities": ["Write Python code", "Optimize SQL queries"],
            "experience_evidence": ["Internship"],
        }
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Skill Matching Tests
    # -------------------------------------------------------------------------

    def test_all_required_skills_matched(self):
        """Verify 100% required skill match when candidate has all required skills."""
        cand_skills = ["Python", "SQL", "Pandas", "Git"]
        result = match_skills(cand_skills, self.sample_profile)
        self.assertEqual(result["matched_required_skills"], ["Python", "SQL", "Pandas", "Git"])
        self.assertEqual(result["missing_required_skills"], [])
        self.assertEqual(result["required_skill_match_percentage"], 100.0)

    def test_some_required_skills_missing(self):
        """Verify partial match when some required skills are missing."""
        cand_skills = ["Python", "SQL"]  # Missing Pandas, Git
        result = match_skills(cand_skills, self.sample_profile)
        self.assertEqual(result["matched_required_skills"], ["Python", "SQL"])
        self.assertEqual(result["missing_required_skills"], ["Pandas", "Git"])
        self.assertEqual(result["required_skill_match_percentage"], 50.0)

    def test_no_required_skills_matched(self):
        """Verify 0% match when no required skills are satisfied."""
        cand_skills = ["Java", "C++", "HTML"]
        result = match_skills(cand_skills, self.sample_profile)
        self.assertEqual(result["matched_required_skills"], [])
        self.assertEqual(result["missing_required_skills"], ["Python", "SQL", "Pandas", "Git"])
        self.assertEqual(result["required_skill_match_percentage"], 0.0)

    def test_all_preferred_skills_matched(self):
        """Verify 100% preferred skill match when all preferred skills are met."""
        cand_skills = ["Python", "SQL", "Pandas", "Git", "FastAPI", "Flask", "PostgreSQL", "Docker"]
        result = match_skills(cand_skills, self.sample_profile)
        self.assertEqual(result["matched_preferred_skills"], ["FastAPI", "Flask", "PostgreSQL", "Docker"])
        self.assertEqual(result["missing_preferred_skills"], [])
        self.assertEqual(result["preferred_skill_match_percentage"], 100.0)
        self.assertEqual(result["overall_skill_coverage_percentage"], 100.0)

    def test_some_preferred_skills_missing(self):
        """Verify partial preferred skill calculation."""
        cand_skills = ["FastAPI", "Docker"]  # 2 of 4 preferred
        result = match_skills(cand_skills, self.sample_profile)
        self.assertEqual(result["matched_preferred_skills"], ["FastAPI", "Docker"])
        self.assertEqual(result["missing_preferred_skills"], ["Flask", "PostgreSQL"])
        self.assertEqual(result["preferred_skill_match_percentage"], 50.0)

    def test_no_preferred_skills_defined(self):
        """Verify safe handling when preferred_skills list is empty."""
        profile = dict(self.sample_profile)
        profile["preferred_skills"] = []
        cand_skills = ["Python", "SQL"]
        result = match_skills(cand_skills, profile)
        self.assertEqual(result["total_preferred_skills"], 0)
        self.assertEqual(result["preferred_skill_match_percentage"], 100.0)
        # 2 of 4 required matched, 0 preferred: overall coverage = 2 / 4 = 50.0%
        self.assertEqual(result["overall_skill_coverage_percentage"], 50.0)

    def test_empty_candidate_skills(self):
        """Verify handling of candidate with no skills."""
        result = match_skills([], self.sample_profile)
        self.assertEqual(result["matched_required_skills"], [])
        self.assertEqual(result["required_skill_match_percentage"], 0.0)
        self.assertEqual(result["matched_preferred_skills"], [])
        self.assertEqual(result["preferred_skill_match_percentage"], 0.0)
        self.assertEqual(result["overall_skill_coverage_percentage"], 0.0)

    def test_duplicate_candidate_skills(self):
        """Verify that duplicate candidate skills do not artificially inflate match count."""
        cand_skills = ["Python", "python", "PYTHON", "SQL", "sql"]
        result = match_skills(cand_skills, self.sample_profile)
        self.assertEqual(result["matched_required_skills"], ["Python", "SQL"])
        self.assertEqual(result["required_skill_match_percentage"], 50.0)

    def test_case_insensitive_matching(self):
        """Verify case-insensitive matching between candidate and job profile."""
        cand_skills = ["python", "sql", "pandas", "git"]
        result = match_skills(cand_skills, self.sample_profile)
        self.assertEqual(result["required_skill_match_percentage"], 100.0)

    def test_alias_normalization_in_matching(self):
        """Verify that candidate skill aliases (e.g. Postgres -> PostgreSQL) match correctly."""
        cand_skills = ["Postgres", "Git Hub", "sklearn"]
        profile = {
            "required_skills": ["PostgreSQL", "GitHub"],
            "preferred_skills": ["Scikit-learn"],
        }
        result = match_skills(cand_skills, profile)
        self.assertEqual(result["matched_required_skills"], ["PostgreSQL", "GitHub"])
        self.assertEqual(result["matched_preferred_skills"], ["Scikit-learn"])
        self.assertEqual(result["required_skill_match_percentage"], 100.0)
        self.assertEqual(result["preferred_skill_match_percentage"], 100.0)

    def test_zero_required_and_preferred_requirements(self):
        """Verify safe handling when neither required nor preferred skills are defined."""
        profile = {"required_skills": [], "preferred_skills": []}
        result = match_skills(["Python"], profile)
        self.assertEqual(result["required_skill_match_percentage"], 100.0)
        self.assertEqual(result["preferred_skill_match_percentage"], 100.0)
        self.assertEqual(result["overall_skill_coverage_percentage"], 100.0)

    def test_skill_coverage_calculation_helper(self):
        """Verify calculate_skill_coverage helper formulas and validations."""
        # 4 required, 4 preferred; 2 required matched, 2 preferred matched -> 4/8 = 50.0%
        self.assertEqual(calculate_skill_coverage(2, 4, 2, 4), 50.0)
        # 0 total -> 100.0%
        self.assertEqual(calculate_skill_coverage(0, 0, 0, 0), 100.0)

        with self.assertRaises(ValueError):
            calculate_skill_coverage(5, 4, 0, 0)  # matched > total
        with self.assertRaises(ValueError):
            calculate_skill_coverage(-1, 4, 0, 0)  # negative count
        with self.assertRaises(TypeError):
            calculate_skill_coverage("2", 4, 0, 0)  # non-int

    # -------------------------------------------------------------------------
    # 2. Text Similarity Tests (TF-IDF & Cosine Similarity)
    # -------------------------------------------------------------------------

    def test_text_similarity_identical_text(self):
        """Verify that identical texts produce similarity score of 1.0 (100%)."""
        text = "Junior Python Developer with SQL Pandas Git and FastAPI."
        result = calculate_text_similarity(text, text)
        self.assertAlmostEqual(result["similarity_score"], 1.0, places=3)
        self.assertAlmostEqual(result["similarity_percentage"], 100.0, places=1)

    def test_text_similarity_related_text(self):
        """Verify that related texts produce positive similarity between 0 and 1."""
        resume = "Python Developer intern building RESTful APIs with Flask and PostgreSQL database."
        job = "Seeking Python Developer for backend REST APIs and PostgreSQL database development."
        result = calculate_text_similarity(resume, job)
        self.assertGreater(result["similarity_score"], 0.2)
        self.assertLess(result["similarity_score"], 1.0)
        self.assertGreater(result["similarity_percentage"], 20.0)

    def test_text_similarity_unrelated_text(self):
        """Verify that completely unrelated texts yield approximately 0 similarity."""
        resume = "Executive chef specializing in Italian pastry baking culinary arts."
        job = "Python software engineer building PostgreSQL microservices."
        result = calculate_text_similarity(resume, job)
        self.assertEqual(result["similarity_score"], 0.0)
        self.assertEqual(result["similarity_percentage"], 0.0)

    def test_text_similarity_empty_inputs(self):
        """Verify safe handling of empty strings and whitespace-only text."""
        empty_tests = [
            ("", "Some job description"),
            ("Some resume text", ""),
            ("", ""),
            ("   \n\t  ", "   "),
        ]
        for resume, job in empty_tests:
            with self.subTest(resume=resume, job=job):
                result = calculate_text_similarity(resume, job)
                self.assertEqual(result["similarity_score"], 0.0)
                self.assertEqual(result["similarity_percentage"], 0.0)

    def test_text_similarity_invalid_types(self):
        """Verify that non-string arguments raise TypeError."""
        with self.assertRaises(TypeError):
            calculate_text_similarity(None, "Valid job")
        with self.assertRaises(TypeError):
            calculate_text_similarity("Valid resume", 123)

    def test_text_similarity_no_usable_tokens(self):
        """Verify that texts containing only punctuation or stop words return 0.0 without crash."""
        result = calculate_text_similarity("!@#$%^&*()_+", ";:[]{}")
        self.assertEqual(result["similarity_score"], 0.0)
        self.assertEqual(result["similarity_percentage"], 0.0)

    # -------------------------------------------------------------------------
    # 3. Single Candidate Matching Integration
    # -------------------------------------------------------------------------

    def test_match_candidate_structured_output(self):
        """Verify that match_candidate returns all required fields and correct metrics."""
        resume = """
        ================================================================================
        CANDIDATE ID: CAND-001
        ================================================================================
        Summary: Python Developer with SQL, Pandas, Git, FastAPI, and Flask experience.
        """
        result = match_candidate(resume, self.sample_profile)
        self.assertEqual(result["candidate_id"], "CAND-001")
        self.assertIn("Python", result["matched_required_skills"])
        self.assertIn("SQL", result["matched_required_skills"])
        self.assertIn("Pandas", result["matched_required_skills"])
        self.assertIn("Git", result["matched_required_skills"])
        self.assertEqual(result["required_skill_match_percentage"], 100.0)
        self.assertIn("FastAPI", result["matched_preferred_skills"])
        self.assertIn("Flask", result["matched_preferred_skills"])
        self.assertGreater(result["text_similarity_score"], 0.0)
        self.assertGreater(result["text_similarity_percentage"], 0.0)

    # -------------------------------------------------------------------------
    # 4. Batch Candidate Matching Integration
    # -------------------------------------------------------------------------

    def test_match_candidates_batch(self):
        """Verify batch matching over multiple candidates."""
        cand1 = "CANDIDATE ID: CAND-A\nPython, SQL, Pandas, Git developer."
        cand2 = "CANDIDATE ID: CAND-B\nExperienced in Java, C++, and Docker."
        results = match_candidates([cand1, cand2], self.sample_profile)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["candidate_id"], "CAND-A")
        self.assertEqual(results[0]["required_skill_match_percentage"], 100.0)
        self.assertEqual(results[1]["candidate_id"], "CAND-B")
        self.assertEqual(results[1]["required_skill_match_percentage"], 0.0)

    def test_match_candidates_synthetic_dataset(self):
        """Verify batch matching against the 8 actual synthetic resume files."""
        resume_dir = Path("resumes/sample_resumes")
        resume_files = sorted(resume_dir.glob("CAND-*.txt"))
        self.assertEqual(len(resume_files), 8, "Must find exactly 8 synthetic resumes.")

        profile = get_standardized_job_profile()
        results = match_candidates(resume_files, profile)
        self.assertEqual(len(results), 8)

        # Confirm CAND-001 has 100% required skill match
        cand1_result = next(r for r in results if r["candidate_id"] == "CAND-001")
        self.assertEqual(cand1_result["required_skill_match_percentage"], 100.0)
        self.assertEqual(cand1_result["total_required_skills"], 4)

        # Confirm CAND-008 has lower match
        cand8_result = next(r for r in results if r["candidate_id"] == "CAND-008")
        self.assertEqual(cand8_result["required_skill_match_percentage"], 25.0)  # Only Python


if __name__ == "__main__":
    unittest.main()
