"""Unit and integration tests for the Streamlit Dashboard (main.py).

Covers job description discovery, pipeline execution on sample and custom resumes,
resilient per-file error isolation, charting helpers, and input validation.
"""

import io
from pathlib import Path
import tempfile
import unittest

from main import (
    discover_job_descriptions,
    render_category_donut_chart,
    render_component_breakdown_chart,
    render_score_distribution_chart,
    run_screening_pipeline,
)
from src.job_parser import get_standardized_job_profile


class MockUploadedFile:
    """Mock Streamlit UploadedFile object with name and getbuffer() interface."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._buffer = io.BytesIO(content)

    def getbuffer(self) -> memoryview:
        return self._buffer.getbuffer()


class TestDashboard(unittest.TestCase):
    """Test suite for Streamlit dashboard backend orchestration."""

    def setUp(self):
        """Locate test fixture paths."""
        self.job_dir = Path("data/job_descriptions")
        self.job_txt = self.job_dir / "junior_python_developer.txt"
        self.job_json = self.job_dir / "junior_python_developer_requirements.json"

    # -------------------------------------------------------------------------
    # 1. Job Description Discovery Tests
    # -------------------------------------------------------------------------

    def test_discover_job_descriptions_found(self):
        """Verify that discover_job_descriptions identifies available job files."""
        jobs = discover_job_descriptions(self.job_dir)
        self.assertIn("Junior Python Developer", jobs)
        self.assertTrue(jobs["Junior Python Developer"]["txt"].exists())
        self.assertTrue(jobs["Junior Python Developer"]["json"].exists())

    def test_discover_job_descriptions_nonexistent_dir(self):
        """Verify that nonexistent directories return an empty dictionary safely."""
        jobs = discover_job_descriptions(Path("nonexistent/dir"))
        self.assertEqual(jobs, {})

    # -------------------------------------------------------------------------
    # 2. Pipeline Execution on Sample Resumes
    # -------------------------------------------------------------------------

    def test_run_screening_pipeline_sample_dataset(self):
        """Verify end-to-end screening pipeline execution on the 8 synthetic resumes."""
        results = run_screening_pipeline(
            job_txt_path=self.job_txt,
            job_json_path=self.job_json,
            source_type="sample",
        )

        ranked = results["ranked_candidates"]
        summary = results["summary"]
        failed = results["failed_files"]

        # Ensure all 8 candidates are ranked
        self.assertEqual(len(ranked), 8)
        self.assertEqual(len(failed), 0)

        # CAND-001 should be rank 1 in Priority Review
        self.assertEqual(ranked[0]["candidate_id"], "CAND-001")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["review_category"], "Priority Review")
        self.assertAlmostEqual(ranked[0]["overall_score"], 88.83, places=1)

        # CAND-008 should be rank 8 in Additional Review
        self.assertEqual(ranked[-1]["candidate_id"], "CAND-008")
        self.assertEqual(ranked[-1]["rank"], 8)
        self.assertEqual(ranked[-1]["review_category"], "Additional Review")

        # Verify summary metrics
        self.assertEqual(summary["total_candidates"], 8)
        self.assertEqual(summary["highest_score"], 88.83)
        self.assertEqual(summary["lowest_score"], 24.84)
        self.assertEqual(summary["review_category_counts"]["Priority Review"], 1)
        self.assertEqual(summary["review_category_counts"]["Review"], 3)
        self.assertEqual(summary["review_category_counts"]["Additional Review"], 4)

    # -------------------------------------------------------------------------
    # 3. Pipeline Execution on Uploaded Resumes
    # -------------------------------------------------------------------------

    def test_run_screening_pipeline_custom_upload_valid(self):
        """Verify pipeline execution with mock uploaded resume files."""
        resume_1 = (
            "CANDIDATE: CAND-UP-01\n"
            "SUMMARY: Python developer with SQL and Git.\n"
            "SKILLS: Python, SQL, Pandas, Git, FastAPI, Flask\n"
            "EXPERIENCE: 1 year software development internship at Tech Co.\n"
            "EDUCATION: Bachelor of Science in Computer Science\n"
        ).encode("utf-8")

        resume_2 = (
            "CANDIDATE: CAND-UP-02\n"
            "SUMMARY: Communications specialist.\n"
            "SKILLS: Git\n"
            "EXPERIENCE: Non-technical administrative assistant.\n"
            "EDUCATION: Bachelor of Arts in Communications\n"
        ).encode("utf-8")

        mock_files = [
            MockUploadedFile("CAND-UP-01.txt", resume_1),
            MockUploadedFile("CAND-UP-02.txt", resume_2),
        ]

        results = run_screening_pipeline(
            job_txt_path=self.job_txt,
            job_json_path=self.job_json,
            source_type="upload",
            uploaded_files=mock_files,
        )

        ranked = results["ranked_candidates"]
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["candidate_id"], "CAND-UP-01")
        self.assertEqual(ranked[1]["candidate_id"], "CAND-UP-02")
        self.assertGreater(ranked[0]["overall_score"], ranked[1]["overall_score"])

    def test_run_screening_pipeline_empty_upload(self):
        """Verify that empty uploads raise ValueError prompting user action."""
        with self.assertRaises(ValueError) as ctx:
            run_screening_pipeline(
                job_txt_path=self.job_txt,
                job_json_path=self.job_json,
                source_type="upload",
                uploaded_files=[],
            )
        self.assertIn("Please upload at least one candidate resume", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. Resilient Per-File Error Isolation
    # -------------------------------------------------------------------------

    def test_run_screening_pipeline_corrupt_file_isolation(self):
        """Verify that corrupt or empty files do not crash the valid batch."""
        valid_resume = (
            "CANDIDATE: CAND-VALID-01\n"
            "SKILLS: Python, SQL, Pandas, Git\n"
            "EXPERIENCE: Junior Python developer with 1 year experience.\n"
            "EDUCATION: Bachelor of Science in Computer Science\n"
        ).encode("utf-8")

        empty_resume = b""  # 0-byte file (EmptyFileError)

        mock_files = [
            MockUploadedFile("valid_resume.txt", valid_resume),
            MockUploadedFile("empty_resume.txt", empty_resume),
        ]

        results = run_screening_pipeline(
            job_txt_path=self.job_txt,
            job_json_path=self.job_json,
            source_type="upload",
            uploaded_files=mock_files,
        )

        ranked = results["ranked_candidates"]
        failed = results["failed_files"]

        # Valid resume should still succeed
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["candidate_id"], "valid_resume")

        # Corrupted resume should be safely recorded in failed_files
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["filename"], "empty_resume.txt")
        self.assertIn("empty", failed[0]["error"].lower())

    # -------------------------------------------------------------------------
    # 5. Charting Functions Safe Execution
    # -------------------------------------------------------------------------

    def test_charting_functions_execute_safely(self):
        """Verify that charting functions run without errors and handle empty data."""
        sample_results = run_screening_pipeline(
            job_txt_path=self.job_txt,
            job_json_path=self.job_json,
            source_type="sample",
        )
        ranked = sample_results["ranked_candidates"]
        cat_counts = sample_results["summary"]["review_category_counts"]

        # Test with actual data
        render_score_distribution_chart(ranked)
        render_category_donut_chart(cat_counts)
        render_component_breakdown_chart(ranked)

        # Test with empty data
        render_score_distribution_chart([])
        render_category_donut_chart({})
        render_component_breakdown_chart([])


if __name__ == "__main__":
    unittest.main()
