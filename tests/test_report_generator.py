"""Unit and integration tests for the Report Generation Module (src/report_generator.py).

Covers CSV export, HTML generation, summary metrics, review tier breakdowns,
skill list rendering, empty candidate handling, input validation, Unicode support,
HTML escaping security, custom output directories, and full integration with all 8
synthetic resumes from Phase 10.
"""

import csv
from pathlib import Path
import tempfile
import unittest

from src.job_parser import get_standardized_job_profile
from src.matcher import match_candidates
from src.ranking import get_ranking_summary, rank_candidates
from src.report_generator import (
    CSV_HEADERS,
    generate_csv_report,
    generate_html_report,
    generate_reports,
    validate_report_candidates,
)
from src.resume_parser import extract_resume_text
from src.score_calculator import score_candidates


class TestReportGenerator(unittest.TestCase):
    """Test suite for CSV and HTML report generation."""

    def setUp(self):
        """Prepare mock candidate fixtures for isolated testing."""
        self.mock_candidates = [
            {
                "candidate_id": "CAND-001",
                "rank": 1,
                "display_order": 1,
                "overall_score": 88.83,
                "review_category": "Priority Review",
                "required_skill_match_percentage": 100.0,
                "preferred_skill_match_percentage": 87.5,
                "overall_skill_coverage_percentage": 91.67,
                "experience_score": 100.0,
                "education_score": 100.0,
                "text_similarity_percentage": 53.56,
                "matched_required_skills": ["Python", "SQL", "Pandas", "Git"],
                "missing_required_skills": [],
                "matched_preferred_skills": ["FastAPI", "Flask", "PostgreSQL", "REST APIs"],
                "missing_preferred_skills": ["Docker"],
                "weighted_contributions": {
                    "required_skills": 35.0,
                    "preferred_skills": 13.12,
                    "experience": 20.0,
                    "text_similarity": 10.71,
                    "education": 10.0,
                },
                "score_explanation": "Strong candidate meeting all core criteria.",
            },
            {
                "candidate_id": "CAND-002",
                "rank": 2,
                "display_order": 2,
                "overall_score": 79.0,
                "review_category": "Review",
                "required_skill_match_percentage": 100.0,
                "preferred_skill_match_percentage": 50.0,
                "overall_skill_coverage_percentage": 66.67,
                "experience_score": 85.0,
                "education_score": 100.0,
                "text_similarity_percentage": 47.5,
                "matched_required_skills": ["Python", "SQL", "Pandas", "Git"],
                "missing_required_skills": [],
                "matched_preferred_skills": ["Flask", "REST APIs"],
                "missing_preferred_skills": ["FastAPI", "Docker", "PostgreSQL"],
                "weighted_contributions": {
                    "required_skills": 35.0,
                    "preferred_skills": 7.5,
                    "experience": 17.0,
                    "text_similarity": 9.5,
                    "education": 10.0,
                },
                "score_explanation": "Solid backend profile.",
            },
            {
                "candidate_id": "CAND-003",
                "rank": 3,
                "display_order": 3,
                "overall_score": 52.0,
                "review_category": "Additional Review",
                "required_skill_match_percentage": 50.0,
                "preferred_skill_match_percentage": 25.0,
                "overall_skill_coverage_percentage": 33.33,
                "experience_score": 60.0,
                "education_score": 80.0,
                "text_similarity_percentage": 35.0,
                "matched_required_skills": ["Python", "Git"],
                "missing_required_skills": ["SQL", "Pandas"],
                "matched_preferred_skills": ["NumPy"],
                "missing_preferred_skills": ["FastAPI", "Flask", "Docker"],
                "weighted_contributions": {
                    "required_skills": 17.5,
                    "preferred_skills": 3.75,
                    "experience": 12.0,
                    "text_similarity": 7.0,
                    "education": 8.0,
                },
                "score_explanation": "Emerging candidate.",
            },
        ]

    # -------------------------------------------------------------------------
    # 1. CSV Report Generation Tests
    # -------------------------------------------------------------------------

    def test_csv_file_creation(self):
        """Verify that generate_csv_report creates a valid CSV file on disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = generate_csv_report(self.mock_candidates, output_dir=tmp_dir)
            self.assertTrue(csv_path.exists())
            self.assertTrue(csv_path.is_file())
            self.assertGreater(csv_path.stat().st_size, 0)

    def test_csv_headers_and_data_correctness(self):
        """Verify headers and exact row content in the generated CSV file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = generate_csv_report(self.mock_candidates, output_dir=tmp_dir)

            with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 4)  # 1 header + 3 candidate rows
            self.assertEqual(rows[0], CSV_HEADERS)

            # Check candidate 1 row
            cand1_row = rows[1]
            self.assertEqual(cand1_row[0], "CAND-001")
            self.assertEqual(cand1_row[1], "1")
            self.assertEqual(cand1_row[3], "88.83")
            self.assertEqual(cand1_row[4], "Priority Review")
            self.assertEqual(cand1_row[5], "100.00")
            self.assertIn("Python; SQL; Pandas; Git", cand1_row[11])
            self.assertEqual(cand1_row[12], "None")  # No missing required skills

    def test_candidate_order_preservation(self):
        """Verify that ranking order is strictly preserved in CSV output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = generate_csv_report(self.mock_candidates, output_dir=tmp_dir)

            with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                rows = list(reader)[1:]  # skip header

            cids = [r[0] for r in rows]
            self.assertEqual(cids, ["CAND-001", "CAND-002", "CAND-003"])

    # -------------------------------------------------------------------------
    # 2. HTML Report Generation Tests
    # -------------------------------------------------------------------------

    def test_html_file_creation(self):
        """Verify that generate_html_report creates a valid HTML file on disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = generate_html_report(
                self.mock_candidates,
                output_dir=tmp_dir,
                job_title="Junior Python Developer",
            )
            self.assertTrue(html_path.exists())
            self.assertTrue(html_path.is_file())
            self.assertGreater(html_path.stat().st_size, 0)

    def test_html_report_content(self):
        """Verify critical sections, job title, and metadata in HTML output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = generate_html_report(
                self.mock_candidates,
                output_dir=tmp_dir,
                job_title="Junior Python Developer",
            )
            content = html_path.read_text(encoding="utf-8")

            # Check structural headers
            self.assertIn("Automated Resume Screening Report", content)
            self.assertIn("Junior Python Developer", content)
            self.assertIn("Cohort Screening Summary", content)
            self.assertIn("Ranked Candidate Overview", content)
            self.assertIn("Detailed Candidate Profiles & Skill Audits", content)
            self.assertIn("Responsible AI & Human-in-the-Loop Governance Notice", content)

            # Check candidates in table
            self.assertIn("CAND-001", content)
            self.assertIn("CAND-002", content)
            self.assertIn("CAND-003", content)

    def test_summary_metrics_correctness(self):
        """Verify summary values (total, mean, median, high, low) in HTML output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = generate_html_report(self.mock_candidates, output_dir=tmp_dir)
            content = html_path.read_text(encoding="utf-8")

            self.assertIn('<div class="metric-value">3</div>', content)  # Total candidates
            self.assertIn("88.83", content)  # Highest score
            self.assertIn("52.00", content)  # Lowest score

    def test_review_category_counts(self):
        """Verify category breakdown badges and percentages in HTML."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = generate_html_report(self.mock_candidates, output_dir=tmp_dir)
            content = html_path.read_text(encoding="utf-8")

            self.assertIn("Priority Review", content)
            self.assertIn("Review", content)
            self.assertIn("Additional Review", content)

    def test_score_breakdown_rendering(self):
        """Verify that score contributions are explicitly rendered."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = generate_html_report(self.mock_candidates, output_dir=tmp_dir)
            content = html_path.read_text(encoding="utf-8")

            self.assertIn("+35.00 pts", content)
            self.assertIn("+13.12 pts", content)
            self.assertIn("+20.00 pts", content)

    def test_skill_list_rendering(self):
        """Verify matched and missing skill badges are properly rendered."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = generate_html_report(self.mock_candidates, output_dir=tmp_dir)
            content = html_path.read_text(encoding="utf-8")

            self.assertIn("badge-skill-match", content)
            self.assertIn("badge-skill-missing", content)
            self.assertIn("FastAPI", content)
            self.assertIn("Docker", content)

    # -------------------------------------------------------------------------
    # 3. Empty and Edge Case Handling
    # -------------------------------------------------------------------------

    def test_empty_candidate_report(self):
        """Verify graceful handling when ranking cohort is empty."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Empty CSV
            csv_path = generate_csv_report([], output_dir=tmp_dir)
            with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), 1)  # Only header
            self.assertEqual(rows[0], CSV_HEADERS)

            # 2. Empty HTML
            html_path = generate_html_report([], output_dir=tmp_dir)
            content = html_path.read_text(encoding="utf-8")
            self.assertIn("No candidate records available.", content)
            self.assertIn('<div class="metric-value">0</div>', content)

    def test_invalid_input_validation(self):
        """Verify type and value error raising for malformed candidate inputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(TypeError):
                generate_csv_report("not a list", output_dir=tmp_dir)  # type: ignore

            with self.assertRaises(TypeError):
                generate_html_report([1, 2, 3], output_dir=tmp_dir)  # type: ignore

            with self.assertRaises(ValueError):
                generate_csv_report([{"rank": 1}], output_dir=tmp_dir)  # missing candidate_id

            with self.assertRaises(TypeError):
                generate_html_report([], summary="not a dict", output_dir=tmp_dir)  # type: ignore

    def test_missing_optional_fields(self):
        """Verify candidates with sparse/missing optional fields render safely."""
        sparse_candidate = [
            {
                "candidate_id": "SPARSE-001",
                "rank": 1,
                "display_order": 1,
                "overall_score": 65.0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = generate_csv_report(sparse_candidate, output_dir=tmp_dir)
            html_path = generate_html_report(sparse_candidate, output_dir=tmp_dir)

            self.assertTrue(csv_path.exists())
            self.assertTrue(html_path.exists())

            csv_text = csv_path.read_text(encoding="utf-8-sig")
            html_text = html_path.read_text(encoding="utf-8")

            self.assertIn("SPARSE-001", csv_text)
            self.assertIn("SPARSE-001", html_text)
            self.assertIn("N/A", csv_text)

    # -------------------------------------------------------------------------
    # 4. Security, Unicode, and Isolation
    # -------------------------------------------------------------------------

    def test_unicode_handling(self):
        """Verify proper handling of non-ASCII characters, accents, and symbols."""
        unicode_candidate = [
            {
                "candidate_id": "CAND-ÜNÎCÖDÉ-007",
                "rank": 1,
                "display_order": 1,
                "overall_score": 82.5,
                "review_category": "Review",
                "matched_required_skills": ["Python", "C++", "Café-Framework"],
                "score_explanation": "Candidate has résumé with São Paulo experience.",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = generate_csv_report(unicode_candidate, output_dir=tmp_dir)
            html_path = generate_html_report(unicode_candidate, output_dir=tmp_dir)

            csv_content = csv_path.read_text(encoding="utf-8-sig")
            html_content = html_path.read_text(encoding="utf-8")

            self.assertIn("CAND-ÜNÎCÖDÉ-007", csv_content)
            self.assertIn("São Paulo", csv_content)
            self.assertIn("Café-Framework", html_content)

    def test_html_escaping_security(self):
        """Verify that malicious script or HTML tags are strictly escaped."""
        xss_candidate = [
            {
                "candidate_id": "<script>alert('XSS')</script>",
                "rank": 1,
                "display_order": 1,
                "overall_score": 75.0,
                "review_category": "<b onclick=evil()>ClickMe</b>",
                "matched_required_skills": ["<img src=x onerror=alert(1)>"],
                "score_explanation": "<script>fetch('https://attacker.com')</script>",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = generate_html_report(
                xss_candidate,
                output_dir=tmp_dir,
                job_title='<script>console.log("PWN")</script>',
            )
            content = html_path.read_text(encoding="utf-8")

            # Dangerous raw tags must NOT exist in the output
            self.assertNotIn("<script>", content)
            self.assertNotIn("<img src=x", content)
            self.assertNotIn("<b onclick=evil()>", content)

            # Escaped representations must be present
            self.assertIn("&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;", content)
            self.assertIn("&lt;img src=x onerror=alert(1)&gt;", content)

    def test_custom_output_directory_and_isolation(self):
        """Verify output directory creation and isolation using temporary directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = Path(tmp_dir) / "sub1" / "sub2"
            results = generate_reports(
                self.mock_candidates,
                output_dir=nested_dir,
                csv_filename="test_report.csv",
                html_filename="test_report.html",
            )

            self.assertTrue(results["csv"].exists())
            self.assertTrue(results["html"].exists())
            self.assertEqual(results["csv"].name, "test_report.csv")
            self.assertEqual(results["html"].name, "test_report.html")

    # -------------------------------------------------------------------------
    # 5. Integration with All 8 Synthetic Resumes
    # -------------------------------------------------------------------------

    def test_integration_all_8_synthetic_candidates(self):
        """End-to-end integration test generating reports for all 8 synthetic candidates."""
        resume_dir = Path("resumes/sample_resumes")
        resume_files = sorted(resume_dir.glob("CAND-*.txt"))
        self.assertEqual(len(resume_files), 8)

        # 1. Matching
        profile = get_standardized_job_profile()
        matches = match_candidates(resume_files, profile)
        self.assertEqual(len(matches), 8)

        # 2. Scoring
        resume_texts = {f.stem: extract_resume_text(f) for f in resume_files}
        scores = score_candidates(matches, resume_texts=resume_texts)
        self.assertEqual(len(scores), 8)

        # 3. Ranking
        ranked = rank_candidates(scores)
        self.assertEqual(len(ranked), 8)

        # 4. Report Generation
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = generate_reports(
                ranked,
                output_dir=tmp_dir,
                job_title=profile.get("job_title", "Junior Python Developer"),
            )

            csv_file = results["csv"]
            html_file = results["html"]

            self.assertTrue(csv_file.exists())
            self.assertTrue(html_file.exists())

            # Verify CSV contents
            with open(csv_file, mode="r", encoding="utf-8-sig") as f:
                csv_rows = list(csv.reader(f))

            self.assertEqual(len(csv_rows), 9)  # 1 header + 8 candidate rows
            # Verify top candidate in CSV
            self.assertEqual(csv_rows[1][0], "CAND-001")
            self.assertEqual(csv_rows[1][1], "1")
            self.assertEqual(csv_rows[1][4], "Priority Review")

            # Verify bottom candidate in CSV
            self.assertEqual(csv_rows[8][0], "CAND-008")
            self.assertEqual(csv_rows[8][1], "8")
            self.assertEqual(csv_rows[8][4], "Additional Review")

            # Verify HTML contents
            html_text = html_file.read_text(encoding="utf-8")
            for i in range(1, 9):
                cid = f"CAND-00{i}"
                self.assertIn(cid, html_text)

            self.assertIn("Junior Python Developer", html_text)
            self.assertIn("88.83", html_text)
            self.assertIn("24.84", html_text)


if __name__ == "__main__":
    unittest.main()
