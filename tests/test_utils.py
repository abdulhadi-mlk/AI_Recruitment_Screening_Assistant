import unittest

from src.utils import build_explanation


class BuildExplanationTests(unittest.TestCase):
    def test_low_score_uses_limited_alignment(self) -> None:
        analysis = {
            "matched_required_skills": ["git"],
            "missing_required_skills": ["pytorch"],
            "matched_preferred_skills": ["natural language processing"],
            "missing_preferred_skills": ["aws", "deep learning", "docker", "matplotlib", "seaborn"],
            "combined_score": 28,
        }

        explanation = build_explanation(analysis)

        self.assertIn("limited alignment", explanation.lower())
        self.assertNotIn("strong alignment", explanation.lower())

    def test_high_score_uses_strong_alignment(self) -> None:
        analysis = {
            "matched_required_skills": ["python", "sql", "git"],
            "missing_required_skills": [],
            "matched_preferred_skills": ["nlp", "docker"],
            "missing_preferred_skills": ["aws"],
            "combined_score": 85,
        }

        explanation = build_explanation(analysis)

        self.assertIn("strong alignment", explanation.lower())


if __name__ == "__main__":
    unittest.main()
