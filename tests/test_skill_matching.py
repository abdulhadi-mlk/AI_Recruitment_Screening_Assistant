import unittest

from src.preprocessing import preprocess_text
from src.scoring import analyze_resume
from src.skill_ontology import flattened_skill_map, full_skill_weights_map


class SkillMatchingTests(unittest.TestCase):
    def test_required_preferred_and_tools_are_matched_separately(self) -> None:
        job_description = """Required Skills — 10
Python
SQL
NumPy
Pandas
Scikit-learn
Machine Learning
Statistics
TensorFlow
PyTorch
Data Preprocessing

Preferred Skills — 4
Natural Language Processing (NLP)
Computer Vision
Deep Learning
Generative AI

Tools
Git
VS Code"""
        resume = "Python developer with machine learning, TensorFlow, SQL, Pandas, NumPy, scikit learn, NLP and Git."

        result = analyze_resume(
            job_description,
            resume,
            flattened_skill_map,
            preprocess_text,
            full_skill_weights_map,
        )

        self.assertEqual(len(result["required_skills"]), 10)
        self.assertEqual(len(result["preferred_skills"]), 4)
        self.assertEqual(result["tools"], ["git", "vs code"])
        self.assertIn("scikit-learn", result["matched_required_skills"])
        self.assertIn("natural language processing", result["matched_preferred_skills"])
        self.assertIn("pytorch", result["missing_required_skills"])
        self.assertNotIn("pytorch", result["matched_required_skills"])
        self.assertEqual(result["matched_tools"], ["git"])
        self.assertEqual(result["missing_tools"], ["vs code"])
        self.assertEqual(result["required_skill_match_score"], 70.0)


if __name__ == "__main__":
    unittest.main()
