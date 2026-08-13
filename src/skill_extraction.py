import re
from typing import Callable, Dict, List, Optional

from src.preprocessing import preprocess_text
from src.skill_ontology import flattened_skill_map


def extract_skills(text: Optional[str], skill_map: Optional[Dict[str, str]] = None, preprocessor_func: Optional[Callable[[Optional[str]], str]] = None) -> List[str]:
    """Extract standardized skills from text using the existing ontology-based matching logic."""
    if skill_map is None:
        skill_map = flattened_skill_map
    if preprocessor_func is None:
        preprocessor_func = preprocess_text

    if not isinstance(text, str) or not text.strip():
        return []

    preprocessed_text = preprocessor_func(text)
    found_skills = set()

    for synonym_key in sorted(skill_map.keys(), key=len, reverse=True):
        pattern = rf"\b{re.escape(synonym_key)}\b"
        if re.search(pattern, preprocessed_text):
            found_skills.add(skill_map[synonym_key])

    return sorted(list(found_skills))


def get_jd_skills(job_description_text: Optional[str], skill_map: Optional[Dict[str, str]] = None, preprocessor_func: Optional[Callable[[Optional[str]], str]] = None) -> tuple[list[str], list[str]]:
    """Extract required and preferred skills from a job description using the existing notebook logic."""
    if skill_map is None:
        skill_map = flattened_skill_map
    if preprocessor_func is None:
        preprocessor_func = preprocess_text

    if not isinstance(job_description_text, str):
        return [], []

    required_skills_raw = ""
    preferred_skills_raw = ""

    required_match = re.search(
        r"REQUIRED SKILLS:(.*?)(?:PREFERRED SKILLS:|Education:|Experience:|Projects/Technical Requirements:|$)",
        job_description_text,
        re.DOTALL,
    )
    if required_match:
        required_skills_raw = required_match.group(1).strip()

    preferred_match = re.search(
        r"PREFERRED SKILLS:(.*?)(?:Education:|Experience:|Projects/Technical Requirements:|$)",
        job_description_text,
        re.DOTALL,
    )
    if preferred_match:
        preferred_skills_raw = preferred_match.group(1).strip()

    required_skills_jd = set(extract_skills(required_skills_raw, skill_map, preprocessor_func))
    preferred_skills_jd = set(extract_skills(preferred_skills_raw, skill_map, preprocessor_func))
    preferred_skills_jd = preferred_skills_jd - required_skills_jd

    return sorted(list(required_skills_jd)), sorted(list(preferred_skills_jd))
