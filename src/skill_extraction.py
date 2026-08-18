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


def _section_text(job_description_text: str, labels: list[str]) -> str:
    """Return text below a labelled JD section, stopping at the next known section."""
    label_pattern = "|".join(re.escape(label) for label in labels)
    all_sections = (
        r"required(?:\s+skills?)?|must[-\s]*have|core\s+skills|"
        r"preferred(?:\s+skills?)?|nice[-\s]*to[-\s]*have|"
        r"tools?(?:\s*(?:and|&)?\s*technologies)?|technologies|"
        r"education|experience|responsibilities|projects?"
    )
    section_suffix = r"(?:\s*[:—–-]\s*\d*)?"
    match = re.search(
        rf"(?im)^\s*(?:{label_pattern}){section_suffix}\s*$\s*(.*?)(?=^\s*(?:{all_sections}){section_suffix}\s*$|\Z)",
        job_description_text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def get_jd_skill_groups(job_description_text: Optional[str], skill_map: Optional[Dict[str, str]] = None, preprocessor_func: Optional[Callable[[Optional[str]], str]] = None) -> tuple[list[str], list[str], list[str]]:
    """Extract separate required, preferred, and tool skill groups from a JD.

    Supports case-insensitive section headings and falls back to all detected JD
    skills when a free-form description has no labelled sections.
    """
    if skill_map is None:
        skill_map = flattened_skill_map
    if preprocessor_func is None:
        preprocessor_func = preprocess_text

    if not isinstance(job_description_text, str):
        return [], [], []

    required_raw = _section_text(job_description_text, ["required skills", "required", "must have", "core skills"])
    preferred_raw = _section_text(job_description_text, ["preferred skills", "preferred", "nice to have"])
    tools_raw = _section_text(job_description_text, ["tools", "tools and technologies", "tools & technologies", "technologies"])

    required_skills = set(extract_skills(required_raw, skill_map, preprocessor_func))
    preferred_skills = set(extract_skills(preferred_raw, skill_map, preprocessor_func))
    tools = set(extract_skills(tools_raw, skill_map, preprocessor_func))

    # Free-form descriptions still produce useful groups. Known tool and preferred
    # canonical skills are not silently counted as required in this fallback.
    if not (required_raw or preferred_raw or tools_raw):
        all_skills = set(extract_skills(job_description_text, skill_map, preprocessor_func))
        preferred_names = {"natural language processing", "computer vision", "deep learning", "generative ai"}
        tool_names = {"git", "vs code"}
        preferred_skills = all_skills & preferred_names
        tools = all_skills & tool_names
        required_skills = all_skills - preferred_skills - tools

    preferred_skills -= required_skills
    tools -= required_skills | preferred_skills
    return sorted(required_skills), sorted(preferred_skills), sorted(tools)


def get_jd_skills(job_description_text: Optional[str], skill_map: Optional[Dict[str, str]] = None, preprocessor_func: Optional[Callable[[Optional[str]], str]] = None) -> tuple[list[str], list[str]]:
    """Backward-compatible required/preferred JD skill API."""
    required_skills, preferred_skills, _ = get_jd_skill_groups(job_description_text, skill_map, preprocessor_func)
    return required_skills, preferred_skills
