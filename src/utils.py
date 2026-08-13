from __future__ import annotations

from typing import Dict, List


def _score_band(score: float) -> str:
    if score >= 80:
        return "strong alignment"
    if score >= 60:
        return "good alignment"
    if score >= 40:
        return "moderate alignment"
    return "limited alignment"


def build_explanation(analysis: Dict[str, object]) -> str:
    matched_required = analysis.get("matched_required_skills", []) or []
    missing_required = analysis.get("missing_required_skills", []) or []
    matched_preferred = analysis.get("matched_preferred_skills", []) or []
    missing_preferred = analysis.get("missing_preferred_skills", []) or []
    combined_score = float(analysis.get("combined_score", 0) or 0)

    matched_required_text = ", ".join(matched_required) or "none"
    missing_required_text = ", ".join(missing_required) or "none"
    matched_preferred_text = ", ".join(matched_preferred) or "none"
    missing_preferred_text = ", ".join(missing_preferred) or "none"
    alignment_label = _score_band(combined_score)

    return (
        f"This resume shows {alignment_label} with the role when the required and preferred skills are compared. "
        f"Matched required skills include {matched_required_text}. Missing required skills include {missing_required_text}. "
        f"Matched preferred skills include {matched_preferred_text}. Missing preferred skills include {missing_preferred_text}."
    )


def group_skills_by_category(skills: List[str], skill_to_category_map: Dict[str, str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for skill in sorted(skills):
        category = skill_to_category_map.get(skill, "Other")
        grouped.setdefault(category, []).append(skill)
    return grouped
