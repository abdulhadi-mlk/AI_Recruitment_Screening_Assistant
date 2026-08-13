from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text
from src.skill_ontology import full_skill_weights_map
from src.skill_extraction import extract_skills


def calculate_baseline_match(candidate_skills: List[str], required_jd_skills: List[str], preferred_jd_skills: List[str], weight_required: float = 1.0, weight_preferred: float = 0.5) -> Dict[str, object]:
    candidate_skills_set = set(candidate_skills)
    required_jd_skills_set = set(required_jd_skills)
    preferred_jd_skills_set = set(preferred_jd_skills)

    matched_required = sorted(list(candidate_skills_set.intersection(required_jd_skills_set)))
    missing_required = sorted(list(required_jd_skills_set - candidate_skills_set))
    matched_preferred = sorted(list(candidate_skills_set.intersection(preferred_jd_skills_set)))
    missing_preferred = sorted(list(preferred_jd_skills_set - candidate_skills_set))

    total_required_count = len(required_jd_skills_set)
    total_preferred_count = len(preferred_jd_skills_set)

    score_matched_required = len(matched_required) * weight_required
    score_matched_preferred = len(matched_preferred) * weight_preferred
    total_possible_score = (total_required_count * weight_required) + (total_preferred_count * weight_preferred)

    baseline_score = ((score_matched_required + score_matched_preferred) / total_possible_score * 100) if total_possible_score > 0 else 0.0
    return {
        "matched_required_skills": matched_required,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred,
        "missing_preferred_skills": missing_preferred,
        "baseline_match_percentage": round(baseline_score, 2),
    }


def calculate_weighted_skill_score(candidate_skills: List[str], required_jd_skills: List[str], preferred_jd_skills: List[str], skill_weights_map: Optional[Dict[str, int]] = None, preferred_skill_multiplier: float = 0.5) -> float:
    if skill_weights_map is None:
        skill_weights_map = full_skill_weights_map

    candidate_skills_set = set(candidate_skills)
    required_jd_skills_set = set(required_jd_skills)
    preferred_jd_skills_set = set(preferred_jd_skills)

    matched_required_score = 0
    matched_preferred_score = 0
    max_possible_score = 0

    for skill in required_jd_skills_set:
        weight = skill_weights_map.get(skill, 0)
        max_possible_score += weight
        if skill in candidate_skills_set:
            matched_required_score += weight

    for skill in preferred_jd_skills_set:
        weight = skill_weights_map.get(skill, 0)
        max_possible_score += weight * preferred_skill_multiplier
        if skill in candidate_skills_set:
            matched_preferred_score += weight * preferred_skill_multiplier

    total_matched_score = matched_required_score + matched_preferred_score
    if max_possible_score > 0:
        return round((total_matched_score / max_possible_score) * 100, 2)
    return 0.0


def calculate_tfidf_similarity(job_description: str, resume_texts: List[str]) -> List[float]:
    preprocessed_jd = preprocess_text(job_description)
    preprocessed_resumes = [preprocess_text(text) for text in resume_texts]
    corpus = [preprocessed_jd] + preprocessed_resumes

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    vectorizer.fit(corpus)
    jd_vector = vectorizer.transform([preprocessed_jd])
    resume_vectors = vectorizer.transform(preprocessed_resumes)
    cosine_scores = cosine_similarity(jd_vector, resume_vectors).flatten()
    return [round(float(score) * 100, 2) for score in cosine_scores]


def calculate_combined_score(weighted_skill_score: float, tfidf_similarity_score: float, weight_skill_score: float = 0.70, weight_tfidf_score: float = 0.30) -> float:
    return round((weighted_skill_score * weight_skill_score) + (tfidf_similarity_score * weight_tfidf_score), 2)


def analyze_resume(job_description: str, resume_text: str, skill_map=None, preprocessor_func=None, skill_weights_map=None) -> Dict[str, object]:
    required_skills_jd, preferred_skills_jd = get_jd_skills_from_text(job_description, skill_map, preprocessor_func)
    extracted_skills = extract_skills(resume_text, skill_map=skill_map, preprocessor_func=preprocessor_func)

    baseline_results = calculate_baseline_match(extracted_skills, required_skills_jd, preferred_skills_jd)
    weighted_score = calculate_weighted_skill_score(extracted_skills, required_skills_jd, preferred_skills_jd, skill_weights_map=skill_weights_map)
    tfidf_scores = calculate_tfidf_similarity(job_description, [resume_text])
    tfidf_similarity_score = tfidf_scores[0] if tfidf_scores else 0.0
    combined_score = calculate_combined_score(weighted_score, tfidf_similarity_score)

    return {
        "required_skills": required_skills_jd,
        "preferred_skills": preferred_skills_jd,
        "extracted_skills": extracted_skills,
        "matched_required_skills": baseline_results["matched_required_skills"],
        "missing_required_skills": baseline_results["missing_required_skills"],
        "matched_preferred_skills": baseline_results["matched_preferred_skills"],
        "missing_preferred_skills": baseline_results["missing_preferred_skills"],
        "baseline_match_percentage": baseline_results["baseline_match_percentage"],
        "weighted_skill_score": weighted_score,
        "tfidf_similarity_score": tfidf_similarity_score,
        "combined_score": combined_score,
    }


def get_jd_skills_from_text(job_description: str, skill_map=None, preprocessor_func=None) -> tuple[list[str], list[str]]:
    from src.skill_extraction import get_jd_skills

    return get_jd_skills(job_description, skill_map=skill_map, preprocessor_func=preprocessor_func)
