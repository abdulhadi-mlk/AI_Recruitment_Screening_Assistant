from __future__ import annotations

from typing import Dict, List, Tuple

from src.preprocessing import preprocess_text


skill_ontology: Dict[str, Dict[str, List[str]]] = {
    "Programming": {
        "python": ["python", "py", "python programming"],
        "r": ["r", "rstudio", "r programming"],
        "java": ["java", "java programming"],
        "c++": ["c++", "cpp", "cplusplus"],
        "sql": ["sql", "postgresql", "mysql", "sqlite", "database query", "sql server", "oracle sql"],
        "javascript": ["javascript", "js", "node.js", "react", "angular", "vuejs"],
        "scala": ["scala"],
        "bash": ["bash", "shell scripting", "linux scripting"],
    },
    "Data Science": {
        "pandas": ["pandas", "pd"],
        "numpy": ["numpy", "np"],
        "scipy": ["scipy"],
        "data analysis": ["data analysis", "data analytics", "eda", "exploratory data analysis"],
        "statistics": ["statistics", "statistical analysis", "hypothesis testing"],
        "excel": ["excel", "microsoft excel", "advanced excel"],
        "tableau": ["tableau"],
        "power bi": ["power bi"],
        "data visualization": ["data visualization", "dataviz"],
    },
    "Machine Learning": {
        "machine learning": ["machine learning", "ml", "supervised learning", "unsupervised learning", "reinforcement learning"],
        "scikit-learn": ["scikit-learn", "sklearn"],
        "xgboost": ["xgboost", "xgb"],
        "lightgbm": ["lightgbm", "lgbm"],
        "model deployment": ["model deployment", "mlops"],
        "feature engineering": ["feature engineering"],
        "model evaluation": ["model evaluation", "metrics", "cross-validation"],
    },
    "Deep Learning": {
        "deep learning": ["deep learning", "dl"],
        "tensorflow": ["tensorflow", "tf"],
        "pytorch": ["pytorch", "torch"],
        "keras": ["keras"],
        "cnns": ["cnns", "convolutional neural networks", "computer vision"],
        "rnns": ["rnns", "recurrent neural networks"],
        "transformers": ["transformers", "bert", "gpt", "encoder decoder models"],
    },
    "Natural Language Processing": {
        "natural language processing": ["natural language processing", "nlp"],
        "spacy": ["spacy"],
        "nltk": ["nltk"],
        "text classification": ["text classification"],
        "sentiment analysis": ["sentiment analysis"],
        "ner": ["ner", "named entity recognition"],
    },
    "Databases": {
        "mongodb": ["mongodb", "nosql"],
        "cassandra": ["cassandra"],
        "data warehousing": ["data warehousing", "data lake"],
        "etl": ["etl", "extract transform load"],
    },
    "Visualization": {
        "matplotlib": ["matplotlib", "plt"],
        "seaborn": ["seaborn", "sns"],
        "plotly": ["plotly"],
        "bokeh": ["bokeh"],
    },
    "Cloud": {
        "aws": ["aws", "amazon web services"],
        "azure": ["azure", "microsoft azure"],
        "gcp": ["gcp", "google cloud platform"],
        "docker": ["docker"],
        "kubernetes": ["kubernetes", "k8s"],
    },
    "Tools": {
        "git": ["git", "github", "gitlab", "bitbucket", "version control"],
        "jira": ["jira"],
        "confluence": ["confluence"],
        "airflow": ["airflow"],
    },
    "Other": {
        "agile": ["agile", "scrum"],
        "problem solving": ["problem solving", "analytical thinking"],
        "communication": ["communication", "teamwork"],
        "data structures": ["data structures", "algorithms"],
        "api": ["api", "rest api"],
        "linux": ["linux", "unix"],
    },
}


def build_flattened_skill_map(ontology: Dict[str, Dict[str, List[str]]]) -> Dict[str, str]:
    flattened: Dict[str, str] = {}
    for _, skills_map in ontology.items():
        for standardized_skill_name, synonyms_list in skills_map.items():
            for synonym in synonyms_list:
                flattened[preprocess_text(synonym)] = standardized_skill_name
    return flattened


flattened_skill_map = build_flattened_skill_map(skill_ontology)


def get_skill_category_map(ontology: Dict[str, Dict[str, List[str]]]) -> Dict[str, str]:
    skill_to_category: Dict[str, str] = {}
    for category, skills_map in ontology.items():
        for standardized_skill_name in skills_map.keys():
            skill_to_category[standardized_skill_name] = category
    return skill_to_category


skill_to_category_map = get_skill_category_map(skill_ontology)


def build_skill_weights(flattened_map: Dict[str, str], default_weight: int = 5) -> Dict[str, int]:
    base_skill_weights = {
        "python": 20,
        "machine learning": 20,
        "sql": 15,
        "pandas": 10,
        "numpy": 10,
        "scikit-learn": 10,
    }

    skill_weights = {skill: weight for skill, weight in base_skill_weights.items()}
    for standardized_skill_name in set(flattened_map.values()):
        skill_weights.setdefault(standardized_skill_name, default_weight)
    return skill_weights


full_skill_weights_map = build_skill_weights(flattened_skill_map)
