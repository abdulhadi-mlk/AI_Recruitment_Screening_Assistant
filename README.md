# Resume Analyzer

A Streamlit-based resume analyzer that reuses the existing notebook logic for preprocessing, ontology-based skill extraction, required/preferred skill analysis, weighted scoring, TF-IDF similarity, and combined scoring.

## Features

- Paste a job description
- Upload a PDF resume
- Extract text from the PDF
- Analyze the resume using the existing skill ontology and scoring approach
- Display match scores, matched/missing skills, and explainable insights

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```
