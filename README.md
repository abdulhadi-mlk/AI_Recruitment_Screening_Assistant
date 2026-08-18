# AI Recruitment Screening Assistant

A Streamlit-based resume analyzer that keeps the existing deterministic Week 1 scoring logic while adding a real Gemini API analysis layer for explainability, skill matching, and screening support.

## Features

- Paste a job description
- Upload one or more PDF resumes
- Extract text from PDF files
- Compare resumes against required and preferred job skills
- Display match scores, matched and missing skills, and explainable insights
- Use the Google Gemini API for structured screening analysis while preserving the existing scoring engine
- Show an explicit AI screening disclaimer and bias-awareness guidance

## Gemini configuration

1. Copy `.env.example` to `.env`.
2. Add your Gemini key:

```env
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-3.6-flash
```

Note: Google may retire older Gemini model IDs for new accounts. The app includes a fallback path, but the recommended model for newer accounts is `gemini-3.6-flash`.

3. The application uses `python-dotenv` and reads the key from the environment automatically, so the key is never hard-coded into source code.

## Windows / VS Code setup

In VS Code terminal, run:

```powershell
copy .env.example .env
```

Then edit `.env` and replace `your_actual_key_here` with your real Gemini API key.

You can also set it directly in the terminal without creating a `.env` file:

```powershell
$env:GEMINI_API_KEY = "your_actual_key_here"
```

For a persistent setup in VS Code, add the environment variable to your terminal profile or use the `.env` file in the project root.

## Installation

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Notes

- The deterministic Week 1 score remains the main assessment signal.
- The Gemini integration is used to produce a structured, job-focused explanation and guardrails around bias-sensitive screening.
- If the API key is missing or the API call fails, the app falls back to the deterministic scoring logic without exposing the API key in the UI.
- The application never evaluates protected characteristics such as age, gender, race, religion, nationality, disability, or other unrelated personal details.
