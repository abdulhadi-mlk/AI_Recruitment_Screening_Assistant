from __future__ import annotations

from typing import Optional

import fitz


def extract_text_from_pdf(file_obj) -> str:
    """Extract text from an uploaded PDF file object using PyMuPDF."""
    if file_obj is None:
        raise ValueError("No file uploaded.")

    try:
        doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Unable to read PDF file: {exc}") from exc

    text_chunks = []
    for page in doc:
        text_chunks.append(page.get_text())
    doc.close()

    extracted_text = "\n".join(text_chunks).strip()
    if not extracted_text:
        raise ValueError("The uploaded PDF did not contain any readable text.")
    return extracted_text
