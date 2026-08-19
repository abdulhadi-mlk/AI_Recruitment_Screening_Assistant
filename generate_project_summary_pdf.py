from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
import os

project_root = os.path.dirname(__file__)
readme_path = os.path.join(project_root, 'README.md')
output_pdf = os.path.join(project_root, 'Project_Summary_AI_Recruitment_Screening_Assistant.pdf')

# Read README.md
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()
else:
    readme = ''

# Build a simple summary from README and file list
file_list = sorted([f for f in os.listdir(project_root) if not f.startswith('.')])

# Extract a brief title (first markdown header) and a short description
title = 'AI Recruitment Screening Assistant'
short_desc = ''
for line in readme.splitlines():
    line = line.strip()
    if line.startswith('#') and len(line) > 1:
        title = line.lstrip('#').strip()
        continue
    if line and not line.startswith('#'):
        short_desc = short_desc + line + ' '\
            if len(short_desc) < 800 else short_desc
    if len(short_desc) >= 800:
        break

# Create PDF
styles = getSampleStyleSheet()
if 'Heading1' not in styles.byName:
    styles.add(ParagraphStyle(name='Heading1', fontSize=16, leading=20, spaceAfter=8))
if 'NormalIndented' not in styles.byName:
    styles.add(ParagraphStyle(name='NormalIndented', leftIndent=6, fontSize=10, leading=14))

doc = SimpleDocTemplate(output_pdf, pagesize=A4,
                        rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
story = []

story.append(Paragraph(title, styles['Heading1']))
story.append(Spacer(1, 4))

if short_desc:
    story.append(Paragraph(short_desc, styles['Normal']))
    story.append(Spacer(1, 6))

# Add key sections from README: Features, Notes, Gemini configuration

# Helper to extract sections
def extract_section(md_text, section_name):
    lines = md_text.splitlines()
    capture = False
    collected = []
    for line in lines:
        if line.strip().lower().startswith('##') and section_name.lower() in line.lower():
            capture = True
            continue
        if capture and line.strip().startswith('##'):
            break
        if capture:
            collected.append(line)
    return '\n'.join(collected).strip()

features = extract_section(readme, 'Features')
notes = extract_section(readme, 'Notes')
gemini = extract_section(readme, 'Gemini configuration')

if features:
    story.append(Paragraph('Features', styles['Heading1']))
    for line in features.splitlines():
        if line.strip().startswith('-'):
            story.append(Paragraph(line.strip().lstrip('-').strip(), styles['NormalIndented']))
    story.append(Spacer(1, 6))

if gemini:
    story.append(Paragraph('Gemini / AI Integration', styles['Heading1']))
    for line in gemini.splitlines():
        if line.strip():
            story.append(Paragraph(line.strip(), styles['NormalIndented']))
    story.append(Spacer(1, 6))

if notes:
    story.append(Paragraph('Notes', styles['Heading1']))
    for line in notes.splitlines():
        if line.strip():
            story.append(Paragraph(line.strip(), styles['NormalIndented']))
    story.append(Spacer(1, 6))

# Add file list
story.append(Paragraph('Top-level files and folders', styles['Heading1']))
for f in file_list:
    story.append(Paragraph(f, styles['NormalIndented']))

# Footer / metadata
story.append(Spacer(1, 12))
story.append(Paragraph('Generated summary of the project. For full details, open the project files in the repository.', styles['Normal']))

# Build PDF
try:
    doc.build(story)
    print('PDF generated at:', output_pdf)
except Exception as e:
    print('Failed to generate PDF:', str(e))
