#!/usr/bin/env python3
"""
generer_v9_docx.py
Generates EndeligForskningsrapport_G15_v9.docx from rapport_full.md.
Adds title page, TOC field, and converts all markdown content to Word formatting.
"""

import re
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

REPORT_DIR = Path(__file__).parent
ANALYSE_RESULTATER = REPORT_DIR.parent / "006 analyse" / "resultater"
MD_FILE = REPORT_DIR / "rapport_full.md"
OUT_FILE = REPORT_DIR / "EndeligForskningsrapport_G15_v9.docx"

TITLE = "Maskinlæringsbasert prognose av norsk lakseksportpris\n4–12 uker frem i tid"
AUTHORS = [
    "Alexander Francke Lindløkken",
    "Joakim Bekkevik Gåseland",
    "Carl-Henrik Solli Lilleng",
    "Morten Røgeberg",
]
INSTITUTION = "Høgskolen i Molde"
STUDY_PROGRAM = "Master i logistikk"
COURSE = "LOG650 – Logistikk og kunstig intelligens"
DATE = "Mai 2026"

# Map markdown image alt-text paths to local files
IMAGE_MAP = {
    "rapport_modellsammenligning.png": REPORT_DIR / "rapport_modellsammenligning.png",
    "ml_ensemble_prediksjon.png": REPORT_DIR / "ml_ensemble_prediksjon.png",
    "usikkerhet_kalibrering.png": REPORT_DIR / "rapport_ci_kalibrering.png",
    "sarima_residualer.png": REPORT_DIR / "sarima_residualer.png",
    "ml_avansert_bias_korr.png": REPORT_DIR / "rapport_ensemble_bias.png",
    "ml_avansert_shap_h4.png": REPORT_DIR / "ml_avansert_shap_h4.png",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_page_break(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    run.add_break(docx_break_type())
    return p


def docx_break_type():
    from docx.oxml import OxmlElement
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    return br


def insert_page_break(doc):
    para = doc.add_paragraph()
    run = para.add_run()
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run._r.append(br)


def add_toc_field(doc):
    """Insert a Word TOC field that auto-updates when opened in Word."""
    para = doc.add_paragraph()
    para.style = doc.styles["Normal"]
    run = para.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = ' TOC \\o "1-3" \\h \\z \\u '
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "[Oppdater innholdsfortegnelsen ved å høyreklikke og velge «Oppdater felt»]"
    fldChar3 = OxmlElement("w:fldChar")
    fldChar3.set(qn("w:fldCharType"), "end")
    run._r.extend([fldChar1, instrText, fldChar2, placeholder, fldChar3])
    return para


def set_paragraph_font(para, size_pt=12, bold=False, italic=False, color=None):
    for run in para.runs:
        run.font.size = Pt(size_pt)
        run.font.bold = bold
        run.font.italic = italic
        if color:
            run.font.color.rgb = RGBColor(*color)


def apply_inline_formatting(para, text):
    """Parse **bold**, *italic*, `code` and plain text, add runs."""
    pattern = re.compile(r"(\*\*.*?\*\*|\*.*?\*|`.*?`)")
    parts = pattern.split(text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = para.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*") and part.endswith("*"):
            run = para.add_run(part[1:-1])
            run.italic = True
        elif part.startswith("`") and part.endswith("`"):
            run = para.add_run(part[1:-1])
            run.font.name = "Courier New"
            run.font.size = Pt(10)
        else:
            para.add_run(part)


# ---------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------

def build_title_page(doc):
    doc.add_paragraph()
    doc.add_paragraph()
    doc.add_paragraph()

    # Institution
    p = doc.add_paragraph(INSTITUTION)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        run.font.size = Pt(14)
        run.font.bold = True

    doc.add_paragraph()

    # Title
    for line in TITLE.split("\n"):
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.size = Pt(20)
            run.font.bold = True

    doc.add_paragraph()
    doc.add_paragraph()

    # Authors (all equal, no leader role)
    for author in AUTHORS:
        p = doc.add_paragraph(author)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.size = Pt(13)

    doc.add_paragraph()

    # Study program and course
    for line in [STUDY_PROGRAM, COURSE]:
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.size = Pt(12)

    doc.add_paragraph()

    # Date
    p = doc.add_paragraph(DATE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        run.font.size = Pt(12)
        run.font.italic = True

    insert_page_break(doc)


# ---------------------------------------------------------------------------
# TOC page
# ---------------------------------------------------------------------------

def build_toc_page(doc):
    p = doc.add_heading("Innholdsfortegnelse", level=1)
    p.clear()
    run = p.add_run("Innholdsfortegnelse")
    run.font.size = Pt(16)
    run.font.bold = True
    doc.add_paragraph()
    add_toc_field(doc)
    insert_page_break(doc)


# ---------------------------------------------------------------------------
# Markdown table parser
# ---------------------------------------------------------------------------

def parse_md_table(lines):
    """Parse markdown table lines into list-of-lists (rows)."""
    rows = []
    for line in lines:
        if re.match(r"^\s*\|[-: |]+\|\s*$", line):
            continue  # separator row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def add_md_table(doc, table_lines):
    rows = parse_md_table(table_lines)
    if not rows:
        return
    col_count = max(len(r) for r in rows)
    # Pad short rows
    rows = [r + [""] * (col_count - len(r)) for r in rows]

    table = doc.add_table(rows=len(rows), cols=col_count)
    table.style = "Table Grid"

    for i, row_data in enumerate(rows):
        row = table.rows[i]
        for j, cell_text in enumerate(row_data):
            cell = row.cells[j]
            clean = re.sub(r"\*\*(.*?)\*\*", r"\1", cell_text)
            clean = re.sub(r"\*(.*?)\*", r"\1", clean)
            clean = re.sub(r"`(.*?)`", r"\1", clean)
            cell.text = clean
            if i == 0:
                for run in cell.paragraphs[0].runs:
                    run.font.bold = True
            for para in cell.paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_paragraph()


# ---------------------------------------------------------------------------
# Image insertion
# ---------------------------------------------------------------------------

def resolve_image(path_in_md):
    """Try to find the image file from the markdown path."""
    filename = Path(path_in_md).name
    if filename in IMAGE_MAP and IMAGE_MAP[filename].exists():
        return IMAGE_MAP[filename]
    # Try direct in report dir
    candidate = REPORT_DIR / filename
    if candidate.exists():
        return candidate
    # Try in analyse resultater
    candidate2 = ANALYSE_RESULTATER / filename
    if candidate2.exists():
        return candidate2
    return None


def add_image_para(doc, alt_text, img_path):
    """Add image with caption below."""
    img_file = resolve_image(img_path)
    if img_file and img_file.exists():
        try:
            para = doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run()
            run.add_picture(str(img_file), width=Inches(5.5))
        except Exception as e:
            print(f"  [WARN] Could not insert image {img_file}: {e}")
            p = doc.add_paragraph(f"[Figur: {alt_text}]")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p = doc.add_paragraph(f"[Figur: {alt_text}]")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.italic = True
        print(f"  [WARN] Image not found: {img_path}")

    # Caption
    caption = doc.add_paragraph(alt_text)
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption.runs:
        run.italic = True
        run.font.size = Pt(10)
    doc.add_paragraph()


# ---------------------------------------------------------------------------
# Block quote / decision guide
# ---------------------------------------------------------------------------

def add_blockquote(doc, lines):
    """Render a blockquote block (the decision guide)."""
    for line in lines:
        stripped = line.lstrip("> ").strip()
        if not stripped:
            continue
        if stripped.startswith("###") or stripped.startswith("**"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(1)
            apply_inline_formatting(p, stripped.lstrip("#").strip())
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(11)
        elif "|" in stripped and not stripped.startswith("| :---"):
            # It's a table row inside a blockquote
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            p = doc.add_paragraph(" | ".join(cells))
            p.paragraph_format.left_indent = Cm(1)
            for run in p.runs:
                run.font.size = Pt(10)
        else:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(1)
            apply_inline_formatting(p, stripped)
            for run in p.runs:
                run.font.size = Pt(11)


# ---------------------------------------------------------------------------
# Main markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(doc, md_text):
    lines = md_text.split("\n")
    i = 0
    in_table = False
    table_lines = []
    in_blockquote = False
    blockquote_lines = []
    in_code_block = False

    while i < len(lines):
        line = lines[i]

        # --- Code block ---
        if line.strip().startswith("```"):
            if in_code_block:
                in_code_block = False
                i += 1
                continue
            else:
                in_code_block = True
                i += 1
                continue

        if in_code_block:
            p = doc.add_paragraph(line)
            for run in p.runs:
                run.font.name = "Courier New"
                run.font.size = Pt(9)
            i += 1
            continue

        # --- Block quote ---
        if line.startswith(">"):
            blockquote_lines.append(line)
            in_blockquote = True
            i += 1
            continue
        elif in_blockquote:
            add_blockquote(doc, blockquote_lines)
            blockquote_lines = []
            in_blockquote = False
            # don't increment — re-process current line

        # --- Table ---
        if line.startswith("|"):
            table_lines.append(line)
            in_table = True
            i += 1
            continue
        elif in_table:
            add_md_table(doc, table_lines)
            table_lines = []
            in_table = False
            # re-process current line

        # --- Heading ---
        m = re.match(r"^(#{1,4})\s+(.+)$", line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            doc.add_heading(text, level=level)
            i += 1
            continue

        # --- Image ---
        m = re.match(r"^!\[(.+?)\]\((.+?)\)\s*$", line)
        if m:
            add_image_para(doc, m.group(1), m.group(2))
            i += 1
            continue

        # --- Horizontal rule ---
        if re.match(r"^---+$", line.strip()):
            i += 1
            continue

        # --- Numbered list ---
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            p = doc.add_paragraph(style="List Number")
            apply_inline_formatting(p, m.group(2))
            i += 1
            continue

        # --- Bullet list ---
        m = re.match(r"^[-*]\s+(.+)$", line)
        if m:
            p = doc.add_paragraph(style="List Bullet")
            apply_inline_formatting(p, m.group(1))
            i += 1
            continue

        # --- Blank line ---
        if not line.strip():
            i += 1
            continue

        # --- Normal paragraph ---
        p = doc.add_paragraph()
        apply_inline_formatting(p, line.strip())
        i += 1

    # Flush remaining
    if in_table:
        add_md_table(doc, table_lines)
    if in_blockquote:
        add_blockquote(doc, blockquote_lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Reading: {MD_FILE}")
    md_text = MD_FILE.read_text(encoding="utf-8")

    print("Building document...")
    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.5)

    # Default body font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(12)

    # Title page
    build_title_page(doc)

    # TOC page
    build_toc_page(doc)

    # Main content
    render_markdown(doc, md_text)

    print(f"Saving: {OUT_FILE}")
    doc.save(str(OUT_FILE))
    print("Done.")


if __name__ == "__main__":
    main()
