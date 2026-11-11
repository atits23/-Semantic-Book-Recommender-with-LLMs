"""Build the editable Word handbook from its readable Markdown source."""

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]


def add_text(paragraph, text):
    for i, part in enumerate(re.split(r"`([^`]+)`", text)):
        run = paragraph.add_run(part)
        if i % 2:
            run.font.name = "Consolas"
            run.font.size = Pt(9)


def main():
    document = Document()
    section = document.sections[0]
    section.top_margin = section.bottom_margin = Inches(0.75)
    section.left_margin = section.right_margin = Inches(0.8)
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.12
    for style_name in ["Title", "Heading 1", "Heading 2"]:
        style = document.styles[style_name]
        style.font.name = "Calibri"
        style.font.color.rgb = RGBColor.from_string("000000")
        style.paragraph_format.keep_with_next = True
    document.styles["Title"].font.size = Pt(28)
    document.styles["Heading 1"].font.size = Pt(18)
    document.styles["Heading 2"].font.size = Pt(12)
    # Some installed python-docx templates inherit a decorative title border.
    for style in document.styles:
        for border in list(style.element.iter(qn("w:pBdr"))):
            border.getparent().remove(border)
    document.styles["Subtitle"].font.color.rgb = RGBColor.from_string("555555")
    # Major sections begin on fresh pages; questions stay with their answers.
    document.styles["Heading 1"].paragraph_format.page_break_before = True
    text = (ROOT / "docs/project_guide.md").read_text(encoding="utf-8")
    for block in text.split("\n\n"):
        if block.startswith("# "):
            document.add_paragraph(block[2:], "Title")
            document.add_paragraph("Project explanation and interview preparation", "Subtitle")
        elif block.startswith("### "):
            document.add_paragraph(block[4:], "Heading 2")
        elif block.startswith("## "):
            if block.startswith("## 1 "):
                document.add_paragraph("Contents", "Heading 2")
                for line in text.splitlines():
                    if line.startswith("## "):
                        p = document.add_paragraph(line[3:])
                        p.paragraph_format.space_after = Pt(4)
            heading = document.add_paragraph(block[3:], "Heading 1")
            if block.startswith(("## 11 ", "## 12 ", "## References")):
                heading.paragraph_format.page_break_before = False
        elif block.strip():
            add_text(document.add_paragraph(), block.replace("\n", " "))
    document.core_properties.title = "Semantic Book Recommender Project Guide"
    document.core_properties.subject = "Implementation walkthrough and interview questions"
    document.core_properties.author = ""
    document.save(ROOT / "docs/Project_Guide_and_Interview_Questions.docx")


if __name__ == "__main__":
    main()
