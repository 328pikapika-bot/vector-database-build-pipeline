"""Apply the requested caption and table formatting to the generated DOCX."""

from __future__ import annotations

import os

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph


CAPTION_PREFIX = "表 "
TARGET_CAPTIONS = {
    "表 5-2 元数据集合",
    "表 5-3 全文切分集合",
    "表 5-4 图片向量化集合",
}
DIAGRAMS = (
    ("3.1 总体数据流", "01-overall-architecture.png", "图 3-1 总体数据流"),
    ("6.1 元数据入库流程图", "02-metadata-flow.png", "图 6-1 元数据入库流程"),
    ("7.1 全文切片入库流程图", "03-fulltext-flow.png", "图 7-1 全文切片入库流程"),
    ("8.1 图片入库流程图", "04-image-flow.png", "图 8-1 图片入库流程"),
    ("9.2 增量更新流程图", "06-incremental-flow.png", "图 9-1 增量更新流程"),
    ("10.1 补偿处理流程图", "05-compensator-flow.png", "图 10-1 失败补偿流程"),
)


def set_run_font(run, *, bold: bool) -> None:
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.color.rgb = RGBColor(0, 0, 0)
    if run._element.rPr is None:
        run._element.get_or_add_rPr()
    fonts = run._element.rPr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        run._element.rPr.insert(0, fonts)
    fonts.set(qn("w:ascii"), "Times New Roman")
    fonts.set(qn("w:hAnsi"), "Times New Roman")
    fonts.set(qn("w:eastAsia"), "宋体")


def clear_shading(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    for shading in tc_pr.findall(qn("w:shd")):
        tc_pr.remove(shading)


def keep_row_together(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:tblHeader")) is None:
        tr_pr.append(OxmlElement("w:tblHeader"))


def make_caption(doc: Document, text: str) -> Paragraph:
    paragraph = doc.add_paragraph()
    paragraph.style = doc.styles["Caption"]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Pt(0)
    paragraph.paragraph_format.left_indent = Pt(0)
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(text)
    set_run_font(run, bold=False)
    return paragraph


def table_caption_from_context(table_index: int, previous_heading: str) -> str:
    heading = " ".join(previous_heading.split())
    return f"表 {table_index} {heading or '字段说明'}"


def insert_diagrams(document: Document, assets_dir: str) -> None:
    heading_map = {paragraph.text.strip(): paragraph for paragraph in document.paragraphs}
    for heading, file_name, caption_text in DIAGRAMS:
        if any(paragraph.text.strip() == caption_text for paragraph in document.paragraphs):
            continue
        heading_paragraph = heading_map.get(heading)
        if heading_paragraph is None:
            raise RuntimeError(f"Missing diagram heading: {heading}")
        image_path = os.path.join(assets_dir, file_name)
        if not os.path.isfile(image_path):
            raise RuntimeError(f"Missing diagram image: {image_path}")

        image_paragraph = document.add_paragraph()
        image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        image_paragraph.paragraph_format.first_line_indent = Pt(0)
        image_paragraph.paragraph_format.left_indent = Pt(0)
        image_paragraph.paragraph_format.space_before = Pt(6)
        image_paragraph.paragraph_format.space_after = Pt(3)
        image_paragraph.add_run().add_picture(
            image_path,
            width=Inches(5.8 if file_name == "01-overall-architecture.png" else 4.6),
        )

        caption = make_caption(document, caption_text)
        heading_paragraph._p.addnext(image_paragraph._p)
        image_paragraph._p.addnext(caption._p)


def resize_existing_diagrams(document: Document) -> None:
    diagram_count = min(len(document.inline_shapes), len(DIAGRAMS))
    for index in range(diagram_count):
        shape = document.inline_shapes[index]
        width = Inches(5.8 if index == 0 else 4.6)
        if shape.width == width:
            continue
        scale = width / shape.width
        shape.width = width
        shape.height = int(shape.height * scale)


def set_all_text_black(document: Document) -> None:
    paragraph_groups = [document.paragraphs]
    for section in document.sections:
        paragraph_groups.extend((section.header.paragraphs, section.footer.paragraphs))
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraph_groups.append(cell.paragraphs)

    for paragraphs in paragraph_groups:
        for paragraph in paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(0, 0, 0)


def main() -> None:
    source = os.environ["SOURCE_DOCX"]
    target = os.environ["TARGET_DOCX"]
    assets_dir = os.environ["ASSETS_DIR"]
    document = Document(source)

    target_caption_paragraphs = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text in TARGET_CAPTIONS:
            target_caption_paragraphs.append(paragraph)
            paragraph.style = document.styles["Caption"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.left_indent = Pt(0)
            paragraph.paragraph_format.space_before = Pt(6)
            paragraph.paragraph_format.space_after = Pt(6)
            for run in paragraph.runs:
                set_run_font(run, bold=False)

    last_heading = ""
    table_number = 0
    body_children = list(document.element.body.iterchildren())
    for position, child in enumerate(body_children):
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            if paragraph.style.name.startswith("Heading") and paragraph.text.strip():
                last_heading = paragraph.text.strip()
            continue
        if not child.tag.endswith("}tbl"):
            continue

        table_number += 1
        table = Table(child, document)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True

        preceding = body_children[position - 1] if position > 0 else None
        has_caption = False
        if preceding is not None and preceding.tag.endswith("}p"):
            previous_paragraph = Paragraph(preceding, document)
            has_caption = previous_paragraph.text.strip().startswith(CAPTION_PREFIX)
        if not has_caption:
            caption = make_caption(document, table_caption_from_context(table_number, last_heading))
            child.addprevious(caption._p)

        for row_index, row in enumerate(table.rows):
            keep_row_together(row)
            if row_index == 0:
                repeat_header(row)
            for cell in row.cells:
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                clear_shading(cell)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.first_line_indent = Pt(0)
                    paragraph.paragraph_format.left_indent = Pt(0)
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for run in paragraph.runs:
                        set_run_font(run, bold=False)

    insert_diagrams(document, assets_dir)
    resize_existing_diagrams(document)
    set_all_text_black(document)
    document.core_properties.title = "NSTL向量库服务设计方案"

    document.save(target)


if __name__ == "__main__":
    main()
