"""
MMD/Markdown 文档解析器 (AST-based)
使用词法分析 (Lexer) + 递归下降解析 (Recursive Descent Parser) 构建抽象语法树 (AST)
"""
import re
from typing import List
from dataclasses import dataclass, field
from enum import Enum, auto


class TokenType(Enum):
    """Token 类型枚举"""
    HEADING = auto()
    TEXT = auto()
    BLANK = auto()
    EOF = auto()


@dataclass
class Token:
    """词法单元"""
    type: TokenType
    value: str
    line: int
    level: int = 0


class Lexer:
    """词法分析器: 将 MMD 原始文本转换为 Token 流"""

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

    def __init__(self, text: str):
        self.lines = text.split("\n")
        self.pos = 0

    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            match = self.HEADING_RE.match(line)
            if match:
                tokens.append(Token(
                    type=TokenType.HEADING,
                    value=match.group(2).strip(),
                    line=self.pos,
                    level=len(match.group(1))
                ))
            elif line.strip() == "":
                tokens.append(Token(type=TokenType.BLANK, value="", line=self.pos))
            else:
                tokens.append(Token(type=TokenType.TEXT, value=line, line=self.pos))
            self.pos += 1
        tokens.append(Token(type=TokenType.EOF, value="", line=self.pos))
        return tokens


# ======================== AST 节点定义 ========================

class ASTNode:
    """AST 基节点"""
    pass


@dataclass
class SectionNode(ASTNode):
    """章节 AST 节点"""
    title: str
    level: int
    start_pos: int
    end_pos: int
    children: List["SectionNode"] = field(default_factory=list)
    paragraphs: List["ParagraphNode"] = field(default_factory=list)
    content_only: str = ""


@dataclass
class ParagraphNode(ASTNode):
    """段落 AST 节点"""
    text: str
    start_pos: int
    end_pos: int
    section_path: str = ""
    section_level: int = 0


@dataclass
class DocumentNode(ASTNode):
    """文档根 AST 节点"""
    raw_text: str = ""
    title: str = ""
    sections: List[SectionNode] = field(default_factory=list)
    paragraphs: List[ParagraphNode] = field(default_factory=list)
    references: str = ""


# ======================== 递归下降解析器 ========================

class Parser:
    """递归下降解析器: 将 Token 流构建为 AST 文档树"""

    REF_PATTERNS = [
        re.compile(r"^#{1,3}\s*(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)\s*$",
                   re.MULTILINE | re.IGNORECASE),
    ]

    def __init__(self, tokens: List[Token], raw_text: str):
        self.tokens = tokens
        self.pos = 0
        self.raw_text = raw_text
        self.doc = DocumentNode(raw_text=raw_text)

    def parse(self) -> DocumentNode:
        if self._peek().type == TokenType.EOF:
            return self.doc

        # 第一阶段: 扁平解析所有章节
        flat_sections: List[SectionNode] = []
        while self._peek().type != TokenType.EOF:
            if self._peek().type == TokenType.HEADING:
                flat_sections.append(self._parse_section_flat())
            else:
                self._advance()

        if not flat_sections:
            # 无标题文档: 整个文本作为一个段落
            if self.raw_text.strip():
                para = ParagraphNode(
                    text=self.raw_text.strip(),
                    start_pos=0,
                    end_pos=len(self.raw_text),
                    section_path="",
                    section_level=0,
                )
                self.doc.paragraphs.append(para)
            return self.doc

        # 记录主标题
        if flat_sections[0].level == 1:
            self.doc.title = flat_sections[0].title

        # 第二阶段: 构建章节树 (父子关系)
        root = SectionNode(title="(root)", level=0, start_pos=0, end_pos=len(self.raw_text))
        stack: List[SectionNode] = [root]

        for section in flat_sections:
            while stack and stack[-1].level >= section.level:
                stack.pop()
            if stack:
                stack[-1].children.append(section)
            stack.append(section)

        self.doc.sections = flat_sections

        # 第三阶段: 提取每个章节的纯文本内容
        for i, section in enumerate(flat_sections):
            self._fill_section_content(section, i, flat_sections)

        # 第四阶段: 按空行分割段落
        for section in flat_sections:
            section_path = self._build_section_path(section, flat_sections)
            paras = self._split_paragraphs(section, section_path)
            section.paragraphs = paras
            self.doc.paragraphs.extend(paras)

        # 第五阶段: 检测参考文献
        self._detect_references()

        return self.doc

    def _parse_section_flat(self) -> SectionNode:
        """解析单个章节 (第一遍: 扁平)"""
        token = self._advance()
        start_pos = token.line

        # 计算章节结束位置: 下一个同级或更高级标题之前
        end_line = len(self.raw_text.split("\n"))
        current_level = token.level
        save_pos = self.pos
        while save_pos < len(self.tokens):
            t = self.tokens[save_pos]
            if t.type == TokenType.HEADING and t.level <= current_level:
                end_line = t.line
                break
            save_pos += 1

        return SectionNode(
            title=token.value,
            level=token.level,
            start_pos=start_pos,
            end_pos=end_line,
        )

    def _fill_section_content(self, section: SectionNode, idx: int,
                               flat_sections: List[SectionNode]):
        """填充章节纯文本内容 (去除子章节)"""
        start_line = section.start_pos
        end_line = section.end_pos

        lines = self.raw_text.split("\n")
        section_lines = lines[start_line:end_line]

        # 跳过标题行
        content_start = 0
        for i, line in enumerate(section_lines):
            if Lexer.HEADING_RE.match(line):
                content_start = i + 1
                break

        # 收集到下一个子标题之前的内容
        result_lines: List[str] = []
        for line in section_lines[content_start:]:
            match = Lexer.HEADING_RE.match(line)
            if match and len(match.group(1)) > section.level:
                break
            result_lines.append(line)

        section.content_only = "\n".join(result_lines).strip()

    def _build_section_path(self, section: SectionNode,
                             flat_sections: List[SectionNode]) -> str:
        """构建章节层级路径"""
        path_parts = [section.title]
        # 向前找各级父章节
        current_level = section.level
        section_idx = flat_sections.index(section)
        for i in range(section_idx - 1, -1, -1):
            parent = flat_sections[i]
            if parent.level < current_level:
                path_parts.insert(0, parent.title)
                current_level = parent.level
        return "/".join(path_parts)

    def _split_paragraphs(self, section: SectionNode, section_path: str) -> List[ParagraphNode]:
        """将章节纯文本按空行分割为段落"""
        if not section.content_only.strip():
            return []

        paragraphs: List[ParagraphNode] = []
        blocks = re.split(r"\n\s*\n", section.content_only)

        current_line = section.start_pos + 1  # +1 for heading line
        for block in blocks:
            block = block.strip()
            if not block:
                current_line += block.count("\n") + 2
                continue

            block_lines = block.split("\n")
            block_start = current_line
            block_end = current_line + len(block_lines)

            para = ParagraphNode(
                text=block,
                start_pos=block_start,
                end_pos=block_end,
                section_path=section_path,
                section_level=section.level,
            )
            paragraphs.append(para)
            current_line = block_end + 2

        return paragraphs

    def _detect_references(self):
        """检测并标记参考文献部分"""
        for pattern in self.REF_PATTERNS:
            match = pattern.search(self.raw_text)
            if match:
                ref_start_line = self.raw_text[:match.start()].count("\n")
                ref_text = self.raw_text[match.start():].strip()
                self.doc.references = ref_text
                self.doc.paragraphs = [
                    p for p in self.doc.paragraphs if p.start_pos < ref_start_line
                ]
                break

    def _peek(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TokenType.EOF, "", -1)

    def _advance(self) -> Token:
        token = self._peek()
        self.pos += 1
        return token


# ======================== 兼容层: AST → 旧数据模型 ========================

@dataclass
class Section:
    """文档章节 (兼容旧接口)"""
    title: str = ""
    level: int = 0
    content: str = ""
    contentOnly: str = ""
    startPos: int = 0
    endPos: int = 0
    children: List["Section"] = field(default_factory=list)
    paragraphs: List["Paragraph"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "contentOnly": self.contentOnly,
            "startPos": self.startPos,
            "endPos": self.endPos,
            "children": [c.to_dict() for c in self.children],
            "paragraphs": [p.to_dict() for p in self.paragraphs],
        }


@dataclass
class Paragraph:
    """文档段落 (兼容旧接口)"""
    text: str = ""
    startPos: int = 0
    endPos: int = 0
    sectionLevel: int = 0
    sectionPath: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "startPos": self.startPos,
            "endPos": self.endPos,
            "sectionLevel": self.sectionLevel,
            "sectionPath": self.sectionPath,
        }


@dataclass
class ParsedDocument:
    """解析后的文档 (兼容旧接口)"""
    rawText: str = ""
    title: str = ""
    sections: List[Section] = field(default_factory=list)
    paragraphs: List[Paragraph] = field(default_factory=list)
    references: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "references": self.references,
        }


class MmdParser:
    """MMD/Markdown 文档解析器 (AST-based 入口)"""

    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    @staticmethod
    def parse(raw_text: str) -> ParsedDocument:
        """
        使用 AST 解析 MMD 全文为结构化文档

        Args:
            raw_text: .mmd 文件的原始文本内容

        Returns:
            ParsedDocument: 解析后的结构化文档
        """
        # Step 1: 词法分析
        lexer = Lexer(raw_text)
        tokens = lexer.tokenize()

        # Step 2: 递归下降解析 → 构建 AST
        parser = Parser(tokens, raw_text)
        ast_doc = parser.parse()

        # Step 3: AST → 兼容数据模型
        return MmdParser._ast_to_parsed_document(ast_doc)

    @staticmethod
    def _ast_to_parsed_document(ast_doc: DocumentNode) -> ParsedDocument:
        """将 AST 文档节点转换为兼容的 ParsedDocument"""
        # 转换章节
        sections = []
        for sec in ast_doc.sections:
            children = [MmdParser._ast_to_section(c) for c in sec.children]
            paragraphs = [MmdParser._ast_to_paragraph(p) for p in sec.paragraphs]
            section = Section(
                title=sec.title,
                level=sec.level,
                content="",
                contentOnly=sec.content_only,
                startPos=sec.start_pos,
                endPos=sec.end_pos,
                children=children,
                paragraphs=paragraphs,
            )
            sections.append(section)

        # 转换段落
        paragraphs = [MmdParser._ast_to_paragraph(p) for p in ast_doc.paragraphs]

        return ParsedDocument(
            rawText=ast_doc.raw_text,
            title=ast_doc.title,
            sections=sections,
            paragraphs=paragraphs,
            references=ast_doc.references,
        )

    @staticmethod
    def _ast_to_section(node: SectionNode) -> Section:
        children = [MmdParser._ast_to_section(c) for c in node.children]
        paragraphs = [MmdParser._ast_to_paragraph(p) for p in node.paragraphs]
        return Section(
            title=node.title,
            level=node.level,
            content="",
            contentOnly=node.content_only,
            startPos=node.start_pos,
            endPos=node.end_pos,
            children=children,
            paragraphs=paragraphs,
        )

    @staticmethod
    def _ast_to_paragraph(node: ParagraphNode) -> Paragraph:
        return Paragraph(
            text=node.text,
            startPos=node.start_pos,
            endPos=node.end_pos,
            sectionLevel=node.section_level,
            sectionPath=node.section_path,
        )

    @staticmethod
    def get_heading_title(raw_text: str) -> str:
        """获取文档第一个 H1 标题"""
        match = MmdParser.HEADING_PATTERN.search(raw_text)
        if match:
            return match.group(2).strip()
        return ""
