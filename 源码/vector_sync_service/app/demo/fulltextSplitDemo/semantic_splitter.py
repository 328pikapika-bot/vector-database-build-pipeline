"""语义化段落切分器 — 基于句子/从句边界 + LaTeX 保护"""
import re
from typing import List, Tuple, Optional


LATEX_INLINE = re.compile(r'(?<!\\)\$(.+?)(?<!\\)\$')
LATEX_DISPLAY = re.compile(r'(?<!\\)\$\$(.+?)(?<!\\)\$\$')
LATEX_PAREN = re.compile(r'\\\((.+?)\\\)')
LATEX_BRACKET = re.compile(r'\\\[(.+?)\\\]')
SENTENCE_END = re.compile(r'[.!?](?=\s+[A-Z])')
CLAUSE_BOUNDARY = re.compile(r'[,;:](?=\s)')
SPACE_BOUNDARY = re.compile(r'\s(?=\S)')


def _find_latex_zones(text: str) -> List[Tuple[int, int]]:
    """找出所有 LaTeX 公式区间 [(start, end), ...]，内部不可切分"""
    zones = []
    for pattern in [LATEX_DISPLAY, LATEX_INLINE, LATEX_BRACKET, LATEX_PAREN]:
        for m in pattern.finditer(text):
            zones.append((m.start(), m.end()))
    zones.sort()
    merged = []
    for start, end in zones:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _is_in_latex_zone(pos: int, zones: List[Tuple[int, int]]) -> bool:
    """判断位置是否在 LaTeX 公式内部"""
    for start, end in zones:
        if start < pos < end:
            return True
    return False


def _find_nearest_latex_boundary(pos: int, zones: List[Tuple[int, int]],
                                  text_len: int) -> Optional[int]:
    """找到最近的非 LaTeX 区域边界"""
    candidates = []
    for start, end in zones:
        if start <= pos <= end:
            if start > 0:
                candidates.append(start)
            if end < text_len:
                candidates.append(end)
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - pos))


def find_semantic_split(text: str, target_pos: int, search_window: int = 200) -> int:
    """
    在 target_pos 附近找到最佳语义切分点

    Args:
        text: 段落原文
        target_pos: 目标切分位置 (字符偏移)
        search_window: 向前搜索范围

    Returns:
        调整后的切分位置
    """
    text_len = len(text)
    if target_pos >= text_len:
        return text_len
    if target_pos <= 0:
        return target_pos

    latex_zones = _find_latex_zones(text)

    # 如果目标位置在 LaTeX 公式内，先跳出公式区间
    if _is_in_latex_zone(target_pos, latex_zones):
        boundary = _find_nearest_latex_boundary(target_pos, latex_zones, text_len)
        if boundary is not None:
            target_pos = boundary

    # 搜索区域: [target_pos - search_window, target_pos]
    search_start = max(0, target_pos - search_window)
    search_text = text[search_start:target_pos]

    # 优先级 1: 句子结尾 (. ! ? 后跟空格+大写字母)
    for m in reversed(list(SENTENCE_END.finditer(search_text))):
        cut_pos = search_start + m.end()
        if not _is_in_latex_zone(cut_pos, latex_zones):
            return cut_pos

    # 优先级 2: 从句边界 (, ; : 后跟空格)
    for m in reversed(list(CLAUSE_BOUNDARY.finditer(search_text))):
        cut_pos = search_start + m.end()
        if not _is_in_latex_zone(cut_pos, latex_zones):
            return cut_pos

    # 优先级 3: 空格边界 (不切破单词)
    for m in reversed(list(SPACE_BOUNDARY.finditer(search_text))):
        cut_pos = search_start + m.end()
        if not _is_in_latex_zone(cut_pos, latex_zones):
            return cut_pos

    # 优先级 4: 硬回退
    return target_pos


def semantic_split(text: str, max_chars: int = 1500, overlap_chars: int = 200,
                   search_window: int = 200) -> List[dict]:
    """
    对长文本执行语义化切分

    Args:
        text: 段落原文
        max_chars: 每个子块最大字符数
        overlap_chars: 相邻子块重叠字符数
        search_window: 语义边界搜索范围

    Returns:
        [{"content": str, "para_start": int, "para_end": int}, ...]
    """
    if len(text) <= max_chars:
        return [{"content": text, "para_start": 0, "para_end": len(text)}]

    chunks = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        raw_end = min(pos + max_chars, text_len)
        if raw_end >= text_len:
            # 最后一块: 取到末尾
            chunk_text = text[pos:]
            chunks.append({
                "content": chunk_text,
                "para_start": pos,
                "para_end": text_len,
            })
            break

        # 语义切分点
        best_end = find_semantic_split(text, raw_end, search_window)
        # 保证最小块长度不低于 max_chars * 0.3
        min_length = int(max_chars * 0.3)
        if best_end - pos < min_length:
            best_end = raw_end

        chunk_text = text[pos:best_end]
        chunks.append({
            "content": chunk_text,
            "para_start": pos,
            "para_end": best_end,
        })

        if best_end >= text_len:
            break

        # 下一块起点 = 当前块终点 - overlap，但必须向前推进
        next_pos = best_end - overlap_chars
        if next_pos <= pos:
            next_pos = best_end  # 避免死循环
        pos = next_pos

    return chunks
