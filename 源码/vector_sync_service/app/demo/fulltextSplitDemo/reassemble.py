"""从切分后的 chunk JSONL 文件还原 MMD 全文文档"""
import json
import os
import glob
from typing import List, Dict, Optional


def reassemble_mmd(chunks_path: str, result_json_path: Optional[str] = None,
                   overlap_chars: int = 200) -> str:
    """
    读取 JSONL 切分数据，还原为完整的 MMD 文档

    Args:
        chunks_path: chunks 的 JSONL 文件路径
        result_json_path: AST 解析结果 JSON 路径 (用于补充 REFERENCES 内容)
        overlap_chars: 固定长度切分的重叠字符数 (默认 200)

    Returns:
        还原后的 MMD 全文文本
    """
    section_headers: List[dict] = []
    paragraphs: List[dict] = []
    fixed_chunks: List[dict] = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            ctype = chunk.get("chunkType", "")

            if ctype == "section_header":
                section_headers.append(chunk)
            elif ctype == "paragraph":
                paragraphs.append(chunk)
            elif ctype == "fixed_chunk":
                fixed_chunks.append(chunk)

    fixed_by_parent: Dict[str, List[dict]] = {}
    for fc in fixed_chunks:
        parent_id = fc.get("parentParaId", "")
        if parent_id not in fixed_by_parent:
            fixed_by_parent[parent_id] = []
        fixed_by_parent[parent_id].append(fc)
    for chunks_list in fixed_by_parent.values():
        chunks_list.sort(key=lambda c: c.get("paraStartIdx", 0))

    resolved_content: Dict[str, str] = {}
    for p in paragraphs:
        pid = p.get("chunkId", "")
        if pid in fixed_by_parent:
            resolved_content[pid] = _merge_fixed_chunks(fixed_by_parent[pid], overlap_chars)
        else:
            resolved_content[pid] = p.get("content", "")

    section_headers.sort(key=lambda s: s.get("charStart", 0))
    paragraphs.sort(key=lambda p: p.get("charStart", 0))

    section_para_map: Dict[str, List[dict]] = {}
    for p in paragraphs:
        sp = p.get("sectionPath", "")
        if sp not in section_para_map:
            section_para_map[sp] = []
        section_para_map[sp].append(p)

    lines: List[str] = []

    # 从 result.json 补充 REFERENCES 内容
    references = _load_references(result_json_path, chunks_path)

    for sec in section_headers:
        title = sec.get("content", "")
        level = sec.get("level", 1)
        section_path = sec.get("sectionPath", title)

        # REFERENCES 由 result.json 单独提供，跳过 chunk 中的标题以避免重复
        if references and _is_reference_section(title):
            continue

        heading_prefix = "#" * level
        lines.append(f"{heading_prefix} {title}")
        lines.append("")

        paras = section_para_map.get(section_path, [])
        for p in paras:
            pid = p.get("chunkId", "")
            content = resolved_content.get(pid, p.get("content", ""))
            if content:
                lines.append(content)
                lines.append("")

    if references:
        lines.append(references)

    return "\n".join(lines).rstrip("\n") + "\n"


def _is_reference_section(title: str) -> bool:
    """判断标题是否为参考文献章节"""
    import re
    return bool(re.match(r"^(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)$", title.strip()))


def _load_references(result_json_path: Optional[str], chunks_path: str) -> str:
    """尝试从 AST 解析结果中加载 REFERENCES 章节"""
    ref_text = ""

    if result_json_path and os.path.exists(result_json_path):
        with open(result_json_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        ref_text = result_data.get("references", "")
        if ref_text:
            return ref_text.strip()

    # 回退: 从 chunks_path 同一目录搜索 result_*.json
    output_dir = os.path.dirname(chunks_path)
    for fname in sorted(os.listdir(output_dir)):
        if fname.startswith("result_") and fname.endswith(".json"):
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            ref_text = result_data.get("references", "")
            if ref_text:
                return ref_text.strip()

    return ""


def _merge_fixed_chunks(fixed_list: List[dict], overlap_chars: int = 200) -> str:
    """合并固定长度切分的子块，去除重叠部分 (兼容语义切分的可变重叠)"""
    if not fixed_list:
        return ""

    result_parts: List[str] = []
    prev_end = 0

    for i, fc in enumerate(fixed_list):
        text = fc.get("content", "")
        if i == 0:
            result_parts.append(text)
            prev_end = fc.get("paraEndIdx", len(text))
        else:
            # 语义切分的重叠不是固定的，使用 paraStartIdx 计算
            para_start = fc.get("paraStartIdx", 0)
            overlap = prev_end - para_start
            if overlap > 0 and overlap < len(text):
                result_parts.append(text[overlap:])
            else:
                result_parts.append(text)
            prev_end = fc.get("paraEndIdx", 0)

    return "".join(result_parts)


def main():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(demo_dir, "output")

    jsonl_files = [f for f in os.listdir(output_dir) if f.startswith("chunks_") and f.endswith(".jsonl")]

    if not jsonl_files:
        print("未找到 chunks JSONL 文件")
        return

    for jsonl_file in jsonl_files:
        jsonl_path = os.path.join(output_dir, jsonl_file)
        print(f"还原: {jsonl_file} ({os.path.getsize(jsonl_path) / 1024:.1f} KB)")

        mmd = reassemble_mmd(jsonl_path)

        sys_id = jsonl_file.replace("chunks_", "").replace(".jsonl", "")
        restore_name = f"restored_{sys_id}.mmd"
        restore_path = os.path.join(output_dir, restore_name)

        with open(restore_path, "w", encoding="utf-8") as f:
            f.write(mmd)

        lines = mmd.count("\n")
        chars = len(mmd)
        print(f"  → {restore_name} ({lines} 行, {chars} 字符)")


if __name__ == "__main__":
    main()
