#!/usr/bin/env python3
"""

直接从 data_*/ 目录读取论文 .mmd 文件，存入 Milvus。

流程:
  1. 启动本地 HTTP 图片服务
  2. 遍历 data_*/ 下的论文目录，读取 .mmd 文件
  3. 按段落拆分，识别 ![](figures/<filename>) 图片引用，提取前后文上下文
  4. 多线程并行: pHash → VL-Instruct 深度解读 → Embedding API → Milvus 批量写入
"""

import argparse
import base64
import io
import json
import logging
import logging.handlers
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Tuple, Dict, List
from urllib.parse import urlparse
import shutil

import imagehash
import numpy as np
import requests
from PIL import Image
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection,
    Function, FunctionType, utility,
)

# =============================================================================
# 配置
# =============================================================================
MILVUS_HOST = "http://192.0.2.14"
MILVUS_PORT = "19530"
COLLECTION_NAME = "scientific_literature_images"

COMMON_API_KEY = os.environ["COMMON_API_KEY"]
VL_CHAT_URL = "http://192.0.2.12:50019/v1/chat/completions"
VL_CHAT_MODEL = "qwen3.5-plus"
VL_EMBED_URL = "http://192.0.2.11:8050/v1/embeddings"
VL_EMBED_MODEL = "8b"
TEXT_EMBED_URL = "http://192.0.2.13:8080/v1/embeddings"
TEXT_EMBED_MODEL = "medium"
VL_CHAT_MAX_TOKENS = 5000

DIM_PHASH = 64
DIM_VISUAL_EMBED = 4096
DIM_TEXT_EMBED = 2560
DIM_VISUAL_TEXT_EMBED = 4096

# Dense-vector indexes are all IVF_PQ.  m=64 divides both 2560 and 4096;
# nbits=8 is the storage-oriented default used by the service.
IVFPQ_NLIST = 4096
IVFPQ_M = 64
IVFPQ_NBITS = 8

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8000
DATA_DIR_NAME = "data_1k"

MAX_PIXELS = 2_000_000
IMAGE_REF_PATTERN = re.compile(r"!\[.*?\]\(figures/([^)]+)\)")
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")
FILENAME_PATTERN = re.compile(
    r"^\d+-\d+-whxyp_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)_(\d+)\.jpg$"
)

# =============================================================================
# 日志
# =============================================================================

def log_setup(log_dir: str = "logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    fh = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "massive_ingestion.log"),
        when="midnight", interval=1, backupCount=30, encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG); ch.setFormatter(fmt); root.addHandler(ch)

_elapsed_timers: dict[str, float] = {}

def log_start_timer(name: str) -> None:
    _elapsed_timers[name] = time.perf_counter()

def log_stop_timer(name: str) -> float:
    t = time.perf_counter() - _elapsed_timers.pop(name, time.perf_counter())
    return t

def log_info(msg: Any) -> None:
    logging.info(_log_indent(msg))

def log_warning(msg: Any) -> None:
    logging.warning(_log_indent(msg))

def log_error(msg: Any) -> None:
    logging.error(_log_indent(msg))

def _log_indent(input: Any) -> str:
    import inspect as _ins
    stack = _ins.stack()
    depth = len(stack) - 2
    caller = stack[2].function
    indent = "  " * depth
    prefix = f"{indent}【{caller}】 "
    lines = str(input).splitlines()
    return "\n" + "\n".join(prefix + line for line in lines)


# =============================================================================
# Embedding API
# =============================================================================

def _call_embedding_api(model: str, input_payload, api_key: str, api_url: str,
                        use_messages_format: bool = False) -> List[float]:
    endpoint = api_url
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if use_messages_format:
        payload = {"model": model, "messages": [{"role": "user", "content": input_payload}],
                   "encoding_format": "float"}
    else:
        payload = {"model": model, "input": input_payload, "encoding_format": "float"}
    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            break
        except Exception:
            wait = min(2.0 * (2 ** (attempt - 1)), 300.0)
            log_info(f"Embed API attempt {attempt} failed, retrying in {wait:.0f}s...")
            time.sleep(wait)
    try:
        return data["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected embedding API response: {data}") from exc


def embed_image_url(url: str) -> np.ndarray:
    raw = _call_embedding_api(
        model=VL_EMBED_MODEL,
        input_payload=[{"type": "image_url", "image_url": {"url": url}}],
        api_key=COMMON_API_KEY, api_url=VL_EMBED_URL, use_messages_format=True,
    )
    vec = np.array(raw, dtype=np.float16)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


def embed_text(text: str) -> np.ndarray:
    raw = _call_embedding_api(
        model=TEXT_EMBED_MODEL, input_payload=text,
        api_key=COMMON_API_KEY, api_url=TEXT_EMBED_URL,
    )
    vec = np.array(raw, dtype=np.float16)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


def embed_image_text(url: str, text: str) -> np.ndarray:
    raw = _call_embedding_api(
        model=VL_EMBED_MODEL,
        input_payload=[{"type": "text", "text": text},
                       {"type": "image_url", "image_url": {"url": url}}],
        api_key=COMMON_API_KEY, api_url=VL_EMBED_URL, use_messages_format=True,
    )
    vec = np.array(raw, dtype=np.float16)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


# =============================================================================
# VL-Instruct 提示词
# =============================================================================

DRAFT_SYSTEM_PROMPT = r"""# Role & Objective
你是一个专为高级多模态语义检索（Semantic Retrieval）系统设计的科学图像深度分析专家。你的核心任务是：穿透图像的表层像素，结合附加的文本上下文，深度提取并编码图像的科学内涵，最终输出结构化的JSON数据，以便于嵌入模型（Embedding Models）进行高质量的语义向量化。

# 分析逻辑与工作流 (Workflow)
请严格遵循以下"由表及里"的层层递进逻辑进行分析：

## 第一层：领域鉴别与类型锚定 (Domain & Type)
1. **领域判断**：首先判断图片是否属于【理、工、农、医】四大核心科学领域。如果不确定或属于文学、艺术、日常等非科学领域，请果断判定为"不相关"，并仅就图片本身内容进行客观描述，不要解释"为什么不是科学图片"或"图片中缺少什么科学元素"。
2. **图像分类**：准确判定图片类型。
   - `chart`：数据图表（折线图、柱状图、散点图、热力图等）
   - `formula`：数学公式、物理方程、化学反应式
   - `molecule`：化学分子式、晶体结构
   - `diagram`：机制示意图、系统架构图、流程图、实验装置图、合成路线、原理图
   - `scientific_photo`：实拍影像（如：光学/电子显微镜照片、医学影像CT/MRI/X光、天文观测图像、地质剖面、生物标本、实验现象记录等）
   - `other`：无法归入以上的其他图像

## 第二层：表层信息提取 (Surface Extraction)
1. **OCR关键文本**：提取图中最具科学意义的文本（限500字）。包含有意义的单词、坐标轴标签、图例说明；忽略孤立无意义的字母、数字、元素符号、分子基团。
2. **视觉特征**：描述图片的排版、颜色分布、对比度特征（如免疫荧光染色颜色）、箭头或标注框的指向。

## 第三层：深度语义解读 (Deep Semantics - 核心重点)
结合提供的"上下文（Context）"，对图像进行深度解读（200-1000字）。请根据图像类型采用特定的解读策略：
- **对于 `scientific_photo`（科学实拍）**：不要只描述"图里有什么"，要指出"这意味着什么"。识别成像模态（如fMRI、冷冻电镜）；描述观测对象的形态学特征（如细胞凋亡特征、组织病变、材料断口形貌）；解释图像中特殊对比度或染色所代表的理化性质或生理状态。
- **对于 `chart`（数据图表）**：指出变量间的映射关系，描述核心趋势、关键拐点、极值或异常值，以及这些数据支持了上下文中的何种科学结论。
- **对于 `diagram`（示意图）**：梳理系统运作的动态机制、要素间的因果关系或物质/信号传导通路。
- **对于 `formula` / `molecule`**：阐明该公式或分子的物理/化学意义及其在上下文中的作用。

## 第四层：高密度知识提炼 (Knowledge Distillation)
提取核心科学实体（物质名称、病理学术语、算法等），并生成5-10个用于倒排索引的高权重领域关键词。

---

# 严格约束条件 (Strict Constraints)
1. **输出格式**：必须且只能输出合法的JSON格式，不要包含任何Markdown代码块语法（如 ```json ），直接输出JSON字符串。
2. **拒绝重复**：各字段内容必须相互独立，绝不重复相同的短语或句子。
3. **LaTeX规范**：
   - 所有的数学表达式、分子式、上下标必须使用LaTeX，用 `$...$`（行内）或 `$$...$$`（独立行）包裹。化学式可使用 `\ce{}`。
   - 对于由纯字母组成的LaTeX命令（如 `\cdots`, `\alpha`），若其后紧跟普通英文字母，**必须**在命令和后续字母之间添加一对空大括号 `{}` 进行显式截断（如：必须输出 `\cdots{}Cu`，绝不可输出 `\cdotsCu` 或 `\cdots Cu`）。
4. **客观真实**：遇到模糊或无法识别的内容，对应字段的值设为 `null` 或标注为"无法识别"，绝不编造（幻觉）。基团等特征仅需枚举一次，逗号分隔。

# 以图片为中心
虽然可能附有文字，但是，这些文字可能只是位于原始文献中的图片位置附近，而不一定真的与图片有关联关系。如果文字与图片无关，甚至与图片信息矛盾，则只关注图片本身。
---

# JSON 输出结构 (JSON Schema)

{
  "domain_relevance": {
    "is_relevant": true/false,
    "domain": "理/工/农/医/组合/无",
    "reasoning": "简明扼要的判断理由"
  },
  "basic_info": {
    "image_type": "chart|formula|molecule|diagram|scientific_photo|other",
    "inferred_title": "结合上下文推断的图片标题或核心主题"
  },
  "surface_extraction": {
    "ocr_text": "提取的关键文本，无则为空字符串",
    "visual_features": "视觉特征描述，例如：左下角可见比例尺，高亮区域呈现红色荧光..."
  },
  "deep_semantics": "深度语义解读，详细描述内在科学规律、病理特征或数据趋势，200-500字。注意：对于判定为"不相关"的非科学图片，只需客观描述图像内容本身，不要出现"图片中没有科学内容"、"缺少图表公式"等元描述语句。",
  "structured_data": {
    "chart_data_trends": "图表关键数据点或趋势描述 (非图表则为null)",
    "formula_latex": "提取的LaTeX代码 (非公式则为null)",
    "scientific_entities": ["实体1", "实体2"]
  },
  "search_keywords": ["关键词1", "关键词2", "关键词3"]
}
"""

FINAL_SYSTEM_PROMPT = r"""# Role & Objective
你是一位严谨的"科学图像分析质量改进专家"。你的任务是对另一模型生成的《图像深度分析JSON初稿》进行审慎的复核与优化，输出一份更准确、更完整的修正版JSON。

你将接收到：
1. **原始图像**
2. **原始附加文本上下文** (Context)
3. **初版JSON输出** (Draft JSON)

你的目标是：在初稿的基础上，**保留真实正确的信息，修正明显的错误，补充遗漏的细节**，输出最终修正版。

# 核心改进准则 (Refinement Criteria)

## 1. 事实核查与修正 (Fact-checking)
- **保留真实信息**：初版中能在图片中找到直接视觉证据的内容，应当保留。不要无故删除准确的信息。
- **删除明显幻觉**：初版中提到的数字、实体、趋势如果在图片中毫无依据，应予以删除或标注为"无法确认"。
- **修正过度解读**：`deep_semantics` 中是否将图片未展示的内容强行解释？如果有，修正为更审慎的表述，但不要整段删除有价值的分析。
- **补充遗漏细节**：检查图片中明显存在但初版遗漏的重要信息（如关键数据点、明显的视觉特征、重要的标注文本等），予以补充。

## 2. 格式与LaTeX规范检查 (Formatting Check)
- **LaTeX大括号截断**：检查所有LaTeX宏命令（如 `\alpha`, `\cdots`, `\text` 等）后是否紧跟普通英文字母。如果是，**添加空大括号 `{}` 进行截断**。
  - 错误：`\cdotsCu` → 正确：`\cdots{}Cu`
- **公式闭合**：确保 `$...$` 和 `$$...$$` 正确闭合。

## 3. 领域与逻辑复核 (Domain Review)
- **domain_relevance**：如果初版将明显非科学的图片强行归类为科学领域，予以修正。如果是科学图片但判断有误，修正领域分类。
- **image_type**：确保图片类型分类准确。

# 输出要求

请输出一个JSON对象，包含两个顶层字段：
1. `audit_report`：简要记录你做了哪些修改（如"补充了遗漏的OCR文本"、"修正了domain判断"等）。如果没有修改，填"无"。
2. `final_corrected_json`：修正后的完整JSON，结构与初版完全一致。

**约束**：
- 直接输出JSON字符串，不要包含Markdown代码块（如 ```json ）。
- `deep_semantics` 应当完整、连贯，不要大幅删减有价值的分析。
- `scientific_entities` 和 `search_keywords` 应基于图片实际内容，不要凭空捏造，但也不要遗漏明显的实体。

JSON Schema:
{
  "audit_report": {
    "changes_made": ["列出主要修改点，如果没有填 '无'"],
    "additions": ["补充的内容"],
    "corrections": ["修正的错误"]
  },
  "final_corrected_json": {
    "domain_relevance": {
      "is_relevant": true/false,
      "domain": "理/工/农/医/组合/无",
      "reasoning": "判断理由"
    },
    "basic_info": {
      "image_type": "chart|formula|molecule|diagram|scientific_photo|other",
      "inferred_title": "图片标题或核心主题"
    },
    "surface_extraction": {
      "ocr_text": "关键文本",
      "visual_features": "视觉特征描述"
    },
    "deep_semantics": "完整、准确的深度语义解读",
    "structured_data": {
      "chart_data_trends": "图表趋势（如适用）",
      "formula_latex": "LaTeX代码（如适用）",
      "scientific_entities": ["实体1", "实体2"]
    },
    "search_keywords": ["关键词1", "关键词2"]
  }
}
"""

CONTEXT_CLEAN_PROMPT = r"""# Role & Objective
你是一个科学文本清洗专家。你的任务是从科学论文的原始文本上下文中，移除所有不适合用于语义嵌入计算的内容，同时保留有意义的科学叙述文字。

# 需要移除的内容类型 (Noise Types to Remove)
请识别并删除以下类型的无意义内容（这些内容对语义检索没有帮助，反而会降低嵌入质量）：

1. **OCR 乱码**：不形成有意义单词的字符序列。例如："T h e s p 1 n - l a t ti c e"、HTML 实体（&#x2206;）、乱码符号串。
2. **纯数学公式（LaTeX）**：完全由数学符号组成的 LaTeX 表达式，如 `$$E = mc^2$$`、`$\int_0^\infty f(x)dx$`、`\begin{equation}...\end{equation}`。注意：公式周围的叙述性文字（如"根据以下方程..."）应保留。
3. **元素符号/分子式字符串**：纯化学符号或分子式序列，如 "Fe-Cr-Ni-Mo"、"H2O NaCl CO2"、"C6H12O6"。
4. **无意义的数字/符号序列**：如 "1. 2. 3."、页码、纯标点符号行。
5. **参考文献编号或格式标记**：如 "[1]", "[2,3]", "(Fig. 1)" 等孤立引用标记。

# 需要保留的内容 (Content to Preserve)
请保留以下有语义价值的科学文字：
- 叙述性的科学描述、方法论说明、实验结论
- 图像说明文字（图注）中有意义的描述部分
- 科学术语、概念解释、研究背景介绍
- 公式中拆解出来的有意义关键词（如"质量"、"能量"、"速度"）

# 处理策略
- **部分删除**：如果一段文字中包含少量的公式或无意义序列，只删除那些无意义部分，保留周围的叙述文字。
- **整段删除**：如果整段文字全是公式、乱码或元素符号，直接整段删除。
- **合并短片段**：删除无意义内容后，如果有多个短文本片段，将它们合并为连贯的段落。
- **保留空行分隔**：不同来源的文字段落之间用空行分隔。

# 输出格式
直接输出一个 JSON 对象：
{
  "cleaned_context": "清洗后的文本内容",
  "removed_items": ["被删除的内容摘要1", "被删除的内容摘要2"],
  "removal_ratio": 0.35
}

注意：
- `cleaned_context` 是清洗后的完整文本。
- `removed_items` 列出被删除的内容的简短摘要。
- `removal_ratio` 是被删除内容占原始文本的比例（0.0 ~ 1.0）。
- 如果原始文本中没有需要清洗的内容，`cleaned_context` 与原始文本完全相同，`removed_items` 为空数组，`removal_ratio` 为 0。
- 直接输出 JSON 字符串，不要包含 Markdown 代码块（如 ```json ）。
"""


# =============================================================================
# VL-Instruct 客户端
# =============================================================================

class VLInstructClient:
    """多模态模型客户端 —— 图像深度解读 (两阶段: Draft → Review)"""

    def __init__(self, api_key: str = COMMON_API_KEY, api_url: str = VL_CHAT_URL,
                 model: str = VL_CHAT_MODEL, temperature: float = 0.2,
                 max_tokens: int = VL_CHAT_MAX_TOKENS):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ---- 图片 URL 处理 ----
    @staticmethod
    def _pil_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime = "image/png" if fmt == "PNG" else f"image/{fmt.lower()}"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _to_image_url(image_source: str) -> str:
        if image_source.startswith("data:"):
            return image_source
        HTTP_BASE = "http://192.0.2.10:8000/"
        temp_dir = Path("out/temp_resized")
        temp_dir.mkdir(parents=True, exist_ok=True)

        if image_source.startswith(("http://", "https://")):
            try:
                resp = requests.get(image_source, timeout=30)
                resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError(f"Failed to download image from {image_source}: {exc}")
            parsed = urlparse(image_source)
            orig_filename = os.path.basename(parsed.path) or "downloaded.jpg"
            temp_path = temp_dir / f"dl_{orig_filename}"
            with open(temp_path, "wb") as f:
                f.write(resp.content)
            with Image.open(temp_path) as img:
                w, h = img.size
                pixels = w * h
            if pixels > MAX_PIXELS:
                with Image.open(temp_path) as img:
                    scale = (MAX_PIXELS / pixels) ** 0.5
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    orig_stem = Path(orig_filename).stem
                    resized_name = f"{orig_stem}_resized.jpg"
                    resized_path = temp_dir / resized_name
                    img.save(resized_path, format="JPEG", quality=90)
                temp_path.unlink(missing_ok=True)
                return f"{HTTP_BASE}out/temp_resized/{resized_name}"
            else:
                final_name = orig_filename
                final_path = temp_dir / final_name
                if temp_path.name.startswith("dl_"):
                    shutil.move(str(temp_path), str(final_path))
                return f"{HTTP_BASE}out/temp_resized/{final_name}"

        if not os.path.isfile(image_source):
            raise FileNotFoundError(f"Image file not found: {image_source}")
        with Image.open(image_source) as img:
            w, h = img.size
            pixels = w * h
        if pixels > MAX_PIXELS:
            with Image.open(image_source) as img:
                scale = (MAX_PIXELS / pixels) ** 0.5
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                orig_name = Path(image_source).stem
                temp_name = f"{orig_name}_resized.jpg"
                temp_path = temp_dir / temp_name
                img.save(temp_path, format="JPEG", quality=90)
            return f"{HTTP_BASE}out/temp_resized/{temp_name}"
        else:
            abs_path = os.path.abspath(image_source)
            proj_root = os.path.abspath(os.getcwd())
            if abs_path.startswith(proj_root):
                rel_path = os.path.relpath(abs_path, proj_root).replace("\\", "/")
            else:
                temp_name = Path(image_source).name
                temp_path = temp_dir / temp_name
                shutil.copy2(image_source, temp_path)
                rel_path = f"out/temp_resized/{temp_name}"
            return f"{HTTP_BASE}{rel_path}"

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```\s*$", "", text)
        return text.strip()

    def _call_api(self, system_prompt: str, user_content: list) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=500)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected API response: {data}") from exc

    def _clean_context(self, raw_context: str) -> str:
        user_content: list = [{
            "type": "text",
            "text": ("请清洗以下科学论文文本上下文，移除无意义的OCR乱码、纯数学公式、"
                     "元素符号序列等内容，保留有语义价值的科学叙述文字。\n\n"
                     f"原始上下文：\n{raw_context}"),
        }]
        raw = self._call_api(system_prompt=CONTEXT_CLEAN_PROMPT, user_content=user_content)
        raw = self._extract_json(raw)
        result = json.loads(raw)
        cleaned = result.get("cleaned_context") or ""
        removed = result.get("removed_items") or []
        ratio = result.get("removal_ratio") or 0
        if not cleaned and not removed:
            raise ValueError("Context cleaning returned empty result with no cleaning evidence")
        if removed:
            log_info(f"Context cleaned: {len(raw_context)} -> {len(cleaned)} chars ({ratio:.0%} removed)")
        return cleaned

    def interpret_image(self, image_source: str, context: str = "",
                        skip_review: bool = False) -> dict:
        image_url = self._to_image_url(image_source)
        if context:
            context = self._clean_context(context)

        # ---- Stage 1: Draft ----
        draft_user_content: list = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": f"上下文（Context）：\n{context}" if context else "请直接分析上述图片，无额外上下文。"},
        ]
        draft_raw = self._call_api(system_prompt=DRAFT_SYSTEM_PROMPT, user_content=draft_user_content)
        draft_raw = self._extract_json(draft_raw)
        try:
            draft_json = json.loads(draft_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Draft stage returned invalid JSON:\n{draft_raw[:500]}") from exc

        if skip_review:
            return draft_json

        # ---- Stage 2: Review ----
        review_user_content: list = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": (
                f"上下文（Context）：\n{context}\n\n" if context else "无额外上下文。\n\n"
            ) + f"初版JSON输出（Draft JSON）：\n{json.dumps(draft_json, ensure_ascii=False, indent=2)}"},
        ]

        attempt = 0
        last_exc = None
        while True:
            attempt += 1
            try:
                review_raw = self._call_api(system_prompt=FINAL_SYSTEM_PROMPT, user_content=review_user_content)
                review_raw = self._extract_json(review_raw)
                review_json = json.loads(review_raw)
                corrected = review_json.get("final_corrected_json")
                if corrected is None:
                    raise ValueError("Review JSON missing 'final_corrected_json' field")
                return corrected
            except Exception as exc:
                last_exc = exc
                wait = min(2.0 * (2 ** (attempt - 1)), 300.0)
                log_info(f"Stage 2 attempt {attempt} failed, retrying in {wait:.0f}s...")
                time.sleep(wait)


# =============================================================================
# 特征提取器
# =============================================================================

class FeatureExtractor:
    def __init__(self, skip_review: bool = False):
        self._vl_client = VLInstructClient()
        self._skip_review = skip_review

    @staticmethod
    def extract_phash(image: Image.Image) -> np.ndarray:
        log_start_timer("phash")
        h = imagehash.phash(image, hash_size=8)
        log_info(f"pHash 完成, {round(log_stop_timer('phash')*1000)}ms")
        return np.packbits(h.hash.flatten()).astype(np.uint8)

    def analyze_image(self, image: Image.Image, context: str = "") -> Tuple[str, str]:
        data_url = VLInstructClient._pil_to_data_url(image)
        log_start_timer("vl_instruct")
        log_info("调用 VL-Instruct 深度解读...")
        result = self._vl_client.interpret_image(data_url, context=context,
                                                  skip_review=self._skip_review)
        vl_ms = round(log_stop_timer("vl_instruct") * 1000)
        deep_text = result.get("deep_semantics") or ""
        keywords = result.get("search_keywords") or []
        keyword_text = " ".join(str(k) for k in keywords)
        log_info(f"VL-Instruct 完成, {vl_ms}ms, deep_text={len(deep_text)}chars, keywords={len(keywords)}")
        return deep_text, keyword_text

    def get_visual_embedding(self, image: Image.Image) -> np.ndarray:
        data_url = VLInstructClient._pil_to_data_url(image)
        log_start_timer("visual_embed")
        vec = embed_image_url(url=data_url)
        log_info(f"VL-Embed 完成, {round(log_stop_timer('visual_embed')*1000)}ms, dim={len(vec)}")
        return vec

    def get_text_embedding(self, text: str) -> np.ndarray:
        log_start_timer("text_embed")
        vec = embed_text(text=text)
        log_info(f"Text-Embed 完成, {round(log_stop_timer('text_embed')*1000)}ms, dim={len(vec)}")
        return vec

    def get_visual_text_embedding(self, image: Image.Image, text: str) -> np.ndarray:
        data_url = VLInstructClient._pil_to_data_url(image)
        log_start_timer("mixed_embed")
        vec = embed_image_text(url=data_url, text=text)
        log_info(f"VL-Embed(mixed) 完成, {round(log_stop_timer('mixed_embed')*1000)}ms, dim={len(vec)}")
        return vec

    def process_image_pipeline(self, image_path: str,
                               enabled_paths: set = None,
                               context: str = "") -> Dict:
        if enabled_paths is None:
            enabled_paths = {"phash", "visual", "text", "mixed", "sparse"}

        log_info(f"开始流水线, 启用路径: {sorted(enabled_paths)}")
        img = Image.open(image_path).convert("RGB")
        log_info(f"图片已加载, size={img.size}")

        need_phash = "phash" in enabled_paths
        need_visual_embed = "visual" in enabled_paths
        need_analyze = bool({"text", "mixed", "sparse"} & enabled_paths)
        need_text_embed = "text" in enabled_paths
        need_mixed_embed = "mixed" in enabled_paths
        need_combined_text = "sparse" in enabled_paths

        result = {}
        skipped = []

        if need_phash:
            result["p_hash"] = self.extract_phash(img)
        else:
            skipped.append("pHash")

        if need_visual_embed:
            result["visual_embed"] = self.get_visual_embedding(img)
        else:
            skipped.append("visual_embed")

        if not need_analyze:
            skipped.append("VL-Instruct(+text/mixed/sparse)")

        if skipped:
            log_info(f"跳过: {', '.join(skipped)}")

        if need_analyze:
            deep_text, keyword_text = self.analyze_image(img, context=context)
            combined_text = f"{deep_text}\n{keyword_text}"
            if need_text_embed:
                result["text_embed"] = self.get_text_embedding(deep_text)
            if need_mixed_embed:
                result["visual_text_embed"] = self.get_visual_text_embedding(img, deep_text)
            if need_combined_text:
                result["text"] = combined_text

        return result


# =============================================================================
# Milvus 集合管理
# =============================================================================

def init_milvus_collection(drop_exists: bool = True) -> Collection:
    _host = MILVUS_HOST.replace("http://", "").replace("https://", "")
    _port = int(MILVUS_PORT)
    connections.connect("default", host=_host, port=_port)

    if utility.has_collection(COLLECTION_NAME):
        if drop_exists:
            log_info(f"集合 '{COLLECTION_NAME}' 已存在，正在删除重建...")
            utility.drop_collection(COLLECTION_NAME)
        else:
            return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="p_hash", dtype=DataType.BINARY_VECTOR, dim=DIM_PHASH),
        FieldSchema(name="visual_embed", dtype=DataType.FLOAT16_VECTOR, dim=DIM_VISUAL_EMBED),
        FieldSchema(name="text_embed", dtype=DataType.FLOAT16_VECTOR, dim=DIM_TEXT_EMBED),
        FieldSchema(name="visual_text_embed", dtype=DataType.FLOAT16_VECTOR, dim=DIM_VISUAL_TEXT_EMBED),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535,
                    enable_analyzer=True, analyzer_params={"type": "chinese"}),
        FieldSchema(name="text_sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="doc_meta", dtype=DataType.JSON),
        FieldSchema(name="img_meta", dtype=DataType.JSON),
    ]

    bm25_function = Function(
        name="text_to_bm25", function_type=FunctionType.BM25,
        input_field_names=["text"], output_field_names=["text_sparse_vec"],
    )

    schema = CollectionSchema(fields=fields, functions=[bm25_function], enable_dynamic_field=True)
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    indexes = [
        {"field": "p_hash", "type": "BIN_FLAT", "metric": "HAMMING"},
        {"field": "visual_embed", "type": "IVF_PQ", "metric": "IP",
         "params": {"nlist": IVFPQ_NLIST, "m": IVFPQ_M, "nbits": IVFPQ_NBITS}},
        {"field": "text_embed", "type": "IVF_PQ", "metric": "IP",
         "params": {"nlist": IVFPQ_NLIST, "m": IVFPQ_M, "nbits": IVFPQ_NBITS}},
        {"field": "visual_text_embed", "type": "IVF_PQ", "metric": "IP",
         "params": {"nlist": IVFPQ_NLIST, "m": IVFPQ_M, "nbits": IVFPQ_NBITS}},
        {"field": "text_sparse_vec", "type": "SPARSE_INVERTED_INDEX", "metric": "BM25"},
    ]
    for idx in indexes:
        collection.create_index(
            field_name=idx["field"],
            index_params={"index_type": idx["type"], "metric_type": idx["metric"],
                          "params": idx.get("params", {})},
        )

    # JSON 字段倒排索引
    json_indexes = [
        {"field_name": "doc_meta", "index_name": "doc_meta_paper_id_idx",
         "params": {"json_path": 'doc_meta["paper_id"]', "json_cast_type": "varchar"}},
        {"field_name": "img_meta", "index_name": "img_meta_page_idx",
         "params": {"json_path": 'img_meta["page"]', "json_cast_type": "double"}},
    ]
    for idx in json_indexes:
        collection.create_index(
            field_name=idx["field_name"], index_name=idx["index_name"],
            index_params={"index_type": "INVERTED", "params": idx["params"]},
        )
        log_info(f"JSON 索引: {idx['params']['json_path']}")

    collection.load()
    log_info(f"集合 '{COLLECTION_NAME}' 初始化完成并已加载!")
    return collection


# =============================================================================
# 辅助函数
# =============================================================================

def _get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def parse_image_filename(filename: str) -> dict:
    m = FILENAME_PATTERN.match(filename)
    if m:
        return {
            "bbox": [float(m.group(1)), float(m.group(2)),
                     float(m.group(3)), float(m.group(4))],
            "page": int(m.group(5)),
        }
    return {"page": None, "bbox": None}


def split_paragraphs(text: str) -> list[str]:
    parts = PARAGRAPH_SPLIT_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


def is_image_paragraph(paragraph: str) -> bool:
    lines = paragraph.splitlines()
    non_empty = [ln.strip() for ln in lines if ln.strip()]
    if not non_empty:
        return False
    return all(IMAGE_REF_PATTERN.fullmatch(ln) for ln in non_empty)


def extract_image_filename(paragraph: str) -> str | None:
    m = IMAGE_REF_PATTERN.fullmatch(paragraph.strip())
    return m.group(1) if m else None


LOCAL_IP = _get_local_ip()
IMAGE_BASE_URL = f"http://{LOCAL_IP}:{HTTP_PORT}"


def make_image_url(paper_id: str, img_filename: str) -> str:
    return f"{IMAGE_BASE_URL}/{DATA_DIR_NAME}/{paper_id}/figures/{img_filename}"


def extract_images_from_mmd(mmd_path: Path, paper_id: str) -> list[dict]:
    with open(mmd_path, "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = split_paragraphs(content)
    image_indices = [i for i, p in enumerate(paragraphs) if is_image_paragraph(p)]
    if not image_indices:
        return []

    figures_dir = mmd_path.parent / "figures"
    images: list[dict] = []

    for img_idx in image_indices:
        img_filename = extract_image_filename(paragraphs[img_idx])
        if not img_filename:
            continue
        local_path = figures_dir / img_filename
        if not local_path.exists():
            continue

        pre_texts = []
        cursor = img_idx - 1
        while cursor >= 0 and len(pre_texts) < 2:
            if is_image_paragraph(paragraphs[cursor]):
                break
            pre_texts.append(paragraphs[cursor])
            cursor -= 1
        pre_texts.reverse()

        post_texts = []
        cursor = img_idx + 1
        while cursor < len(paragraphs) and len(post_texts) < 2:
            if is_image_paragraph(paragraphs[cursor]):
                break
            post_texts.append(paragraphs[cursor])
            cursor += 1

        context_parts = pre_texts + post_texts
        context = "\n\n".join(context_parts)

        images.append({
            "filename": img_filename,
            "url": make_image_url(paper_id, img_filename),
            "context": context,
            "local_path": str(local_path),
        })

    return images


# =============================================================================
# HTTP 服务管理
# =============================================================================

def start_http_server(timeout: int = 15) -> subprocess.Popen:
    project_root = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.Popen(
        [sys.executable, "-u", "image_server.py"],
        cwd=project_root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    test_url = f"http://127.0.0.1:{HTTP_PORT}/"
    start_time = time.time()
    while time.time() - start_time < timeout:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(f"HTTP server exited early (code {proc.returncode})")
        try:
            resp = requests.get(test_url, timeout=2)
            if resp.status_code in (200, 404):
                print(f"[+] HTTP server ready at {test_url} (PID {proc.pid})")
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"HTTP server did not become ready within {timeout}s")


def stop_http_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    print(f"[*] Stopping HTTP server (PID {proc.pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    print("[+] HTTP server stopped.")


# =============================================================================
# 进度指示器
# =============================================================================

class ProgressTracker:
    def __init__(self, total: int, bar_width: int = 40):
        self.total = total
        self.bar_width = bar_width
        self.completed = 0
        self.success = 0
        self.failed = 0
        self.start_time = time.time()
        self._lock = threading.Lock()

    def update(self, success: bool = True) -> None:
        with self._lock:
            self.completed += 1
            if success: self.success += 1
            else: self.failed += 1
            self._print()

    def _print(self) -> None:
        pct = self.completed / self.total if self.total > 0 else 0
        filled = int(self.bar_width * pct)
        bar = "█" * filled + "░" * (self.bar_width - filled)
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta_sec = (self.total - self.completed) / rate if rate > 0 else 0

        def fmt(t: float) -> str:
            m, s = divmod(int(t), 60)
            h, m = divmod(m, 60)
            if h > 0: return f"{h}h{m:02d}m{s:02d}s"
            elif m > 0: return f"{m}m{s:02d}s"
            else: return f"{s}s"

        line = (f"\r  [{bar}] {self.completed}/{self.total} "
                f"({pct*100:.1f}%) ✓{self.success} ✗{self.failed} "
                f"| {rate:.1f} img/s | elapsed {fmt(elapsed)} | ETA {fmt(eta_sec)}   ")
        print(line, end="", flush=True)
        if self.completed >= self.total:
            print()


# =============================================================================
# 单张图片处理
# =============================================================================

def process_single_image(img_info: dict, paper_id: str,
                         extractor: FeatureExtractor) -> dict | None:
    img_filename = img_info["filename"]
    context = img_info["context"]
    local_path = img_info["local_path"]
    img_id = f"{paper_id}_{img_filename}"

    try:
        features = extractor.process_image_pipeline(local_path, context=context)
        img_meta = parse_image_filename(img_filename)
        data_row = {
            "img_id": img_id,
            "p_hash": features.get("p_hash"),
            "visual_embed": features.get("visual_embed"),
            "text_embed": features.get("text_embed"),
            "visual_text_embed": features.get("visual_text_embed"),
            "text": features.get("text", ""),
            "doc_meta": {"paper_id": paper_id},
            "img_meta": img_meta,
        }
        return data_row
    except Exception as exc:
        print(f"      [-] Error processing {img_id}: {exc}")
        return None


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="大规模数据摄入 —— 直接从 data_* 目录读入 Milvus (自给自足版)"
    )
    parser.add_argument("--data-dir", default=DATA_DIR_NAME,
                        help=f"数据目录名 (默认: {DATA_DIR_NAME})")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Milvus 批量插入大小")
    parser.add_argument("--limit", type=int, default=0,
                        help="最大摄入图片数 (0=不限制)")
    parser.add_argument("--workers", type=int, default=20,
                        help="并行工作线程数")
    args = parser.parse_args()

    log_setup(log_dir="logs")

    # 1. 启动 HTTP 图片服务
    http_proc = start_http_server()

    try:
        # 2. 初始化 Milvus
        collection = init_milvus_collection(drop_exists=True)

        # 3. 扫描论文目录
        data_dir = Path(args.data_dir)
        paper_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
        total_papers = len(paper_dirs)
        print(f"[*] Found {total_papers} paper directories.")

        all_tasks: list[tuple[dict, str]] = []
        for paper_dir in paper_dirs:
            paper_id = paper_dir.name
            mmd_files = list(paper_dir.glob("*.mmd"))
            if not mmd_files:
                continue
            try:
                img_infos = extract_images_from_mmd(mmd_files[0], paper_id)
            except Exception as exc:
                print(f"  [-] Failed to parse {paper_id}: {exc}")
                continue
            for img_info in img_infos:
                all_tasks.append((img_info, paper_id))

        total_images = len(all_tasks)
        print(f"[*] Total images found: {total_images}")

        if args.limit > 0:
            all_tasks = all_tasks[:args.limit]
            print(f"[*] Limited to {args.limit} images.")

        # 4. 多线程并行处理
        print(f"\n[*] Starting processing with {args.workers} worker threads...\n")
        progress = ProgressTracker(total=len(all_tasks))
        # 摄入端使用完整两阶段审查
        extractor = FeatureExtractor(skip_review=False)
        batch: list[dict] = []
        batch_lock = threading.Lock()

        def flush_batch(force: bool = False) -> None:
            nonlocal batch
            if not batch:
                return
            if force or len(batch) >= args.batch_size:
                collection.insert(batch)
                batch = []

        def on_result(future, img_info, paper_id) -> None:
            try:
                data_row = future.result()
                with batch_lock:
                    if data_row is not None:
                        batch.append(data_row)
                        progress.update(success=True)
                        flush_batch(force=False)
                    else:
                        progress.update(success=False)
            except Exception:
                progress.update(success=False)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for img_info, paper_id in all_tasks:
                future = executor.submit(
                    process_single_image, img_info=img_info,
                    paper_id=paper_id, extractor=extractor,
                )
                future.add_done_callback(
                    lambda f, info=img_info, pid=paper_id: on_result(f, info, pid)
                )
                futures.append(future)
            for future in futures:
                future.result()

        # 5. 插入剩余批次
        with batch_lock:
            flush_batch(force=True)

        collection.flush()
        print("\n" + "=" * 56)
        print("  Ingestion Summary")
        print("=" * 56)
        print(f"  Papers scanned     : {total_papers}")
        print(f"  Images found       : {total_images}")
        print(f"  Success            : {progress.success}")
        print(f"  Failed             : {progress.failed}")
        print(f"  Worker threads     : {args.workers}")
        print("=" * 56)

    finally:
        stop_http_server(http_proc)


if __name__ == "__main__":
    main()
