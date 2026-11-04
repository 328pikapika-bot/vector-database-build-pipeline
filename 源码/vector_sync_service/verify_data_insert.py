import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))

from app.models.document import Document
from app.utils.parser import map_to_document
from app.services.milvus_service import get_milvus_service, close_milvus_service
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("verify")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COLLECTION = settings.MILVUS_COLLECTION
VECTOR_DIM = settings.VECTOR_DIM

logger.info("Collection: %s, VectorDim: %d", COLLECTION, VECTOR_DIM)
logger.info("dp_ignore: %s", settings.dp_ignore_set)
logger.info("json_force: %s", settings.dynamic_json_set)

milvus = get_milvus_service()

if milvus.client.has_collection(COLLECTION):
    logger.info("Collection %s exists, dropping it for clean test ...", COLLECTION)
    milvus.client.drop_collection(COLLECTION)

milvus.init_collection(COLLECTION, VECTOR_DIM)

files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f.split(".")[0].isdigit()],
    key=lambda x: int(x.split(".")[0]),
)

logger.info("Found %d JSON files in %s", len(files), DATA_DIR)

success_count = 0
fail_count = 0
skip_count = 0

for filename in files:
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sub_doc_type = raw.get("subDocType")
    data_type = raw.get("dataType")
    data_list = raw.get("data", [])

    logger.info("--- %s ---", filename)
    logger.info("  subDocType=%s, dataType=%s, data数组长度=%d", sub_doc_type, data_type, len(data_list))

    try:
        document = map_to_document(Document, raw)
        articles = document.data or []

        if not articles:
            logger.warning("  → SKIP: 解析后无有效 Article")
            skip_count += 1
            continue

        for i, article in enumerate(articles):
            dyn_count = len(article.dynamic_fields) if article.dynamic_fields else 0
            logger.info(
                "  Article[%d]: sysId=%s, title_len=%d, abstract_len=%d, year=%s, sysSubType=%s, keyword=%s, dynamic_fields=%d",
                i,
                article.sysId,
                len(article.title) if article.title else 0,
                len(article.abstract[0]) if article.abstract else 0,
                article.year,
                article.sysSubType,
                article.keyword,
                dyn_count,
            )

        res = milvus.process_and_insert(COLLECTION, articles, vector_dim=VECTOR_DIM)
        logger.info("  → INSERT OK: %s records, insert_result=%s", len(articles), res)
        success_count += 1

    except Exception as e:
        logger.error("  → FAIL: %s: %s", type(e).__name__, str(e)[:300])
        fail_count += 1

desc = milvus.client.describe_collection(COLLECTION)
field_names = [f["name"] for f in desc["fields"]]
field_types = {f["name"]: f.get("type") or f.get("data_type") for f in desc["fields"]}
logger.info("Collection 最终字段数: %d", len(desc["fields"]))
for fn in field_names:
    ft = field_types[fn]
    type_name = ft
    if hasattr(ft, "name"):
        type_name = ft.name
    logger.info("  %-40s %s", fn, type_name)

row_count_res = milvus.client.query(
    collection_name=COLLECTION,
    filter="",
    output_fields=["sysId"],
    limit=100,
)
logger.info("Collection 当前总记录数: %d", len(row_count_res))

close_milvus_service()

logger.info("========== 验证完成 ==========")
logger.info("成功: %d, 跳过: %d, 失败: %d", success_count, skip_count, fail_count)
