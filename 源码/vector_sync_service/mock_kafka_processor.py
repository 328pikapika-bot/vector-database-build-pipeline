import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))

from app.models.document import Document
from app.utils.parser import map_to_document
from app.services.milvus_service import get_milvus_service
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "scientific_documents_v7_test"

JSON_FILE = os.path.join(os.path.dirname(__file__), "..", "sample_json", "sample.json")


def load_json_message():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

 
def run():
    raw_message = load_json_message()
    logger.info("=== 加载 %s ===", JSON_FILE)
    logger.info("data 数组长度: %d", len(raw_message.get("data", [])))

    logger.info("=== 解析 Kafka 消息 ===")
    logger.info("过滤配置: dp_ignore=%s, json_force=%s",
                 settings.dp_ignore_set, settings.dynamic_json_set)
    document = map_to_document(Document, raw_message)

    for i, article in enumerate(document.data):
        logger.info(f"--- Article[{i}] ---")
        logger.info(f"  sysId        = {article.sysId}")
        logger.info(f"  title        = {article.title}")
        logger.info(f"  year         = {article.year}")
        logger.info(f"  subDocType   = {article.subDocType}")
        logger.info(f"  keyword      = {article.keyword}")
        logger.info(f"  dynamic_fields = {len(article.dynamic_fields)} 个字段")

    milvus_service = get_milvus_service()
    VECTOR_DIM = settings.VECTOR_DIM

    logger.info("========== 逐字段动态展开入库 (集合: %s) ==========", COLLECTION_NAME)
    try:
        if milvus_service.client.has_collection(COLLECTION_NAME):
            milvus_service.client.drop_collection(COLLECTION_NAME)

        milvus_service.init_collection(COLLECTION_NAME, VECTOR_DIM)

        res = milvus_service.process_and_insert(COLLECTION_NAME, document.data, vector_dim=VECTOR_DIM)
        logger.info("入库完成: %s", res)

        desc = milvus_service.client.describe_collection(COLLECTION_NAME)
        logger.info("最终字段数 = %d", len(desc["fields"]))
        logger.info("最终字段列表: %s", [f["name"] for f in desc["fields"]])
    except Exception as e:
        logger.error("入库失败: %s", e, exc_info=True)

    logger.info("========== 全部完成 ==========")


if __name__ == "__main__":
    run()
