from datetime import datetime, timezone
from typing import Dict, Any, Type, TypeVar, Optional
from pydantic import BaseModel
from app.config import settings
from app.contracts.docx_contract import (
    DOCUMENT_SCALAR_FIELD_NAME_SET,
    candidate_relation_scalar_field_names,
    normalize_document_id,
)

T = TypeVar("T", bound=BaseModel)

# 特殊字段映射表，处理 Document 模型中与 JSON 洗后的结构命名不一致的情况
FIELD_MAPPING = {
    "ID": ["ID", "sysId", "docId", "doc_id", "id"],
    "docId": ["docId", "doc_id", "sysId", "ID", "id"],
    "subDocType": ["subDocType", "sysDataType"],
    "articleId": ["articleId"],
    "recordUpdateTime": ["sysDataUpdateTime"],
    "recordCreateTime": ["sysDataCreateTime"],
    "publishDate": ["year"],
    "sourceData": ["sourceData"]
}


def _cast_value(val: Any, type_str: str) -> Any:
    if val is None:
        return None
    t = (type_str or "").strip().lower()
    if t in ("int",):
        return int(val)
    elif t in ("double", "float"):
        return float(val)
    elif t == "string":
        return str(val)
    elif t == "date":
        try:
            if isinstance(val, (int, float)):
                if val > 1e12:
                    val = val / 1000.0
                return datetime.fromtimestamp(val, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(val))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
        except (ValueError, TypeError, OSError):
            return str(val)
    else:
        return val

def preprocess_kafka_item(raw_item: dict) -> dict:
    """
    第一步：递归扁平化原始 Kafka 节点
    将复杂的 {"field": "...", "value": [...]} 以及 dp, op 层级清理为干净的数据字典
    """
    flat = {}
    
    if "id" in raw_item:
        flat["id"] = raw_item["id"]
    if "ID" in raw_item:
        flat["ID"] = raw_item["ID"]
    if "sysId" in raw_item:
        flat["sysId"] = raw_item["sysId"]
    if "docId" in raw_item:
        flat["docId"] = raw_item["docId"]
    if "doc_id" in raw_item:
        flat["doc_id"] = raw_item["doc_id"]
        
    if "cls" in raw_item:
        flat["cls"] = raw_item["cls"]

    # 1. 扁平化 dp（跳过配置中指定忽略的字段）
    dp_ignore = settings.dp_ignore_set
    dp = raw_item.get("dp", {})
    for field_name, field_obj in dp.items():
        if field_name in dp_ignore:
            continue
        if isinstance(field_obj, dict) and "value" in field_obj:
            values = field_obj["value"]
            multi_value = field_obj.get("multiValue", False)
            type_str = field_obj.get("type", "string")

            if multi_value:
                flat[field_name] = [_cast_value(v, type_str) for v in values]
            else:
                flat[field_name] = _cast_value(values[0], type_str) if values else None
        else:
            flat[field_name] = field_obj

    # 2. 扁平化 op
    op = raw_item.get("op") or {}
    for rel_name, rel_obj in op.items():
        if isinstance(rel_obj, dict) and "value" in rel_obj:
            rel_list = rel_obj["value"]
        elif isinstance(rel_obj, list):
            rel_list = rel_obj
        else:
            rel_list = [rel_obj]
            
        flat_rel_list = []
        for nested_item in rel_list:
            if isinstance(nested_item, dict):
                nested_flat = preprocess_kafka_item(nested_item)
                flat_rel_list.append(nested_flat)
                for child_key, child_value in nested_flat.items():
                    for scalar_field in candidate_relation_scalar_field_names(rel_name, child_key):
                        _append_relation_scalar(flat, scalar_field, child_value)
            else:
                flat_rel_list.append(nested_item)
        flat[rel_name] = flat_rel_list

    document_id = normalize_document_id(flat)
    if document_id:
        flat["ID"] = document_id

    return flat


def map_to_document(model_cls: Type[T], raw_data: dict) -> T:
    """
    第二步：基于清洗后的标准数据结构，动态将其映射到指定的 Pydantic 模型
    """
    if getattr(model_cls, "__name__", "") == "Document":
        data_list = raw_data.get("data", [])
        parsed_articles = []
        
        article_cls = None
        for name, field in model_cls.model_fields.items():
            if name == "data":
                try:
                    import typing
                    article_cls = typing.get_args(field.annotation)[0]
                except Exception:
                    from app.models.base.article import Article
                    article_cls = Article
                break

        for item in data_list:
            clean_item = preprocess_kafka_item(item)
            clean_item["sourceData"] = item
            
            article_kwargs = _map_fields_to_model(article_cls, clean_item)
            parsed_articles.append(article_cls(**article_kwargs))
            
        doc_kwargs = {
            "success": raw_data.get("success"),
            "subDocType": raw_data.get("subDocType"),
            "dataType": raw_data.get("dataType"),
            "data": parsed_articles
        }
        return model_cls(**doc_kwargs)
    else:
        clean_data = preprocess_kafka_item(raw_data)
        kwargs = _map_fields_to_model(model_cls, clean_data)
        return model_cls(**kwargs)


def _append_relation_scalar(flat: dict, field_name: str, value: Any) -> None:
    if value is None:
        return
    values = flat.setdefault(field_name, [])
    if isinstance(value, list):
        values.extend(item for item in value if item is not None)
    else:
        values.append(value)

def _map_fields_to_model(model_cls: Type[BaseModel], clean_data: dict) -> dict:
    """Map cleaned Kafka data into model kwargs."""
    kwargs = {}
    is_article_model = getattr(model_cls, "__name__", "") == "Article"
    
    used_keys = set()
    
    for field_name, field_info in model_cls.model_fields.items():
        if field_name == "dynamic_fields" or (is_article_model and field_name == "id"):
            continue
            
        val = None
        sources = FIELD_MAPPING.get(field_name, [field_name])
        for src in sources:
            if src in clean_data:
                val = clean_data[src]
                used_keys.add(src)
                break
        
        if val is not None and "datetime" in str(field_info.annotation):
            try:
                val = datetime.strptime(str(val), "%Y-%m-%dT%H:%M:%S.%f%z")
            except ValueError:
                try:
                    val = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                except ValueError:
                    pass
        elif val is not None and field_info.annotation == Optional[str]:
            val = str(val)
            
        if val is not None:
            import typing
            origin_type = typing.get_origin(field_info.annotation)
            args_type = typing.get_args(field_info.annotation)
            
            is_list_field = origin_type is list or (origin_type is typing.Union and list in [typing.get_origin(arg) or arg for arg in args_type])
            
            if is_list_field and not isinstance(val, list):
                val = [val]
            elif not is_list_field and isinstance(val, list):
                val = val[0] if len(val) > 0 else None
                
        kwargs[field_name] = val

    # Article keeps known contract scalars as top-level fields for Milvus.
    if is_article_model:
        document_id = normalize_document_id(clean_data)
        if document_id:
            kwargs["ID"] = document_id
        for key, value in clean_data.items():
            if key in DOCUMENT_SCALAR_FIELD_NAME_SET and key not in kwargs:
                kwargs[key] = value
                used_keys.add(key)
            
    if "dynamic_fields" in model_cls.model_fields:
        kwargs["dynamic_fields"] = {}
        
    return kwargs
