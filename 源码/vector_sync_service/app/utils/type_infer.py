from datetime import datetime
from pymilvus import DataType
from typing import Any, Dict

def infer_milvus_type(value: Any) -> Dict[str, Any]:
    """
    根据 Python 数据类型推断 Milvus DataType。
    返回一个包含 data_type 和额外参数（如 max_length）的字典，可直接用于 add_collection_field。
    """
    if value is None:
        return {"data_type": DataType.VARCHAR, "max_length": 65535}
        
    if isinstance(value, datetime):
        return {"data_type": DataType.VARCHAR, "max_length": 64}
    elif isinstance(value, bool):
        return {"data_type": DataType.BOOL}
    elif isinstance(value, int):
        return {"data_type": DataType.INT64}
    elif isinstance(value, float):
        return {"data_type": DataType.DOUBLE}
    elif isinstance(value, str):
        return {"data_type": DataType.VARCHAR, "max_length": 65535}
    elif isinstance(value, list):
        if not value:
            return {"data_type": DataType.JSON}
        sample = value[0]
        if isinstance(sample, (dict, list)):
            return {"data_type": DataType.JSON}
        elem_type_info = infer_milvus_type(sample)
        result = {
            "data_type": DataType.ARRAY,
            "element_type": elem_type_info["data_type"],
            "max_capacity": 4096,
        }
        if elem_type_info["data_type"] == DataType.VARCHAR:
            result["max_length"] = elem_type_info.get("max_length", 65535)
        return result
    elif isinstance(value, dict):
        return {"data_type": DataType.JSON}
    else:
        # fallback
        return {"data_type": DataType.VARCHAR, "max_length": 65535}
