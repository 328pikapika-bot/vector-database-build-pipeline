#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/21
@Author  : st
@File    : import_qa_process.py
@Project : Smart_Search
@Software: PyCharm
@文件说明:
问答对数据导入Milvus数据库
基于vllm部署的qwen3-embedding-0.6B模型进行向量化
"""
import datetime
import json
import os
import pandas as pd
import requests
import time
import traceback
from pymilvus import DataType
from pymilvus import connections, CollectionSchema, FieldSchema, Collection, utility, db

# 数据库配置
MILVUS_DB = {
    "host": "127.0.0.1",
    "port": "19530",
    "user": "ninemax",
    "password": os.getenv("MILVUS_PASSWORD", ""),
    "db_name": "ckzx_db",
    "secure": False,
}

# 问答对集合配置
QA_COLLECTION_INFO = {
    "name": "qa_collection",
    "num_partitions": 4,  # 针对4000条数据，减少分区数以提高性能
    "fields": [
        {
            "name": 'id',
            "dtype": DataType.VARCHAR,
            "description": "主键ID",
            "max_length": 64,
            "is_primary": True,
            "auto_id": False
        },
        {
            "name": 'vector',
            "description": "问题向量值",
            "dtype": DataType.FLOAT_VECTOR,
            "dim": 2560,  # qwen3-embedding-0.6B模型输出维度为1024
            "index_params": {
                "metric_type": "L2",  # 使用L2距离度量
                "index_type": "IVF_PQ",
                "nlist": 32,
                "m": 64,
                "nbits": 8,
            }
        },
        {
            "name": 'question',
            "description": "问题",
            "dtype": DataType.VARCHAR,
            "max_length": 4096
        },
        {
            "name": 'answer',
            "description": "答案",
            "dtype": DataType.VARCHAR,
            "max_length": 16384
        },
        {
            "name": 'class_name',
            "description": "答案",
            "dtype": DataType.VARCHAR,
            "max_length": 16384
        }
    ]
}
# VLLM服务配置 (请根据实际部署修改)
VLLM_CONFIG = {
    "base_url": "http://127.0.0.1:8080/v1",  # vllm服务地址
    "api_key": "your-api-key",  # 如有需要
    "model": "medium",
    "embedding_dim": 2560
}

# 连接数据库
conn = connections.connect(
    user=MILVUS_DB['user'],
    password=MILVUS_DB['password'],
    host=MILVUS_DB['host'],
    port=MILVUS_DB['port'],
    alias="default"
)

# 创建数据库
if not db.list_database().__contains__(MILVUS_DB['db_name']):
    db.create_database(MILVUS_DB['db_name'])

# 使用数据库
db.using_database(MILVUS_DB['db_name'])

CONSISTENCY_LEVEL = "Bounded"


def qwen3_embedding(texts, batch_size=32):
    """
    使用vllm部署的qwen3-embedding-0.6B模型生成向量
    :param texts: 文本列表
    :param batch_size: 批处理大小
    :return: 向量列表
    """
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        try:
            # 调用vllm服务的embedding接口
            response = requests.post(
                f"{VLLM_CONFIG['base_url']}/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {VLLM_CONFIG['api_key']}" if VLLM_CONFIG['api_key'] else ""
                },
                json={
                    "model": VLLM_CONFIG['model'],
                    "input": batch_texts
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                batch_embeddings = [item['embedding'] for item in result['data']]
                embeddings.extend(batch_embeddings)
            else:
                print(f"向量化请求失败: {response.status_code}, {response.text}")
                # 如果失败，返回零向量
                zero_vector = [0.0] * VLLM_CONFIG['embedding_dim']
                embeddings.extend([zero_vector] * len(batch_texts))

        except Exception as e:
            print(f"向量化异常: {e}")
            # 如果异常，返回零向量
            zero_vector = [0.0] * VLLM_CONFIG['embedding_dim']
            embeddings.extend([zero_vector] * len(batch_texts))

    return embeddings


def read_qa_excel(file_path):
    """
    读取问答对Excel文件
    :param file_path: Excel文件路径
    :return: 问答对数据列表
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 检查必要字段id	question	answer	class_name
        required_columns = ['id', 'question', 'answer', 'class_name']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Excel文件缺少必要字段: {missing_columns}")

        # 转换为字典列表
        qa_data = df.to_dict('records')

        print(f"成功读取 {len(qa_data)} 条问答对数据")
        return qa_data

    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        raise


def process_qa_data(qa_data, batch_size=50):
    """
    处理问答对数据，生成向量
    :param qa_data: 问答对数据
    :param batch_size: 批处理大小
    :return: 处理后的数据列表
    """
    processed_data = []

    # 分批处理
    for i in range(0, len(qa_data), batch_size):
        batch_data = qa_data[i:i + batch_size]

        # 提取问题文本
        questions = [item['question'] for item in batch_data]

        # 生成向量
        print(f"正在生成第 {i // batch_size + 1} 批数据的向量...")
        vectors = qwen3_embedding(questions, batch_size=min(batch_size, 32))

        # 组合数据
        for j, item in enumerate(batch_data):
            processed_item = {
                'id': str(item['id']),
                'vector': vectors[j] if j < len(vectors) else [0.0] * VLLM_CONFIG['embedding_dim'],
                'question': str(item['question']),
                'answer': str(item['answer'])
            }
            processed_data.append(processed_item)

        print(f"已完成第 {i // batch_size + 1} 批数据处理 ({len(batch_data)} 条)")

    return processed_data


def create_qa_collection(collection_name, fields_list, num_partitions=16):
    """
    创建问答对集合
    :param collection_name: 集合名称
    :param fields_list: 字段信息
    :param num_partitions: 分区数
    :return: 是否创建成功
    """
    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在")
        return True

    fields = []
    index_params = []

    for field in fields_list:
        if field['dtype'] == DataType.FLOAT_VECTOR:
            # 向量字段
            temp_params = FieldSchema(
                name=field['name'],
                dtype=field['dtype'],
                description=field['description'],
                dim=field['dim']
            )
        elif field.__contains__('max_length'):
            # 字符串字段
            temp_params = FieldSchema(
                name=field['name'],
                dtype=field['dtype'],
                description=field['description'],
                max_length=field['max_length']
            )
        else:
            # 其他字段
            temp_params = FieldSchema(
                name=field['name'],
                dtype=field['dtype'],
                description=field['description']
            )

        if field.__contains__('is_primary'):
            temp_params.is_primary = field['is_primary']
            temp_params.auto_id = field['auto_id']

        fields.append(temp_params)

        # 处理索引参数
        if field.__contains__('index_params'):
            if field['index_params']:
                temp_params = {
                    "metric_type": field['index_params']['metric_type'],
                    "index_type": field['index_params']['index_type'],
                    "params": {
                        "nlist": field['index_params']['nlist'],
                        "m": field['index_params'].get('m', 64),
                        "nbits": field['index_params'].get('nbits', 8),
                    } if 'nlist' in field['index_params'] else field['index_params'].get('params', {})
                }
                index_params.append((field['name'], 'index_' + field['name'], temp_params))

    schema = CollectionSchema(
        fields=fields,
        description="问答对检索集合"
    )

    # 创建集合
    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default',
        shards_num=2,
        consistency_level=CONSISTENCY_LEVEL
    )

    # 创建索引
    for field_name, index_name, index_param in index_params:
        collection.create_index(
            field_name=field_name,
            index_params=index_param,
            index_name=index_name
        )

    print(f"集合 {collection_name} 创建成功")
    return True


def insert_qa_datas(datas, collection_name, batch_size=1000):
    """
    插入问答对数据
    :param datas: 数据列表
    :param collection_name: 集合名称
    :param batch_size: 批处理大小
    :return: 成功插入数量
    """
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return 0

    total_inserted = 0

    # 分批插入
    for i in range(0, len(datas), batch_size):
        batch_data = datas[i:i + batch_size]

        try:
            collection = Collection(collection_name)
            res = collection.insert(batch_data)
            inserted_count = res.insert_count
            total_inserted += inserted_count

            print(f"批次 {i // batch_size + 1}: 成功插入 {inserted_count} 条数据")

        except Exception as e:
            print(f"批次 {i // batch_size + 1} 插入失败: {e}")
            continue

    return total_inserted


def print_log(message):
    """
    打印日志
    :param message: 日志消息
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{current_time}] {message}"
    print(log_message)

    # 写入日志文件
    with open("./qa_import_log.txt", "a+", encoding="utf-8") as f:
        f.write(log_message + "\n")


def import_qa_data(excel_file_path):
    """
    主函数：导入问答对数据
    :param excel_file_path: Excel文件路径
    """
    start_time = time.time()
    print_log("开始导入问答对数据")

    try:
        # 1. 读取Excel数据
        print_log("步骤1: 读取Excel文件")
        qa_data = read_qa_excel(excel_file_path)

        # 2. 创建集合
        print_log("步骤2: 创建Milvus集合")
        create_qa_collection(
            QA_COLLECTION_INFO['name'],
            QA_COLLECTION_INFO['fields'],
            QA_COLLECTION_INFO['num_partitions']
        )

        # 3. 处理数据并生成向量
        print_log("步骤3: 处理数据并生成向量")
        processed_data = process_qa_data(qa_data)

        # 4. 插入数据
        print_log("步骤4: 插入数据到Milvus")
        inserted_count = insert_qa_datas(processed_data, QA_COLLECTION_INFO['name'])

        # 5. 统计结果
        end_time = time.time()
        duration = end_time - start_time

        print_log(f"导入完成! 总数据量: {len(qa_data)}, 成功插入: {inserted_count}")
        print_log(f"总耗时: {duration:.2f} 秒")

        if inserted_count < len(qa_data):
            print_log(f"警告: 有 {len(qa_data) - inserted_count} 条数据插入失败")

    except Exception as e:
        print_log(f"导入过程出现异常: {e}")
        print_log(traceback.format_exc())
        raise


if __name__ == '__main__':
    # 使用示例
    excel_file = "参考咨询.xlsx"  # 请修改为实际的Excel文件路径

    # 检查文件是否存在
    if not os.path.exists(excel_file):
        print(f"错误: Excel文件 {excel_file} 不存在")
        print("请确保文件存在或修改excel_file变量为正确的文件路径")
    else:
        import_qa_data(excel_file)
