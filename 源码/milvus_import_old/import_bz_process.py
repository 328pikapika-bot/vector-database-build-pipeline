#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/13 10:05
@Author  : st
@File    : import_process.py
@Project : Smart_Search
@Software: PyCharm
@文件说明:

"""
import os
import re

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch.types
from pymilvus import DataType
from pymilvus import connections, CollectionSchema, FieldSchema, Collection, utility, db
from sentence_transformers import SentenceTransformer
import datetime

# 数据库配置
MILVUS_DB = {
    "host": "127.0.0.1",
    # "host": "127.0.0.1",
    "port": "19530",
    "user": "ninemax",
    "password": os.getenv("MILVUS_PASSWORD", ""),
    "db_name": "bz_db",
    "secure": False,
}

COLLECTION_INFO = {
    "name": "bz_business",
    "num_partitions": 16,
    "fields": [
        {
            "name": 'id',
            "dtype": DataType.VARCHAR,
            "description": "主键ID,与源数据id对应",
            "max_length": 64,
            "is_primary": True,
            "auto_id": False
        },
        {
            "name": 'vec_name_range_preface',
            "description": "使用标准的中文名称、范围和前言来构建向量字段",
            "dtype": DataType.FLOAT_VECTOR,
            "dim": 768,
            "index_params": {
                "metric_type": "IP",
                "index_type": "IVF_PQ",
                "nlist": 1024,
                "m": 64,
                "nbits": 8
            }
        },
        {
            "name": 'name_zh',
            "description": "标准中文名称",
            "dtype": DataType.VARCHAR,
            "max_length": 1024
        },
        {
            "name": 'name_en',
            "description": "标准英文名称",
            "dtype": DataType.VARCHAR,
            "max_length": 65535
        },

        {
            "name": 'system_identifier',
            "description": "系统标识符",
            "dtype": DataType.VARCHAR,
            "max_length": 100,
            "index_params": {}
        },
        {
            "name": 'standard_level',
            "description": "标准等级",
            "dtype": DataType.VARCHAR,
            "max_length": 10
        },
        {
            "name": 'chinese_keywords',
            "description": "中文关键词",
            "dtype": DataType.VARCHAR,
            "max_length": 10000,
            "index_params": {}
        },
        {
            "name": 'english_keywords',
            "description": "英文关键词",
            "dtype": DataType.VARCHAR,
            "max_length": 10000,
            "index_params": {}
        },
        {
            "name": 'specified_content',
            "description": "规定内容",
            "dtype": DataType.VARCHAR,
            "max_length": 10000
        },

        {
            "name": 'applicable_scope',
            "description": "适用范围",
            "dtype": DataType.VARCHAR,
            "max_length": 10000
        },
        {
            "name": 'foreword',
            "description": "前言",
            "dtype": DataType.VARCHAR,
            "max_length": 10000
        },
        {
            "name": 'element_collection',
            "description": "元素集合",
            "dtype": DataType.VARCHAR,
            "max_length": 65535
        },
        {
            "name": 'element_group_collection',
            "description": "元素组集合",
            "dtype": DataType.VARCHAR,
            "max_length": 65535
        }
    ]
}

# 连接数据库
conn = connections.connect(user=MILVUS_DB['user'], password=MILVUS_DB['password'], host=MILVUS_DB['host'], port=MILVUS_DB['port'], alias="default")

if not db.list_database().__contains__(MILVUS_DB['db_name']):
    db.create_database(MILVUS_DB['db_name'])

db.using_database(MILVUS_DB['db_name'])

CONSISTENCY_LEVEL = "Bounded"

model = SentenceTransformer('./m3e-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text2vec(sentences, batch_size=64, normalize=True):
    embeddings = model.encode(sentences, batch_size=batch_size, device=device, normalize_embeddings=normalize)
    return embeddings


def find_info(pattern, str_info):
    return re.compile(rf"<{pattern}>(.*?)</{pattern}>", re.M).findall(str_info)


def find_sub_info(info_lists, name_lists):
    author_result = []
    for author in info_lists:
        temp = []
        for name in name_lists:
            temp.append("".join(find_info(name, author)))
        author_result.append("<+>".join(temp))
    return "<::>".join(author_result)


def get_time_str():
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return str(t)


def get_datas_api():
    """
    从api获取数据
    :return:
    """
    start_time = time.time()
    datas = pd.read_excel("标准融合表.xlsx")

    reverse_key_mapping = {
    }

    datas = datas.rename(columns=reverse_key_mapping, inplace=True)

    datas = datas.to_dict(orient="records")
    temp_list = []
    for data in datas:
        try:
            temp_dict = dict()
            temp_dict['id'] = data['id']
            vec_str = f"标准名称：{data['name_zh']}\n标准号:{data['system_identifier']}\n适用范围:{data['applicable_scope']}\n前言:{data['foreword']}"
            temp_dict['vec_name_range_preface'] = text2vec([vec_str])[0]
            temp_dict['name_zh'] = data['name_zh']
            temp_dict['name_en'] = data['name_en']
            temp_dict['system_identifier'] = data['system_identifier']
            temp_dict['standard_level'] = data['standard_level']
            temp_dict['chinese_keywords'] = data['chinese_keywords']
            temp_dict['english_keywords'] = data['english_keywords']
            temp_dict['specified_content'] = data['specified_content']
            temp_dict['applicable_scope'] = data['applicable_scope']
            temp_dict['foreword'] = data['foreword']
            temp_dict['element_collection'] = data['element_collection']
            temp_dict['element_group_collection'] = data['element_group_collection']
            temp_list.append(temp_dict)
        except Exception as e:
            print(e)
            continue
    return temp_list


def process_insert(datas):
    """
    数据导入任务
    :return:
    """

    # 创建集合
    res = create_collection(COLLECTION_INFO.get('name'), COLLECTION_INFO.get('fields'), COLLECTION_INFO.get('num_partitions'))
    if not res:
        raise "集合创建失败：" + COLLECTION_INFO.get('name')
    num_all = 0
    num_success = 0
    num = insert_datas(datas, COLLECTION_INFO.get('name'))
    num_success += num
    print('Added datas:' + str(num) + '; all datas: ' + str(num_all) + '; success: ' + str(num_success))


def start_insert_task():
    datas = get_datas_api()
    process_insert(datas)
    print('启动入库任务')


def create_collection(collection_name, fields_list, num_partitions=1024):
    """
    创建数据库集合
    :param collection_name: 集合名称
    :param fields_list: 字段信息
    :param num_partitions: 分区数
    :return:
    """
    if utility.has_collection(collection_name):
        return True
    fields = []
    index_params = []
    for field in fields_list:
        temp_params = FieldSchema(
            name=field['name'],
            dtype=field['dtype'],
            description=field['description']
        )

        if field.__contains__('dim'):
            temp_params = FieldSchema(
                name=field['name'],
                dtype=field['dtype'],
                description=field['description'],
                dim=field['dim'],
            )

        if field.__contains__('max_length'):
            temp_params = FieldSchema(
                name=field['name'],
                dtype=field['dtype'],
                description=field['description'],
                max_length=field['max_length'],
            )
        if field.__contains__('is_primary'):
            temp_params.is_primary = field['is_primary']
            temp_params.auto_id = field['auto_id']
        fields.append(temp_params)
        # 判断是否创建索引
        if field.__contains__('index_params'):
            # 向量索引
            if field['index_params']:
                temp_params = {
                    "metric_type": field['index_params']['metric_type'],  # 'IP'
                    "index_type": field['index_params']['index_type'],
                    "params": {
                        "nlist": field['index_params']['nlist'],
                        "m": field['index_params'].get('m', 64),
                        "nbits": field['index_params'].get('nbits', 8),
                    }
                }
            # 标量索引
            else:
                temp_params = None
            index_params.append((field['name'], 'index_' + field['name'], temp_params))

    schema = CollectionSchema(
        fields=fields,
        description="nstl serach"
    )
    # 创建集合
    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default',
        shards_num=2,  # 要创建的集合的分片数。	[1,16]
        # num_partitions=num_partitions,  # 要创建的集合的逻辑分区数。	[1,4096]
        consistency_level=CONSISTENCY_LEVEL
    )
    # 集合创建索引
    for field_name, index_name, index_param in index_params:
        collection.create_index(
            field_name=field_name,
            index_params=index_param,
            index_name=index_name
        )
    return True


def insert_datas(datas, collection_name, partition_name=None):
    """
    插入数据
    :param collection_name:
    :param datas: 数据集 list/tuple/pandas.DataFrame
    :param partition_name: 分区名称
    :return:
    """
    if not utility.has_collection(collection_name):
        return 0
    try:
        collection = Collection(collection_name)
        res = collection.insert(datas, partition_name=partition_name)
        return res.insert_count
    except Exception as e:
        return 0



def print_data(res):
    # 获取当前时间并格式化输出
    current_time = datetime.datetime.now()
    print("当前时间: ", current_time)

    # 或者格式化为指定样式（如 'YYYY-MM-DD HH:MM:SS'）
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print("格式化后的时间: ", formatted_time)
    with open("./record_chinese_data.txt", "a+") as f:
        f.writelines(f"{res}:{formatted_time}\n")


if __name__ == '__main__':
    start_insert_task()
