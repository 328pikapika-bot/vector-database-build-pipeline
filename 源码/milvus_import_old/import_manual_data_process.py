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
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
import torch.types
from pymilvus import DataType
from pymilvus import connections, CollectionSchema, FieldSchema, Collection, utility, db
from sentence_transformers import SentenceTransformer
import datetime

# 数据库配置
MILVUS_DB = {
    # "host": "127.0.0.1",
    # "host": "127.0.0.1",
    "host": "127.0.0.1",
    "port": "19530",
    "user": os.getenv("MILVUS_USER", ""),
    "password": os.getenv("MILVUS_PASSWORD", ""),
    "db_name": "PHDA_test",
    "secure": False,
}

# Collection字段配置
COLLECTION_INFO = {
    "name": "nstl_paper_abstract_updata",
    "num_partitions": 1024,
    "fields": [
        {
            "name": 'id',
            "dtype": DataType.VARCHAR,
            "description": "主键ID,与源数据id对应",
            "max_length": 64,
            "is_primary": True,
            "auto_id": True
        },
        {
            "name": 'vec_name_disc',
            "description": "数据集名称+描述向量值",
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
            "name": 'title',
            "description": "paper标题",
            "dtype": DataType.VARCHAR,
            "max_length": 1024
        },
        {
            "name": 'abstract',
            "description": "摘要原文",
            "dtype": DataType.VARCHAR,
            "max_length": 65535
        },
        {
            "name": 'paper_type',
            "description": "文献类型",
            "dtype": DataType.VARCHAR,
            "max_length": 64
        },
        {
            "name": 'language',
            "description": "语言",
            "dtype": DataType.VARCHAR,
            "max_length": 2,
            "index_params": {}
        },
        {
            "name": 'source_id',
            "description": "来源id",
            "dtype": DataType.VARCHAR,
            "max_length": 64
        },
        {
            "name": 'source_name',
            "description": "来源名称",
            "dtype": DataType.VARCHAR,
            "max_length": 1024,
            "index_params": {}
        },
        {
            "name": 'source_publisher',
            "description": "来源发布者",
            "dtype": DataType.VARCHAR,
            "max_length": 1024,
            "index_params": {}
        },
        {
            "name": 'year',
            "description": "源数据创建年份",
            "dtype": DataType.VARCHAR,
            "max_length": 4,
            "index_params": {}
        },
        {
            "name": 'paper_time',
            "description": "源数据的创建时间",
            "dtype": DataType.VARCHAR,
            "max_length": 19
        },
        {
            "name": 'create_time',
            "description": "入库时间",
            "dtype": DataType.VARCHAR,
            "max_length": 19
        }
    ]
}

# 连接数据库
conn = connections.connect(user=MILVUS_DB['user'], password=MILVUS_DB['password'], host=MILVUS_DB['host'],
                           port=MILVUS_DB['port'], alias="default")

if not db.list_database().__contains__(MILVUS_DB['db_name']):

    db.create_database(MILVUS_DB['db_name'])

db.using_database(MILVUS_DB['db_name'])

CONSISTENCY_LEVEL = "Bounded"

model = SentenceTransformer('./m3e-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def text2vec(sentences, batch_size=64, normalize=True):
    embeddings = model.encode(sentences, batch_size=batch_size, device=device, normalize_embeddings=normalize)
    return embeddings


def get_time_str():
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return str(t)

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

def search_mysql_and_meger():
    """
    搜索mysql数据并合并S
    :return:
    """

#修改
def process_insert():
    """
    数据导入任务
    :return:
    """
    # 创建集合
    res = create_collection(COLLECTION_INFO.get('name'), COLLECTION_INFO.get('fields'),
                            COLLECTION_INFO.get('num_partitions'))
    if not res:
        raise "集合创建失败：" + COLLECTION_INFO.get('name')

    vec_text2vec = text2vec("The quantum Hall effect is a prototypical realization of a topological state of matter. It emerges from a subtle interplay between topology, interactions and disorder. The disorder enables the formation of localized states in the bulk that stabilize the quantum Hall states with respect to the magnetic field and carrier density3. Still, the details of the localized states and their contribution to transport remain beyond the reach of most experimental techniques. Here we describe an extensive study of the bulk's heat conductance. Using a novel 'multiterminal' short device (on a scale of 10 μm), we separate the longitudinal thermal conductance, κ_(xx)T (owing to the bulk's contribution), from the topological transverse value κ_(xy)T by eliminating the contribution of the edge modes. When the magnetic field is tuned away from the conductance plateau centre, the localized states in the bulk conduct heat efficiently (κ_(xx)T∝ T), whereas the bulk remains electrically insulating. Fractional states in the first excited Landau level, such as the v = 7/3 and v = 5/2, conduct heat throughout the plateau with a finite κ_(xx)T. We propose a theoretical model that identifies the localized states as the cause of the finite heat conductance, agreeing qualitatively with our experimental findings.")
    temp_dict = dict()
    temp_dict['id'] = "a1e9eaffa46c09ab35d9d99837851b44"
    temp_dict['title'] = "Heat conductance of the quantum Hall bulk"
    temp_dict['abstract'] = "The quantum Hall effect is a prototypical realization of a topological state of matter. It emerges from a subtle interplay between topology, interactions and disorder. The disorder enables the formation of localized states in the bulk that stabilize the quantum Hall states with respect to the magnetic field and carrier density3. Still, the details of the localized states and their contribution to transport remain beyond the reach of most experimental techniques. Here we describe an extensive study of the bulk's heat conductance. Using a novel 'multiterminal' short device (on a scale of 10 μm), we separate the longitudinal thermal conductance, κ_(xx)T (owing to the bulk's contribution), from the topological transverse value κ_(xy)T by eliminating the contribution of the edge modes. When the magnetic field is tuned away from the conductance plateau centre, the localized states in the bulk conduct heat efficiently (κ_(xx)T∝ T), whereas the bulk remains electrically insulating. Fractional states in the first excited Landau level, such as the v = 7/3 and v = 5/2, conduct heat throughout the plateau with a finite κ_(xx)T. We propose a theoretical model that identifies the localized states as the cause of the finite heat conductance, agreeing qualitatively with our experimental findings."

    # 文献类型
    temp_dict['paper_type'] = "JournalPaper"

    # 年份
    temp_dict['year'] = "2024"
    temp_dict['paper_time'] = "2024-01-18 00:00:00"
    temp_dict['create_time'] = get_time_str()

    # 语种
    temp_dict['language'] = "en"
    temp_dict['vec_abstract'] = vec_text2vec
    temp_dict['source_id'] = "a1e9eaffa46c09ab35d9d99837851b44"
    temp_dict['source_name'] = "自然科学总论"
    temp_dict['source_publisher'] = '中国科学技术信息研究所'

    num = insert_datas(temp_dict, COLLECTION_INFO.get('name'))


import math

def euclidean_distance(point1, point2):
    # 假设point1和point2是两个列表，表示n维空间中的点
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def calculate_similarity(distance):
    # 将距离转换为相似度
    if distance != 0:
        return 1 / (distance + 1e-10)  # 添加1e-10以避免除以零
    else:
        return 0


if __name__ == '__main__':
    # process_insert()
    # collection = Collection(COLLECTION_INFO.get('name'))
    connection = connections.connect(user=os.getenv("MILVUS_USER", ""), password=os.getenv("MILVUS_PASSWORD", ""), host=os.getenv("MILVUS_HOST", "127.0.0.1"), port=os.getenv("MILVUS_PORT", "19530"))

    params = {
        "collection_name": "nstl_paper_abstract_updata",
        "query_condition": {"abstract": "无"},
        "limit": 1000  # 设置限制数量， Milvus默认返回前1000个结果
    }

    # 执行查询
    status, results = connection.search(**params)
    # 获取查询结果的ID列表
    entity_ids = [result[0] for result in results]
    print(entity_ids)
    # 删除满足条件的数据
    # collection.remove(entity_ids)

    # 关闭集合
    connections.close()

    # 断开与Milvus的连接
    connections.disconnect("default")
