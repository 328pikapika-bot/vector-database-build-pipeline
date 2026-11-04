#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/1
@Author  : st
@File    : import_process.py
@Project : Smart_Search
@Software: PyCharm
@文件说明:
为优化混合检索，创建10w数据量的测试集合

"""
import json
import os
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import queue
import threading
import time
import torch.types
from pymilvus import DataType
from pymilvus import connections, CollectionSchema, FieldSchema, Collection, utility, db
from sentence_transformers import SentenceTransformer
import datetime

# 数据库配置
MILVUS_DB = {
    # "host": "127.0.0.1",
    "host": "127.0.0.1",
    "port": "19530",
    "user": "ninemax",
    "password": os.getenv("MILVUS_PASSWORD", ""),
    "db_name": "nstl_db",
    "secure": False,
}

COLLECTION_INFO = {
    # "name": "nstl_paper_abstract_ec_expand_test_10w_old",
    "name": "nstl_paper_abstract_ec_expand_test_1w_old",
    "num_partitions": 1024,
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
            "name": 'vec_abstract',
            "description": "摘要向量值",
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
            "description": "摘要",
            "dtype": DataType.VARCHAR,
            "max_length": 65535
        },
        {
            "name": 'language',
            "description": "语言",
            "dtype": DataType.VARCHAR,
            "max_length": 10,
            "index_params": {}
        },
        {
            "name": 'sysProcessingUnit',
            "description": "加工单位",
            "dtype": DataType.VARCHAR,
            "max_length": 100
        },
        {
            "name": 'classificationCode',
            "description": "中图分类号",
            "dtype": DataType.VARCHAR,
            "max_length": 100,
            "index_params": {}
        },
        {
            "name": 'hasHoldinglibraryCode',
            "description": "馆藏单位",
            "dtype": DataType.VARCHAR,
            "max_length": 100,
            "index_params": {}
        },
        {
            "name": 'entityType',
            "description": "文献类型",
            "dtype": DataType.VARCHAR,
            "max_length": 50
        },
        {
            "name": 'year',
            "description": "年",
            "dtype": DataType.VARCHAR,
            "max_length": 50
        },
        {
            "name": 'volume',
            "description": "卷",
            "dtype": DataType.VARCHAR,
            "max_length": 50
        },
        {
            "name": 'issue',
            "description": "期",
            "dtype": DataType.VARCHAR,
            "max_length": 50
        },
        {
            "name": 'startPage',
            "description": "起始页",
            "dtype": DataType.VARCHAR,
            "max_length": 50
        },
        {
            "name": 'endPage',
            "description": "结束页",
            "dtype": DataType.VARCHAR,
            "max_length": 50
        },
        {
            "name": 'pageCount',
            "description": "总页数",
            "dtype": DataType.VARCHAR,
            "max_length": 50
        },
        {
            "name": 'accessType',
            "description": "获取来源",
            "dtype": DataType.VARCHAR,
            "max_length": 50
        },
        {
            "name": 'createTime',
            "description": "创建时间",
            "dtype": DataType.VARCHAR,
            "max_length": 50
        },
        {
            "name": 'doi',
            "description": "Doi",
            "dtype": DataType.VARCHAR,
            "max_length": 100
        },
        {
            "name": 'publisher',
            "description": "出版者",
            "dtype": DataType.VARCHAR,
            "max_length": 100
        },
        {
            "name": 'labelEvaluateGrade',
            "description": "收录级别",
            "dtype": DataType.VARCHAR,
            "max_length": 100
        },
        {
            "name": 'hasAuthor',
            "description": "作者列表",
            "dtype": DataType.VARCHAR,
            "max_length": 20000
        },
        {
            "name": 'hasSource',
            "description": "有母体",
            "dtype": DataType.VARCHAR,
            "max_length": 20000
        },
        {
            "name": 'hasFile',
            "description": "有全文",
            "dtype": DataType.VARCHAR,
            "max_length": 500
        },
        {
            "name": 'hasCreateOrganization',
            "description": "有机构",
            "dtype": DataType.VARCHAR,
            "max_length": 20000
        }
    ]
}

# 连接数据库
conn = connections.connect(user=MILVUS_DB['user'], password=MILVUS_DB['password'], host=MILVUS_DB['host'],
                           port=MILVUS_DB['port'], alias="default")

# 创建数据库
if not db.list_database().__contains__(MILVUS_DB['db_name']):
    db.create_database(MILVUS_DB['db_name'])
# 使用数据库
db.using_database(MILVUS_DB['db_name'])
# 创建集合
CONSISTENCY_LEVEL = "Bounded"

# 文献数据队列
queue_papers = queue.Queue(1000000)

model = SentenceTransformer('./m3e-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text2vec(sentences, batch_size=64, normalize=True):
    embeddings = model.encode(sentences, batch_size=batch_size, device=device, normalize_embeddings=normalize)
    return embeddings


def get_time_str():
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return str(t)


import pathlib


def get_all_files(root_dir):
    """获取指定目录及子目录下的所有文件路径"""
    root = pathlib.Path(root_dir)
    # 使用rglob递归查找所有文件
    return [str(file) for file in root.rglob('*') if file.is_file()]


def extract_id_and_name(data, field_names):
    temp_list = []
    # 用"∷"分割数据
    parts = data.split("∷")
    # 提取第一个::前面的唯一id
    unique_id = parts[0] if parts else ""
    temp_list.append(unique_id)
    for field_name in field_names:
        # 提取name≡后面的内容
        for part in parts:
            if part.startswith(field_name + "≡"):
                # 截取"≡"后面的部分
                name = part.split("≡")[1]
                temp_list.append(name)
                break
    return "<+>".join(temp_list)


def get_datas_api():
    """
    从api获取数据
    :return:
    """
    file_list = get_all_files("./data/")
    # file_list = file_list[:10]
    temp_abs_list = []
    temp_list = []
    num_all = 0
    num_success = 0
    num = 0
    for file_name in file_list:
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data = json.loads(line)
                if not (data.__contains__('auto_stringITMS_abstract') and data.__contains__('auto_stringITMS_title')):
                    continue
                try:
                    temp_dict = dict()
                    temp_dict['id'] = data['id']
                    temp_title = data['auto_stringITMS_title'][0].split("|")[-1] if data.get('auto_stringITMS_title',"") else ""
                    temp_dict['title'] = temp_title
                    temp_abstract = data['auto_stringITMS_abstract'][0].split("|")[-1] if data.get('auto_stringITMS_abstract', "") else ""
                    temp_dict['abstract'] = temp_abstract
                    temp_dict['language'] = data.get('stringIMS_language', [])[0] if data.get('stringIMS_language',"") else ""
                    temp_dict['sysProcessingUnit'] = data.get('stringIS_sysProcessingUnit', "")
                    temp_dict['classificationCode'] = data.get('stringIMS_classificationCode', [])[0] if data.get('stringIMS_classificationCode', "") else ""
                    temp_dict['hasHoldinglibraryCode'] = data['stringIMS_hasHoldinglibraryCode'][0] if data.get('stringIMS_hasHoldinglibraryCode', "") else ""
                    temp_dict['entityType'] = data.get('stringIMS_entityType', [])[0] if data.get('stringIMS_entityType', "") else ""
                    temp_dict['year'] = data.get('stringIS_year', "")
                    temp_dict['volume'] = data.get('stringIMS_volume', [])[0] if data.get('stringIMS_volume',"") else ""
                    temp_dict['issue'] = data.get('stringIMS_issue', [])[0] if data.get('stringIMS_issue', []) else ""
                    temp_dict['startPage'] = data.get('stringIS_startPage', "")
                    temp_dict['endPage'] = data.get('stringIS_endPage', "")
                    temp_dict['pageCount'] = data.get('stringIS_pageCount', "")
                    temp_dict['accessType'] = data['stringIMS_accessType'][0] if data.get('stringIMS_accessType',"") else ""
                    temp_dict['createTime'] = data.get('dateIS_createTime', "")
                    temp_dict['doi'] = data.get('stringIMS_doi', "")
                    temp_dict['publisher'] = ""
                    temp_dict['labelEvaluateGrade'] = data.get('stringIMS_labelEvaluateGrade')[0] if data.get('stringIMS_labelEvaluateGrade', "") else ""
                    temp_dict['hasAuthor'] = '<::>'.join([extract_id_and_name(x, ["name"]) for x in data.get('stringOM_hasAuthor', [])])
                    temp_dict['hasSource'] = '<::>'.join([extract_id_and_name(x, ["sysSourceId", "title"]) for x in data.get('stringOM_hasSource', [])])
                    temp_dict['hasFile'] = ""
                    temp_dict['hasCreateOrganization'] = '<::>'.join([extract_id_and_name(x, ["name"]) for x in data.get('hasCreateOrganization', [])])
                    temp_abs_list.append(temp_title + "\n" + temp_abstract)
                    temp_list.append(temp_dict)
                    num += 1
                except Exception as e:
                    print(traceback.print_exception())
                    continue

                if num % 50 == 0:
                    vec_abs_list = text2vec(temp_abs_list)
                    for x, y in zip(temp_list, vec_abs_list):
                        x['vec_abstract'] = y
                        queue_papers.put(x)
                    num_1 = insert_datas(temp_list, COLLECTION_INFO.get('name'))
                    num_success += num_1
                    print(
                        'Added datas:' + str(num_1) + '; all datas: ' + str(num_all) + '; success: ' + str(num_success))

                    temp_list = []
                    temp_abs_list = []

    if temp_list and temp_abs_list:
        vec_abs_list = text2vec(temp_abs_list)
        for x, y in zip(temp_list, vec_abs_list):
            x['vec_abstract'] = y
            queue_papers.put(x)
        num_1 = insert_datas(temp_list, COLLECTION_INFO.get('name'))
        num_success += num_1
        print('Added datas:' + str(num_1) + '; all datas: ' + str(num_all) + '; success: ' + str(num_success))

    print_data(f"获取数据：--获取数据结束时间")


def process_insert():
    """
    数据导入任务
    :return:
    """

    # 创建集合
    res = create_collection(COLLECTION_INFO.get('name'), COLLECTION_INFO.get('fields'),COLLECTION_INFO.get('num_partitions'))
    if not res:
        raise "集合创建失败：" + COLLECTION_INFO.get('name')
    num_all = 0
    num_success = 0
    print_data(f"数据入库：--数据入库开始时间")

    while True:
        start_time = time.time()

        if queue_papers.empty():
            print('暂无数据，休息30秒')
            time.sleep(30)

        temp_list = []

        for _ in range(1000):
            temp_list.append(queue_papers.get())
            num_all += 1
            if queue_papers.empty():
                break

        if not temp_list:
            continue

        num = insert_datas(temp_list, COLLECTION_INFO.get('name'))
        num_success += num
        print('Added datas:' + str(num) + '; all datas: ' + str(num_all) + '; success: ' + str(num_success), "time:",
              time.time() - start_time)

    print_data(f"数据入库:--数据入库结束时间")


def start_insert_task() -> object:
    # print_data(f"{time_res}开始时间")
    t1 = threading.Thread(target=get_datas_api, name='get_datas_api')
    t1.start()

    # time.sleep(10)

    print("启动api获取数据任务")
    t2 = threading.Thread(target=process_insert, name='process_insert')
    t2.start()
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
