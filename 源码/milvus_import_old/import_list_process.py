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
import json
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"]="1"
import queue
import threading
import time
import xml.etree.ElementTree as ET
import requests
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
    "user": "ninemax",
    "password": os.getenv("MILVUS_PASSWORD", ""),
    "db_name": "nstl_db",
    "secure": False,
}

COLLECTION_INFO = {
    "name": "nstl_paper_abstract_ec_expand",
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

if not db.list_database().__contains__(MILVUS_DB['db_name']):

    db.create_database(MILVUS_DB['db_name'])

db.using_database(MILVUS_DB['db_name'])

CONSISTENCY_LEVEL = "Bounded"

time_res = "二版-第1次请求"
# 批次大小
num_items_to_get = 1000

# 文献数据队列
queue_papers = queue.Queue(1000000)

# api
api_url = 'http://127.0.0.1:14106/api/service/extract/data-handle/interface/getArticle'

# 游标设置
file_next_cursor = 'cursor_log.text'


next_cursor = open(file_next_cursor, 'r', encoding='utf-8').readline().strip()

api_paper_num = 0
model = SentenceTransformer('./m3e-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def text2vec(sentences, batch_size=64, normalize=True):
    embeddings = model.encode(sentences, batch_size=batch_size, device=device, normalize_embeddings=normalize)
    return embeddings

def find_info(pattern,str_info):
    return re.compile(rf"<{pattern}>(.*?)</{pattern}>",re.M).findall(str_info)

def find_sub_info(info_lists,name_lists):
    author_result = []
    for author in info_lists:
        temp = []
        for name in name_lists:
            temp.append("".join(find_info(name, author)))
        author_result.append("<+>".join(temp))
    return "<::>".join(author_result)

def parse_xml(xml_str):
    root = ET.fromstring(xml_str)
    papers = find_info("paper", xml_str)
    all_datas = []
    # 遍历子节点，并打印信息
    for child,paper in zip(root,papers):
        temp_dict = dict()
        for sub_child in child:
            if sub_child.tag == "hasSources" or sub_child.tag == "hasAuthors" or sub_child.tag == "hasCreateOrganizations" or sub_child.tag == "hasFiles":
                continue
            temp_dict[sub_child.tag] = sub_child.text

        temp_dict["hasAuthor"] = find_sub_info(find_info("hasAuthor", paper), ["id", "name"])
        temp_dict["hasFile"] = find_sub_info(find_info("hasFile", paper), ["id", "fileType"])
        temp_dict["hasSource"] = find_sub_info(find_info("hasSource", paper), ["id", "sysSourceId", "title"])
        temp_dict["hasCreateOrganization"] = find_sub_info(find_info("hasCreateOrganization", paper), ["id", "name"])

        if "abstract" in temp_dict.keys():
            temp_dict["classificationCode"] = "" if "classificationCode" not in temp_dict.keys() else temp_dict["classificationCode"]
            temp_dict["hasHoldinglibraryCode"] = "" if "hasHoldinglibraryCode" not in temp_dict.keys() else temp_dict["hasHoldinglibraryCode"]
            temp_dict["labelEvaluateGrade"] = "" if "labelEvaluateGrade" not in temp_dict.keys() else temp_dict["labelEvaluateGrade"]
            temp_dict["publisher"] = "" if "publisher" not in temp_dict.keys() else temp_dict["publisher"]
            temp_dict["issue"] = "" if "issue" not in temp_dict.keys() else temp_dict["issue"]
            temp_dict["startPage"] = "" if "startPage" not in temp_dict.keys() else temp_dict["startPage"]
            temp_dict["doi"] = "" if "doi" not in temp_dict.keys() else temp_dict["doi"]
            temp_dict["pageCount"] = "" if "pageCount" not in temp_dict.keys() else temp_dict["pageCount"]
            temp_dict["endPage"] = "" if "endPage" not in temp_dict.keys() else temp_dict["endPage"]
            temp_dict["volume"] = "" if "volume" not in temp_dict.keys() else temp_dict["volume"]
            temp_dict["sysProcessingUnit"] = "" if "sysProcessingUnit" not in temp_dict.keys() else temp_dict["sysProcessingUnit"]
            temp_dict["accessType"] = "" if "accessType" not in temp_dict.keys() else temp_dict["accessType"]
            all_datas.append(temp_dict)

    return all_datas



def get_time_str():
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return str(t)


def get_datas_api():
    """
    从api获取数据
    :return:
    """

    global next_cursor, api_paper_num

    print_data(f"获取数据：--{time_res}获取数据开始时间")

    with open("./journals.txt", "r+",encoding="utf-8") as f:
        journals = f.readlines()
    f.close()
    journals = [i.replace("\n", "") for i in journals]

    for journal in journals:
        next_cursor = "*"
        num = 0
        temp_list = []
        temp_abs_list = []
        while True:
            query_text =  '{"c":150,"q":[{"k":"hasSourcetitle_snt","v": '+f'"{journal}"'+',"o":"OR","a":1},{"k":"sysDataStatus","v":"enable","o":"AND","a":1}],"op":"AND","s":["id:asc"],"t":["Z01"]}'
            query_text =  '{"c":150,"q":[{"k":"hasSourcetitle_snt","v": "铁道标准化","o":"OR","a":1},{"k":"sysDataStatus","v":"enable","o":"AND","a":1}],"op":"AND","s":["id:asc"],"t":["Z01"]}'
            api_url = 'http://127.0.0.1:14106/api/service/extract/data-handle/interface/getArticle?fields=id,title,abstract,entityType,year,createTime,language,doi,hasSourcetitle,publisher,hasSourcesysSourceId&nextCursor=' + next_cursor + '&queryText=' + query_text

            res = requests.get(api_url)
            res = json.loads(res.text)

            if res['code'] == -1:
                print("Error for API request. nextCursor:", next_cursor)
                break

            cur_data = res['data']
            datas = cur_data['data']

            if not datas or datas == "":
                print("journal:",journal,"was finished")
                break


            datas = parse_xml(datas)
            start_time = time.time()
            for data in datas:

                if not data.__contains__('abstract'):
                    continue
                try:
                    temp_dict = dict()
                    temp_dict['id'] = data['id']
                    temp_dict['title'] = data['title']
                    temp_dict['abstract'] = data['abstract']
                    temp_dict['language'] = data['language']
                    temp_dict['sysProcessingUnit'] = data['sysProcessingUnit']
                    temp_dict['classificationCode'] = data['classificationCode']
                    temp_dict['hasHoldinglibraryCode'] = data['hasHoldinglibraryCode']
                    temp_dict['entityType'] = data['entityType']
                    temp_dict['year'] = data['year']
                    temp_dict['volume'] = data['volume']
                    temp_dict['issue'] = data['issue']
                    temp_dict['startPage'] = data['startPage']
                    temp_dict['endPage'] = data['endPage']
                    temp_dict['pageCount'] = data['pageCount']
                    temp_dict['accessType'] = data['accessType']
                    temp_dict['createTime'] = data['createTime']
                    temp_dict['doi'] = data['doi']
                    temp_dict['publisher'] = data['publisher']
                    temp_dict['labelEvaluateGrade'] = data['labelEvaluateGrade']
                    temp_dict['hasAuthor'] = data['hasAuthor']
                    temp_dict['hasSource'] = data['hasSource']
                    temp_dict['hasFile'] = data['hasFile']
                    temp_dict['hasCreateOrganization'] = data['hasCreateOrganization']
                    temp_abs_list.append(data['abstract'])
                    temp_list.append(temp_dict)
                    num += 1
                    api_paper_num += 1

                except Exception as e:
                    print(e)
                    continue

                if num % 50 == 0:
                    vec_abs_list = text2vec(temp_abs_list)
                    # print(vec_abs_list)

                    for x, y in zip(temp_list, vec_abs_list):
                        x['vec_abstract'] = y
                        queue_papers.put(x)

                    temp_list = []
                    temp_abs_list = []

                if api_paper_num % 1500 == 0:
                    print("========================api get num:", api_paper_num)

            # print("res——time",time.time()-start_time)
            # 更新游标记录日志
            next_cursor = cur_data['cursor']

            with open(file_next_cursor, 'w', encoding='utf-8') as f:
                f.write(next_cursor)


        if temp_list and temp_abs_list:
            vec_abs_list = text2vec(temp_abs_list)
            for x, y in zip(temp_list, vec_abs_list):
                x['vec_abstract'] = y
                queue_papers.put(x)

    print_data(f"获取数据：--{time_res}获取数据结束时间")


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
    num_all = 0
    num_success = 0
    print_data(f"数据入库：--{time_res}数据入库开始时间")

    while True:
        start_time = time.time()

        if queue_papers.empty():
            print('暂无数据，休息30秒')
            time.sleep(30)

        temp_list = []

        for _ in range(num_items_to_get):
            temp_list.append(queue_papers.get())
            num_all += 1
            if queue_papers.empty():
                break

        if not temp_list:
            continue

        num = insert_datas(temp_list, COLLECTION_INFO.get('name'))
        num_success += num
        print('Added datas:' + str(num) + '; all datas: ' + str(num_all) + '; success: ' + str(num_success),"time:",time.time()-start_time)

    print_data(f"数据入库:--{time_res}数据入库结束时间")
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


def import_data():
    """
        从api获取数据
        :return:
        """

    num = 0
    temp_list = []
    temp_abs_list = []

    cur_cursor = '*'

    # 创建集合
    res = create_collection(COLLECTION_INFO.get('name'), COLLECTION_INFO.get('fields'),
                            COLLECTION_INFO.get('num_partitions'))
    if not res:
        raise "集合创建失败：" + COLLECTION_INFO.get('name')

    while True:
        try:
            start_time = time.time()
            api_url = 'http://127.0.0.1:14106/api/service/extract/data-handle/interface/getArticle?fields=id,title,abstract,entityType,year,volume,issue,startPage,endPage,pageCount,accessType,createTime,doi,publisher,labelEvaluateGrade,hasAuthor,hasSource,hasFile,sysProcessingUnit,language,classificationCode,hasHoldinglibraryCode,hasCreateOrganization&nextCursor=' + cur_cursor + '&queryText={"c":500,"t":["J02"],"q":[{"k":"sysHaveAbsAuK","v":"1","f":"1"},{"k":"sysFieldInfo","v":"absStructureFieldInfo","o":"AND","a":"1"}],"s":["id"]}'

            res = requests.get(api_url)
            res = json.loads(res.text)

            if res['code'] != 0:

                print("Error for API request. nextCursor:", cur_cursor)
                time.sleep(5)

                continue
            print("current cursor: ", cur_cursor)

            cur_data = res['data']
            datas = cur_data['data']

            if datas == "" or num >= 1339027 // 50:
                break

            datas = parse_xml(datas)

            for data in datas:

                if not data.__contains__('abstract'):
                    continue

                temp_dict = dict()
                temp_dict['id'] = data['id']
                temp_dict['title'] = data['title']
                temp_dict['abstract'] = data['abstract']
                temp_abs_list.append('标题：' + data['title'] + '\n摘要：' + data['abstract'])
                # 文献类型
                temp_dict['paper_type'] = data['entityType']
                # 年份
                temp_dict['year'] = data['year']
                temp_dict['paper_time'] = data['createTime']
                temp_dict['create_time'] = get_time_str()
                # 语种
                temp_dict['language'] = data['language']
                temp_dict['source_id'] = data['hasSourcesysSourceId']
                temp_dict['source_name'] = data['hasSourcetitle']
                temp_dict['source_publisher'] = ''

                temp_list.append(temp_dict)
                num += 1

                if num % 20 == 0:

                    vec_abs_list = text2vec(temp_abs_list)
                    # print(vec_abs_list)

                    for x, y in zip(temp_list, vec_abs_list):
                        x['vec_abstract'] = y
                        cur_num = insert_datas(x, COLLECTION_INFO.get('name'))
                    print('Added datas num:' + str(num),"import data 用时：", time.time()-start_time,"average time:",str((time.time()-start_time)//num))

                    temp_list = []
                    temp_abs_list = []


            # 更新游标记录日志
            cur_cursor = cur_data['cursor']

        except Exception as e:
            print(e)
            continue

    if temp_list and temp_abs_list:
        vec_abs_list = text2vec(temp_list)
        for x, y in zip(temp_list, vec_abs_list):
            x['vec_abstract'] = y
            num = insert_datas(temp_list, COLLECTION_INFO.get('name'))
            print('Added datas:' + str(num))

def print_data(res):
    # 获取当前时间并格式化输出
    current_time = datetime.datetime.now()
    print("当前时间: ", current_time)

    # 或者格式化为指定样式（如 'YYYY-MM-DD HH:MM:SS'）
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print("格式化后的时间: ", formatted_time)
    with open("./record_chinese_data.txt","a+")as f:
        f.writelines(f"{res}:{formatted_time}\n")

if __name__ == '__main__':
    start_insert_task()
    print_data(f"{time_res}结束时间")
    # print_data("test")
    # import_data()
