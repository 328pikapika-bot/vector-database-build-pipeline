import asyncio
import json

import requests
import uvicorn
from fastapi import FastAPI, Body
import requests
import json
from kgs_reasoning_update.neo4j_interface.neo4j_interface_tool import search_multiple_relationship
app = FastAPI()
INDEX = 2

def open_llm_old(messages_query):
    # 定义 API URL
    url = "http://127.0.0.1:11434/api/generate"

    # 定义请求的 payload（请求数据）
    data = {
        "model": "deepseek-r1:70b",
        "prompt": messages_query
        }

    # 发送 POST 请求
    response = requests.post(url, json=data)

    if response.status_code == 200:
        response_data = response.content
        response_data = response_data.decode('utf-8')  # 假设数据是 UTF-8 编码
        response_data = json.dumps(response_data, indent=4, ensure_ascii=False)
        text = ""
        for chunk in response_data.split("\\n"):
            if chunk.strip():
                try:
                    if not chunk.startswith('"'):
                        chunk='"'+chunk
                    if not chunk.endswith('"'):
                        chunk = chunk+'"'

                    #chunk = chunk.decode('utf-8')  # 假设数据是 UTF-8 编码
                    chunk = json.dumps(chunk, indent=4, ensure_ascii=False)
                    c = chunk.replace("\\\\","").replace('\\"{','{').replace('}\\"','}').replace("false",'\\"False\\"').replace('\\"',"")

                    for i in c.split(","):
                        if "response" in i:
                            text+=i.split(":")[-1]

                except Exception as e:
                    continue

        return text.split("u003c/thinku003e")[-1]
    else:
        return "请求失败"

def open_llm(messages_query):
    # 定义 API URL
    url = "http://127.0.0.1:11434/api/generate"

    # 定义请求的 payload（请求数据）
    data = {
        "model": "deepseek-r1:70b",
        "prompt": messages_query,
        "stream": False,
        }

    # 发送 POST 请求
    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        result = result["response"]
        print("模型生成结果：", result)
        text = result.split("</think>")[-1]
        text = text.strip()
        return text
    else:
        return ""

def convert_to_text(result_list):
    text = ""
    for record in result_list:
        for key, value in record.items():
            if isinstance(value, dict):  # 如果值是字典，进一步展开
                text += f"{key}: {value}\n"
            else:
                text += f"{key}: {value}\n"
        text += "\n"  # 每个记录之后添加一个空行

    return text


def get_path_text(path):
    path_text = ""
    nodes = path.nodes
    relationships = path.relationships
    for n, r in zip(nodes, relationships):
        if n['name'] == "":
            n['name'] = n['name'].replace("", " ")
        path_text += "({})-[{}]-".format(n['name'], type(r).__name__)
    path_text += "(" + nodes[-1]['name']+")"  # 添加最后一个节点的名称
    # path_text += relationships[-1]['name']  # 添加最后一个节点的名称
    return path_text


def get_question_keywords(question):
    messages = f"""你是一个精通医学文献的专家。请从下面输入的问题中提取出与医学的关键词，只输出关键词，不要有任何提示性语句，严格保证你的回答遵循下面的要求:
        1、使用中文回答。
        2、如果问题中存在关键词，输出所有符合要求的关键词，多个关键词用英文分号“;”分隔。
        3、必须提取至少两个关键词。
        4、只输出关键词，不要有任何提示性语句。
        5、问题:{question}
        6、举例如下：
            问题1：广东省医院协会发布的行业标准在医学研究中，特别是心血管内科领域的先天性疾病诊疗中是否起到了重要指导作用？
            请你输出关键词1：广东省医院协会;行业标准;医学研究;心血管内科;先天性疾病;诊疗;指导作用。
            问题2：WS377.7-2013标准，标准名称是什么？
            请你输出关键词2：标准;标准名称。
        7、只输出关键词，如：广东省医院协会;行业标准;医学研究;心血管内科;先天性疾病;诊疗;指导作用,不要有任何提示性语句。
        """
    res = open_llm(messages)
    return res


async def query_neo4j(keywords):
    try:
        # print(data)
        match_infos = []
        # res = get_question_keywords(data["question"])
        res = keywords
        res = res.replace("；", ";").split(";")
        for r in res:
            match_infos.append(r.strip())

        print(match_infos)

        # 条件筛选规则1
        result = []
        if len(match_infos) == 0:
            result = []

        # 条件筛选规则2
        if len(match_infos) == 1:
            for i in search_multiple_relationship(after_attribute=match_infos[0], index=INDEX, behind_attribute=""):
                if '$' in get_path_text(i['p']):
                    name = get_path_text(i['p']).split('-')
                    for name_one in name[0].split('$'):
                        name_one += '-'
                        name_one += '-'.join(name[1:])
                        result.append(name_one)
                else:
                    result.append(get_path_text(i['p']))
            if not result:
                for i in search_multiple_relationship(after_attribute="", index=INDEX, behind_attribute=match_infos[0]):
                    if '$' in get_path_text(i['p']):
                        name = get_path_text(i['p']).split('-')
                        for name_one in name[0].split('$'):
                            name_one += '-'
                            name_one += '-'.join(name[1:])
                            result.append(name_one)
                    else:
                        result.append(get_path_text(i['p']))


        # 条件筛选规则3
        # TODO:
        for j in range(len(match_infos)):
            for z in range(j+1, len(match_infos)):
                for i in search_multiple_relationship(after_attribute=match_infos[j], index=INDEX,
                                                      behind_attribute=match_infos[z]):

                    if '$' in get_path_text(i['p']):
                        name = get_path_text(i['p']).split('-')
                        for name_one in name[0].split('$'):
                            name_one +='-'
                            name_one +='-'.join(name[1:])
                            result.append(name_one)
                    else:
                        result.append(get_path_text(i['p']))


                if len(result) == 0:
                    for i in search_multiple_relationship(after_attribute=match_infos[z], index=INDEX,
                                                          behind_attribute=match_infos[j]):
                        if '$' in get_path_text(i['p']):
                            name = get_path_text(i['p']).split('-')
                            for name_one in name[0].split('$'):
                                name_one += '-'
                                name_one += '-'.join(name[1:])
                                result.append(name_one)
                        else:
                            result.append(get_path_text(i['p']))


        # 条件筛选规则4
        for info in match_infos:
            for i in search_multiple_relationship(after_attribute="", index=INDEX,
                                                  behind_attribute=info):

                if '$' in get_path_text(i['p']):
                    name = get_path_text(i['p']).split('-')
                    for name_one in name[0].split('$'):
                        name_one += '-'
                        name_one += '-'.join(name[1:])
                        result.append(name_one)
                else:
                    result.append(get_path_text(i['p']))
                if len(result) > 100:
                    break

            if len(result) > 100:
                break

        # 条件筛选规则5
        for info in match_infos:
            for i in search_multiple_relationship(after_attribute=info, index=INDEX,
                                                  behind_attribute=""):

                if '$' in get_path_text(i['p']):
                    name = get_path_text(i['p']).split('-')
                    for name_one in name[0].split('$'):
                        name_one += '-'
                        name_one += '-'.join(name[1:])
                        result.append(name_one)
                else:
                    result.append(get_path_text(i['p']))
                if len(result) > 100:
                    break

            if len(result) > 100:
                break
        return list(set(result))
    except Exception as e:
        print("请求失败", e)
        return e


async def query_solr(keywords):
    # ==================查询solr=======================
    # print(data)
    match_infos = []
    # res = get_question_keywords(data["question"])
    res = keywords
    res = res.replace("；", ";").split(";")
    for r in res:
        match_infos.append(r.strip())

    print("solr_keywords", match_infos)

    # URL 地址
    # url = "http://127.0.0.1/prod-api/dataSearch/searchStandards"
    url = "http://127.0.0.1:8001/search_standards"

    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 请求体
    data = {
        "keywords": match_infos
    }

    # 发送 GET 请求
    response = requests.post(url, headers=headers, json=data)
    result_list=[]

    # 检查请求是否成功
    if response.status_code == 200:
        try:
            for data in response.json()["data"]["data"]:
                result_dict = {}
                result_dict["dmsp_dmsp_standard_standardId"] = data["dmsp_dmsp_standard_standardId"]
                result_dict["dmsp_dmsp_standard_standardName"] = data["dmsp_dmsp_standard_standardName"]
                result_dict["content"] = data
                result_list.append(result_dict)
            return result_list  # 打印响应的 JSON 内容
        except Exception as e:
            print("查询内容为空")
            return []
    else:
        return f"请求失败，状态码：{response.status_code}"


@app.post("/query")
async def query(question=Body(), query_type=Body(default=None)):
    # 提取出问题的关键词
    keywords = get_question_keywords(question)
    # keywords = "广东省医院协会;行业标准;医学研究;心血管内科;先天性疾病;诊疗;指导作用"

    # 使用 asyncio.gather 来并行执行多个异步操作，并确保结果被 await
    if query_type == "neo4j":
        results = await asyncio.gather(
            query_neo4j(keywords)
        )
    elif query_type == "solr":
        results = await asyncio.gather(
            query_solr(keywords)
        )
    else:
        results = await asyncio.gather(
            query_question_neo4j(question),
            query_question_neo4j(question),
            query_solr(keywords)
        )

    # results = await asyncio.gather(
    #     query_neo4j(keywords),
    #     query_solr(keywords)
    # )

    # 返回两个异步函数执行后的结果
    return results


if __name__ == "__main__":
    uvicorn.run(app="main_match:app", host="0.0.0.0", port=8998)
