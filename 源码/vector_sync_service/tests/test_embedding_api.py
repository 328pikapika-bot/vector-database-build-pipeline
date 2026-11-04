"""Embedding API 连通性测试脚本"""
import requests
import json
import time
import sys

API_URL = "http://127.0.0.1:8080/v1/embeddings"
MODEL = "medium"

def embed_single(text: str) -> dict:
    """单文本向量化"""
    payload = json.dumps({"model": MODEL, "input": text})
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, headers=headers, data=payload, timeout=30)
    response.raise_for_status()
    return response.json()

def test_single_text():
    """测试1：单文本"""
    print("=" * 60)
    print("测试1: 单文本向量化")
    print("=" * 60)
    text = "这是一个测试文本"
    result = embed_single(text)
    
    # 提取向量
    data = result.get("data", [])
    if data:
        embedding = data[0].get("embedding", [])
        print(f"  输入: {text}")
        print(f"  向量维度: {len(embedding)}")
        print(f"  前5个值: {embedding[:5]}")
        print(f"  模型: {data[0].get('model', 'unknown')}")
        print(f"  [PASS] 单文本向量化成功\n")
        return len(embedding)
    else:
        print(f"  完整响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        print(f"  [FAIL] 未找到 embedding 数据\n")
        return None

def test_batch_texts():
    """测试2：批量文本（如果 API 支持数组输入）"""
    print("=" * 60)
    print("测试2: 批量文本向量化")
    print("=" * 60)
    texts = [
        "机器学习是人工智能的一个分支",
        "深度学习使用多层神经网络",
        "自然语言处理是AI的重要应用"
    ]
    
    # 尝试数组输入
    payload = json.dumps({"model": MODEL, "input": texts})
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, headers=headers, data=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        data = result.get("data", [])
        print(f"  输入文本数: {len(texts)}")
        print(f"  返回向量数: {len(data)}")
        for i, item in enumerate(data):
            emb = item.get("embedding", [])
            print(f"  文本{i+1}: 维度={len(emb)}, 前3值={emb[:3]}")
        if len(data) == len(texts):
            print(f"  [PASS] 批量向量化成功\n")
            return len(data[0].get("embedding", []))
        else:
            print(f"  [WARN] 返回数量不匹配\n")
            return None
    except Exception as e:
        print(f"  批量请求失败: {e}")
        print(f"  降级为逐个请求...")
        
        # 逐个请求
        results = []
        for i, text in enumerate(texts):
            try:
                r = embed_single(text)
                d = r.get("data", [])
                if d:
                    results.append(d[0].get("embedding", []))
                    print(f"  文本{i+1}: 维度={len(results[-1])}")
            except Exception as e2:
                print(f"  文本{i+1} 失败: {e2}")
        
        if len(results) == len(texts):
            print(f"  [PASS] 逐个请求全部成功\n")
            return len(results[0])
        else:
            print(f"  [FAIL] 部分失败\n")
            return None

def test_sample_data():
    """测试3：使用 sample.json 中的真实数据"""
    print("=" * 60)
    print("测试3: 真实样本数据向量化")
    print("=" * 60)
    
    # 模拟 parser 处理后的 title/abstract/keyword
    sample_title = "Effect of Furadan, Sevin, Rogor and Endotaf pesticides on nitrogen fixation by Nostocpaludosum"
    sample_abstract = "The effect of different concentrations of furadan, sevin, rogor and endotaf pesticides on nitrogen fixation by Nostocpaludosum was studied."
    sample_keywords = ["Nostocpaludosum", "furadan", "sevin", "rogor and endotaf pesticides", "nitrogen fixation"]
    
    fields = {
        "title": sample_title,
        "abstract": sample_abstract,
        "keyword": "; ".join(sample_keywords)
    }
    
    dims = {}
    for field_name, text in fields.items():
        try:
            r = embed_single(text)
            d = r.get("data", [])
            if d:
                emb = d[0].get("embedding", [])
                dims[field_name] = len(emb)
                print(f"  {field_name}: 维度={len(emb)}")
                print(f"    文本: {text[:80]}...")
                print(f"    向量前3: {emb[:3]}")
        except Exception as e:
            print(f"  {field_name}: 失败 - {e}")
    
    unique_dims = set(dims.values())
    if len(unique_dims) == 1:
        print(f"  [PASS] 所有字段向量维度一致: {unique_dims.pop()}\n")
    elif dims:
        print(f"  [WARN] 向量维度不一致: {dims}\n")
    else:
        print(f"  [FAIL] 所有字段向量化失败\n")
    return unique_dims.pop() if len(unique_dims) == 1 else None

def test_performance():
    """测试4：简单性能测试"""
    print("=" * 60)
    print("测试4: 性能测试（10次请求）")
    print("=" * 60)
    text = "性能测试文本 Performance test text for embedding API latency measurement"
    
    latencies = []
    for i in range(10):
        start = time.time()
        try:
            embed_single(text)
            elapsed = time.time() - start
            latencies.append(elapsed)
            print(f"  请求{i+1}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  请求{i+1}: 失败 - {e}")
    
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"  平均延迟: {avg:.3f}s")
        print(f"  最快: {min(latencies):.3f}s")
        print(f"  最慢: {max(latencies):.3f}s")
        print(f"  [INFO] QPS ≈ {1/avg:.1f}\n")

if __name__ == "__main__":
    print("\nNSTL Embedding API 测试")
    print(f"API: {API_URL}")
    print(f"Model: {MODEL}")
    print()
    
    try:
        dim1 = test_single_text()
        dim2 = test_batch_texts()
        dim3 = test_sample_data()
        test_performance()
        
        print("=" * 60)
        print("测试汇总")
        print("=" * 60)
        dims = [d for d in [dim1, dim2, dim3] if d is not None]
        if dims:
            print(f"  向量维度: {dims[0]}")
            all_same = all(d == dims[0] for d in dims)
            print(f"  维度一致性: {'PASS' if all_same else 'WARN - 维度不一致!'}")
        else:
            print(f"  状态: FAIL - 所有测试未获取到有效向量")
            sys.exit(1)
        
        print("\n所有测试完成。")
    except requests.exceptions.ConnectionError as e:
        print(f"\n[FATAL] 无法连接 API: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] 测试异常: {e}")
        sys.exit(1)
