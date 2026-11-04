#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试简化版BGE-base模型向量化逻辑
参考用户提供的m3e-base实现方式
"""

import logging
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_simple_bge():
    """测试简化版BGE模型实现"""
    
    print("=" * 60)
    print("测试简化版BGE-base模型向量化逻辑")
    print("=" * 60)
    
    try:
        from build_milvus_pdf.md_embedding_calculator import BGEBaseEmbeddingCalculator
        
        # 创建简化版BGE模型计算器
        calculator = BGEBaseEmbeddingCalculator(
            model_name="BAAI/bge-base-zh",
            device="cpu",  # 使用CPU避免CUDA依赖
            max_length=512
        )
        
        print(f"✓ BGE模型计算器创建成功")
        print(f"  模型名称: {calculator.model_name}")
        print(f"  嵌入维度: {calculator.get_embedding_dimension()}")
        
        # 测试单个文本向量化
        test_text = "深度学习在自然语言处理中的应用已经非常广泛"
        
        print(f"\n测试单个文本向量化:")
        print(f"  文本: {test_text}")
        
        embedding = calculator.calculate_embedding_sync(test_text)
        
        if embedding is not None:
            print(f"✓ 嵌入计算成功")
            print(f"  向量长度: {len(embedding)}")
            print(f"  向量前5个值: {embedding[:5]}")
        else:
            print(f"✗ 嵌入计算失败")
        
        # 测试批量向量化（参考用户提供的text2vec_1函数）
        print(f"\n测试批量文本向量化（text2vec方法）:")
        
        test_sentences = [
            "机器学习是人工智能的核心技术",
            "深度学习在计算机视觉领域表现优异",
            "自然语言处理让机器理解人类语言"
        ]
        
        print(f"  文本列表: {test_sentences}")
        
        embeddings = calculator.text2vec(test_sentences, batch_size=2, normalize=True)
        
        if embeddings is not None and len(embeddings) > 0:
            print(f"✓ 批量嵌入计算成功")
            print(f"  嵌入数量: {len(embeddings)}")
            print(f"  每个向量的维度: {embeddings.shape[1] if hasattr(embeddings, 'shape') else len(embeddings[0])}")
            
            # 展示第一个向量的前几个值
            if hasattr(embeddings, '__len__') and len(embeddings) > 0:
                first_embedding = embeddings[0]
                if hasattr(first_embedding, 'tolist'):
                    first_embedding = first_embedding.tolist()
                print(f"  第一个向量前5个值: {first_embedding[:5]}")
        
        # 测试工厂函数兼容性
        print(f"\n测试工厂函数兼容性:")
        
        from build_milvus_pdf.md_embedding_calculator import create_embedding_calculator
        
        bge_config = {
            'model_type': 'bge_base',
            'model_name': "BAAI/bge-base-zh",
            'device': "cpu",
            'embedding_dimension': 768
        }
        
        factory_calculator = create_embedding_calculator(bge_config)
        
        print(f"✓ 工厂函数创建成功: {type(factory_calculator).__name__}")
        
        # 展示使用方法
        print(f"\n" + "=" * 60)
        print("简化版BGE模型使用方法:")
        print("=" * 60)
        
        usage_example = '''
# 1. 创建模型实例
from build_milvus_pdf.md_embedding_calculator import BGEBaseEmbeddingCalculator

model = BGEBaseEmbeddingCalculator(
    model_name="BAAI/bge-base-zh",  # 或指定本地路径
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

# 2. 单个文本向量化
embedding = model.calculate_embedding_sync("你的文本内容")

# 3. 批量文本向量化（推荐使用text2vec方法）
sentences = ["文本1", "文本2", "文本3"]
embeddings = model.text2vec(sentences, batch_size=16, normalize=True)

# 4. 或者使用工厂函数
from build_milvus_pdf.md_embedding_calculator import create_embedding_calculator

config = {
    'model_type': 'bge_base',
    'model_name': "BAAI/bge-base-zh",
    'device': "cuda",
    'embedding_dimension': 768
}
calculator = create_embedding_calculator(config)
        '''
        
        print(usage_example)
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("可能需要安装依赖: pip install sentence-transformers transformers torch")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_reference():
    """对比简化实现与参考代码的相似性"""
    
    print("\n" + "=" * 60)
    print("对比简化实现与参考代码")
    print("=" * 60)
    
    print("参考代码 (用户提供):")
    print("-" * 40)
    print('''model = SentenceTransformer(EMBEDDING_MODEL_LOCAL_DICT['m3e-base'])

def text2vec_1(sentences, batch_size=16, normalize=False):
    try:
        embeddings = model.encode(sentences, batch_size=batch_size, device='cuda:0', normalize_embeddings=normalize)
        return embeddings
    except:
        log.error(traceback.format_exc())
        log.error("embedding error: sentences:" + str(sentences))
        return [np.zeros(768) for _ in sentences]''')
    
    print("\n简化版BGE实现核心部分:")
    print("-" * 40)
    print('''def text2vec(self, sentences, batch_size=16, normalize=True):
    try:
        self._initialize_model()
        embeddings = self.model.encode(
            sentences, 
            batch_size=batch_size, 
            device=self.device, 
            normalize_embeddings=normalize
        )
        return embeddings
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"embedding error: sentences:" + str(sentences))
        dim = self.get_embedding_dimension()
        return np.array([np.zeros(dim) for _ in sentences])''')
    
    print("\n✓ 简化实现的优势:")
    print("  1. 代码结构极简，易于理解和维护")
    print("  2. 直接使用SentenceTransformer.encode方法")
    print("  3. 异常处理简洁明了")
    print("  4. 保持与原始API的兼容性")

if __name__ == "__main__":
    print("简化版BGE模型向量化逻辑测试")
    print("=" * 60)
    
    success = test_simple_bge()
    
    if success:
        compare_with_reference()
        
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)