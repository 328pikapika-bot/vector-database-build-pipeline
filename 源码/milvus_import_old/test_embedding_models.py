#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新的向量化模块集成
验证Qwen4B和BGE-base模型的支持
"""

import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from build_milvus_pdf.md_embedding_calculator import (
    create_embedding_calculator,
    Qwen4BEmbeddingCalculator,
    BGEBaseEmbeddingCalculator
)


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def test_factory_function():
    """测试工厂函数"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("测试嵌入计算器工厂函数")
    logger.info("=" * 60)
    
    # 测试Qwen4B配置
    qwen_config = {
        'model_type': 'qwen4b',
        'api_base': "https://api.siliconflow.cn/v1",
        'api_key': "test-key",  # 测试用，不会实际调用
        'model_name': "Qwen/Qwen3-Embedding-4B",
        'embedding_dimension': 2560  # 手动指定维度
    }
    
    try:
        qwen_calculator = create_embedding_calculator(qwen_config)
        logger.info(f"✓ Qwen4B计算器创建成功: {type(qwen_calculator).__name__}")
        logger.info(f"  模型名称: {qwen_calculator.model_name}")
        logger.info(f"  嵌入维度: {qwen_calculator.get_embedding_dimension()}")
    except Exception as e:
        logger.error(f"✗ Qwen4B计算器创建失败: {e}")
    
    print()
    
    # 测试BGE-base配置
    bge_config = {
        'model_type': 'bge_base',
        'model_name': "BAAI/bge-base-zh",
        'device': "cpu",  # 使用CPU避免GPU依赖
        'embedding_dimension': 768  # 手动指定维度
    }
    
    try:
        bge_calculator = create_embedding_calculator(bge_config)
        logger.info(f"✓ BGE-base计算器创建成功: {type(bge_calculator).__name__}")
        logger.info(f"  模型名称: {bge_calculator.model_name}")
        logger.info(f"  设备: {bge_calculator.device}")
        logger.info(f"  嵌入维度: {bge_calculator.get_embedding_dimension()}")
    except Exception as e:
        logger.warning(f"✗ BGE-base计算器创建失败（可能需要下载模型）: {e}")
        logger.warning("  注意：这只是创建阶段的测试，不需要实际模型权重")


def test_backward_compatibility():
    """测试向后兼容性"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("测试向后兼容性")
    logger.info("=" * 60)
    
    # 测试旧的参数方式（仅构建器层面）
    try:
        from build_milvus_pdf.md_milvus_builder import MarkdownMilvusBuilder
        
        # 使用旧的参数方式
        old_style_builder = MarkdownMilvusBuilder(
            milvus_host="test-host",
            milvus_port="19530",
            embedding_api_base="https://api.siliconflow.cn/v1",
            embedding_api_key="test-key",
            collection_name="test_collection",
            db_name="test_db",
            embedding_model="Qwen/Qwen3-Embedding-4B"
        )
        
        logger.info(f"✓ 旧的参数方式支持成功")
        logger.info(f"  嵌入维度: {old_style_builder.embedding_calculator.get_embedding_dimension()}")
        
    except Exception as e:
        logger.error(f"✗ 向后兼容性测试失败: {e}")


def test_embedding_calculator_classes():
    """测试嵌入计算器类的基本功能"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("测试嵌入计算器类")
    logger.info("=" * 60)
    
    # 测试Qwen4B类
    logger.info("测试Qwen4BEmbeddingCalculator:")
    try:
        qwen_calc = Qwen4BEmbeddingCalculator(
            api_base="https://api.siliconflow.cn/v1",
            api_key="test-key",
            model_name="Qwen/Qwen3-Embedding-4B",
            embedding_dimension=2560
        )
        
        dim = qwen_calc.get_embedding_dimension()
        logger.info(f"  ✓ 嵌入维度: {dim}")
        
        # 测试文本准备
        test_text = "测试文本"
        prepared = qwen_calc.prepare_text_for_embedding(test_text, 'segment_abstract')
        logger.info(f"  ✓ 文本准备完成，长度: {len(prepared)}")
        
    except Exception as e:
        logger.error(f"  ✗ Qwen4B测试失败: {e}")
    
    print()
    
    # 测试BGE类
    logger.info("测试BGEBaseEmbeddingCalculator:")
    try:
        bge_calc = BGEBaseEmbeddingCalculator(
            model_name="BAAI/bge-base-zh",
            device="cpu",
            embedding_dimension=768
        )
        
        dim = bge_calc.get_embedding_dimension()
        logger.info(f"  ✓ 嵌入维度: {dim}")
        
        # 测试文本准备
        test_text = "测试文本"
        prepared = bge_calc.prepare_text_for_embedding(test_text, 'segment_abstract')
        logger.info(f"  ✓ 文本准备完成，长度: {len(prepared)}")
        
    except Exception as e:
        logger.warning(f"  ✗ BGE测试失败（这是预期的，因为需要模型文件）: {e}")


def test_configuration_options():
    """测试不同的配置选项"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("测试配置选项")
    logger.info("=" * 60)
    
    config_examples = [
        {
            'name': 'Qwen4B API调用',
            'config': {
                'model_type': 'qwen4b',
                'api_base': "https://api.siliconflow.cn/v1",
                'api_key': "real-key-here",
                'model_name': "Qwen/Qwen3-Embedding-4B"
            }
        },
        {
            'name': 'Qwen4B 指定维度',
            'config': {
                'model_type': 'qwen4b',
                'api_base': "https://api.siliconflow.cn/v1",
                'api_key': "real-key-here",
                'model_name': "Qwen/Qwen3-Embedding-4B",
                'embedding_dimension': 2560  # 强制指定维度
            }
        },
        {
            'name': 'BGE-base 本地模型',
            'config': {
                'model_type': 'bge_base',
                'model_path': "./models/bge-base-zh",  # 本地路径
                'device': "cuda",
                'max_length': 512,
                'embedding_dimension': 768
            }
        },
        {
            'name': 'BGE-base HuggingFace',
            'config': {
                'model_type': 'bge_base',
                'model_name': "BAAI/bge-base-zh",  # 从HuggingFace下载
                'device': "cpu",
                'max_length': 256
            }
        }
    ]
    
    for example in config_examples:
        logger.info(f"测试: {example['name']}")
        try:
            calculator = create_embedding_calculator(example['config'])
            logger.info(f"  ✓ 配置有效")
            logger.info(f"    模型: {calculator.model_name}")
            logger.info(f"    维度: {calculator.get_embedding_dimension()}")
        except Exception as e:
            logger.info(f"  ✗ 配置测试失败: {e}")
        print()


def main():
    """主测试函数"""
    logger = setup_logging()
    
    logger.info("新的向量化模块集成测试")
    logger.info("=" * 60)
    
    # 运行所有测试
    test_factory_function()
    print("\n")
    
    test_backward_compatibility()
    print("\n")
    
    test_embedding_calculator_classes()
    print("\n")
    
    test_configuration_options()
    
    logger.info("=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)
    
    print("\n使用说明:")
    print("1. Qwen4B模型（API调用）:")
    print("   - 设置 model_type: 'qwen4b'")
    print("   - 提供 api_base 和 api_key")
    print("   - 可选: 通过 embedding_dimension 手动指定维度")
    print()
    print("2. BGE-base模型（本地推理）:")
    print("   - 设置 model_type: 'bge_base'")
    print("   - 提供 model_path（本地路径）或 model_name（HuggingFace模型名）")
    print("   - 设置 device: 'cuda' 或 'cpu'")
    print("   - 可选: 通过 embedding_dimension 手动指定维度")
    print()
    print("3. 在配置文件中使用:")
    print("""
    'embedding_config': {
        'model_type': 'qwen4b',  # 或 'bge_base'
        'api_base': 'your-api-base',
        'api_key': 'your-api-key',
        # 对于BGE模型:
        # 'model_path': './models/bge-base-zh',
        # 'device': 'cuda',
        'embedding_dimension': 2560  # 可选，覆盖自动检测
    }
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()