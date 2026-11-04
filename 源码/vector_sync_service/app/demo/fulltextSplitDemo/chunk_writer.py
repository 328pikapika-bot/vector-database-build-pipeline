"""
子块存储写入器
输出格式: JSONL (每行一个子块) + 汇总 JSON
"""
import json
import os
from typing import List, Dict, Any
from .models import FullTextChunk, SplitResult


class ChunkWriter:
    """子块存储写入器"""

    def __init__(self, output_dir: str = ""):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def save_chunks(
        self, result: SplitResult, prefix: str = "chunks"
    ) -> str:
        """
        将单篇文章的子块保存为 JSONL 文件 (每行一个子块)

        Args:
            result: 切分结果
            prefix: 文件名前缀

        Returns:
            输出文件路径
        """
        filename = f"{prefix}_{result.sysId}.jsonl"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            for chunk in result.chunks:
                line = json.dumps(chunk.to_dict(), ensure_ascii=False)
                f.write(line + "\n")

        return filepath

    def save_chunks_collection(
        self,
        results: List[SplitResult],
        filename: str = "all_chunks.jsonl",
    ) -> str:
        """
        将所有文章的子块保存到一个 JSONL 集合文件 (每行一个子块)

        Args:
            results: 多篇文献的切分结果列表
            filename: 输出文件名

        Returns:
            输出文件路径
        """
        filepath = os.path.join(self.output_dir, filename)

        total = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for result in results:
                if result.success:
                    for chunk in result.chunks:
                        line = json.dumps(chunk.to_dict(), ensure_ascii=False)
                        f.write(line + "\n")
                        total += 1

        return filepath

    def save_summary(
        self, results: List[SplitResult], filename: str = "summary.json"
    ) -> str:
        """
        保存多篇文献的汇总结果

        Args:
            results: 多篇文献的切分结果列表
            filename: 输出文件名

        Returns:
            输出文件路径
        """
        filepath = os.path.join(self.output_dir, filename)

        summary = {
            "totalDocuments": len(results),
            "successCount": sum(1 for r in results if r.success),
            "failureCount": sum(1 for r in results if not r.success),
            "totalChunks": sum(r.totalChunks for r in results),
            "totalChunksByType": self._aggregate_chunk_types(results),
            "details": [
                {
                    "sysId": r.sysId,
                    "success": r.success,
                    "totalChunks": r.totalChunks,
                    "chunksByType": r.chunksByType,
                    "sourceFile": r.sourceFile,
                    "errorMessage": r.errorMessage,
                }
                for r in results
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return filepath

    @staticmethod
    def _aggregate_chunk_types(
        results: List[SplitResult]
    ) -> Dict[str, int]:
        """聚合各类型的子块数量"""
        aggregated: Dict[str, int] = {}
        for result in results:
            for ctype, count in result.chunksByType.items():
                aggregated[ctype] = aggregated.get(ctype, 0) + count
        return aggregated
