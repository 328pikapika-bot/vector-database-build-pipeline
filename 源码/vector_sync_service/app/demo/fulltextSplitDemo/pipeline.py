"""全文切分 Pipeline - 兼容 Milvus 扁平格式和 Kafka 嵌套格式"""
import json, os
from typing import List, Optional, Dict, Any
from .models import FullTextChunk, FullTextFileInfo, RedundantMeta, SplitResult, ChunkType
from .mmd_parser import MmdParser
from .text_splitter import TextSplitter
from .chunk_writer import ChunkWriter


class FullTextPipeline:
    def __init__(self, metadata_source="", milvus_uri="", milvus_collection="scientific_documents_v30",
                 base_dir="", output_dir="", mmd_dir="", max_chars=1500, overlap_chars=200, vector_dim=2560,
                 cleanup_attachments=True, download_base_url="http://127.0.0.1:12001/"):
        self.metadata_source = metadata_source
        self.milvus_uri = milvus_uri
        self.milvus_collection = milvus_collection
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_json"))
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "output")
        self.mmd_dir = mmd_dir or os.path.join(os.path.dirname(__file__), "mmd")
        self.download_base_url = download_base_url
        self.cleanup_attachments = cleanup_attachments
        self.splitter = TextSplitter(max_chars=max_chars, overlap_chars=overlap_chars, vector_dim=vector_dim)
        self.writer = ChunkWriter(output_dir=self.output_dir)
        os.makedirs(self.mmd_dir, exist_ok=True)
        self._milvus_client = None

    # ======================== Public API ========================

    def process_by_sys_id(self, sys_id: str) -> SplitResult:
        try:
            article_data = self._fetch_metadata(sys_id)
            if article_data is None:
                article_data = self._fetch_from_milvus(sys_id)
            if article_data is None:
                return self._process_local_only(sys_id)

            self._save_metadata_json(sys_id, article_data)

            fulltext_files = self._extract_fulltext_files(article_data)
            if not fulltext_files:
                fulltext_files = self._scan_local_mmd_files(sys_id)
            if not fulltext_files:
                return SplitResult(sysId=sys_id, success=False,
                                   errorMessage="No fulltext files found (no hasFullTextFile, no local .mmd)")

            meta = self._extract_redundant_meta(article_data)
            return self._process_fulltext_files(sys_id, fulltext_files, meta)
        except Exception as e:
            import traceback; traceback.print_exc()
            return SplitResult(sysId=sys_id, success=False, errorMessage=str(e))

    def process_batch(self, sys_ids):
        results = []
        for sid in sys_ids:
            print(f"\n[Pipeline] Processing sysId={sid} ...")
            r = self.process_by_sys_id(sid)
            results.append(r)
            print(f"[Pipeline] sysId={sid} -> {'OK' if r.success else 'FAIL'} (chunks={r.totalChunks})")
        if results:
            print(f"\n[Pipeline] Summary: {self.writer.save_summary(results)}")
            coll_path = self.writer.save_chunks_collection(results)
            total_chunks = sum(r.totalChunks for r in results)
            print(f"[Pipeline] All chunks (JSONL, {total_chunks} lines): {coll_path}")
        return results

    def query_milvus_record(self, sys_id: str) -> Optional[Dict]:
        client = self._get_milvus_client()
        if client is None:
            return None
        try:
            results = client.query(collection_name=self.milvus_collection,
                                   filter=f'ID == "{sys_id}"', output_fields=["*"], limit=1)
            return results[0] if results else None
        except Exception as e:
            print(f"[Pipeline] Query failed: {e}")
            return None

    def close(self):
        if self._milvus_client and not isinstance(self._milvus_client, bool):
            try: self._milvus_client.close()
            except: pass
        self._milvus_client = None

    # ======================== Data Sources ========================

    def _fetch_metadata(self, sys_id: str) -> Optional[Dict]:
        """从本地 JSON 文件查询。兼容两种格式:
          格式A: {"data": [{"dp": {...}, "op": {...}}]}  (Kafka 原始)
          格式B: {"sysId": "...", "title": [...], ...}   (Milvus 扁平)
        """
        if not self.metadata_source or not os.path.exists(self.metadata_source):
            return None
        try:
            with open(self.metadata_source, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Pipeline] Failed to load JSON: {e}")
            return None

        # 格式A: Kafka 包装
        if "data" in data and isinstance(data["data"], list):
            for article in data["data"]:
                sid = self._get_sys_id_from_record(article)
                if sid == sys_id:
                    print(f"[Pipeline] Found in JSON (Kafka format): sysId={sys_id}")
                    return article
            return None

        # 格式B: 扁平单条记录
        sid = self._get_sys_id_from_record(data)
        if sid == sys_id:
            print(f"[Pipeline] Found in JSON (flat format): sysId={sys_id}")
            return data
        return None

    def _get_sys_id_from_record(self, record: Dict) -> Optional[str]:
        """从记录中提取 sysId，兼容多种路径"""
        # 扁平: record["sysId"]
        if "sysId" in record:
            sid = record["sysId"]
            if isinstance(sid, list):
                return sid[0] if sid else None
            return str(sid) if sid else None
        # Kafka嵌套: record["dp"]["sysId"]["value"][0]
        dp = record.get("dp", {})
        if isinstance(dp, dict):
            sid_field = dp.get("sysId", {})
            if isinstance(sid_field, dict):
                vals = sid_field.get("value", [])
                return vals[0] if vals else None
        return None

    def _fetch_from_milvus(self, sys_id: str) -> Optional[Dict]:
        if not self.milvus_uri:
            return None
        client = self._get_milvus_client()
        if client is None:
            return None
        try:
            if not client.has_collection(self.milvus_collection):
                print(f"[Pipeline] Milvus collection '{self.milvus_collection}' not found")
                return None
            results = client.query(collection_name=self.milvus_collection,
                                   filter=f'ID == "{sys_id}"', output_fields=["*"], limit=1)
            if not results:
                print(f"[Pipeline] sysId={sys_id} not found in Milvus")
                return None
            print(f"[Pipeline] Found in Milvus: sysId={sys_id}")
            return results[0]  # Milvus 返回扁平记录
        except Exception as e:
            print(f"[Pipeline] Milvus query failed: {e}")
            return None

    def _get_milvus_client(self):
        if self._milvus_client is not None:
            return self._milvus_client if not isinstance(self._milvus_client, bool) else None
        try:
            from pymilvus import MilvusClient
            self._milvus_client = MilvusClient(uri=self.milvus_uri, timeout=10)
            print(f"[Pipeline] Milvus connected: {self.milvus_uri}")
            return self._milvus_client
        except Exception as e:
            print(f"[Pipeline] Milvus connection failed: {e}")
            self._milvus_client = None
            return None

    # ======================== FullText 提取 (核心修复) ========================

    def _extract_fulltext_files(self, article: Dict) -> List[FullTextFileInfo]:
        """
        提取全文文件信息。兼容两种格式:

        格式A (Kafka 嵌套):
          article["op"]["hasFullTextFile"]["value"] = [{"dp": {"name": {"value": [...]}, ...}}]

        格式B (Milvus 扁平 / sample_json):
          article["hasFullTextFile"] = [{"name": "xxx.mmd", "url": ["/fulltext/..."], "size": 4538, ...}]
        """
        # --- 尝试 Kafka 嵌套路径 ---
        op = article.get("op", {})
        if isinstance(op, dict) and "hasFullTextFile" in op:
            has_ft = op["hasFullTextFile"]
            if isinstance(has_ft, dict):
                raw_list = has_ft.get("value", [])
            elif isinstance(has_ft, list):
                raw_list = has_ft
            else:
                raw_list = []
            if raw_list:
                return self._parse_fulltext_items_kafka(raw_list)

        # --- 扁平格式 (Milvus / sample_json) ---
        has_ft = article.get("hasFullTextFile")
        if has_ft is None:
            return []

        # 可能是 JSON 字符串
        if isinstance(has_ft, str):
            try:
                has_ft = json.loads(has_ft)
            except (json.JSONDecodeError, TypeError):
                return []

        if not isinstance(has_ft, list):
            return []

        return self._parse_fulltext_items_flat(has_ft)

    def _parse_fulltext_items_kafka(self, items: list) -> List[FullTextFileInfo]:
        """解析 Kafka 嵌套格式的全文文件列表"""
        files = []
        for item in items:
            if not isinstance(item, dict):
                continue
            dp = item.get("dp", item)
            if not isinstance(dp, dict):
                continue
            name = self._pick_first(dp, "name")
            url = self._pick_first(dp, "url")
            ftype = self._pick_first(dp, "fileType") or "mmd"
            size = self._pick_first(dp, "size") or 0
            if name or url:
                files.append(FullTextFileInfo(
                    fileId=item.get("id", ""), fileName=str(name or ""),
                    url=str(url or ""), fileType=str(ftype),
                    fileSize=int(size) if size else 0))
        return files

    def _parse_fulltext_items_flat(self, items: list) -> List[FullTextFileInfo]:
        """解析扁平格式的全文文件列表"""
        files = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            url_raw = item.get("url", "")
            ftype = item.get("fileType", "mmd")
            size = item.get("size", 0)

            # url 可能是字符串或列表
            if isinstance(url_raw, list):
                url = url_raw[0] if url_raw else ""
            else:
                url = str(url_raw) if url_raw else ""

            if not name and url:
                name = os.path.basename(url)

            if name or url:
                files.append(FullTextFileInfo(
                    fileId=item.get("id", ""),
                    fileName=str(name),
                    url=url,
                    fileType=str(ftype),
                    fileSize=int(size) if size else 0))
                print(f"[Pipeline] Fulltext file: {name} (url={url[:80]}...)")
        return files

    @staticmethod
    def _pick_first(dp: Dict, field: str):
        """从 dp.field.value[0] 取值"""
        f = dp.get(field, {})
        if isinstance(f, dict):
            vals = f.get("value", [])
            return vals[0] if vals else None
        return f

    # ======================== 冗余元数据提取 ========================

    def _extract_redundant_meta(self, article: Dict) -> RedundantMeta:
        """提取冗余元数据，兼容 Kafka 嵌套和 Milvus 扁平两种格式"""
        dp = article.get("dp", None)

        if isinstance(dp, dict):
            # Kafka 嵌套格式
            return self._extract_meta_kafka(dp, article.get("op", {}))
        else:
            # Milvus 扁平格式
            return self._extract_meta_flat(article)

    def _extract_meta_kafka(self, dp: Dict, op: Dict) -> RedundantMeta:
        def _v(field):
            f = dp.get(field, {}); v = f.get("value", []); return v[0] if v else None
        def _vl(field):
            f = dp.get(field, {}); return f.get("value", [])

        authors = []
        for a in op.get("hasAuthor", {}).get("value", []):
            if isinstance(a, dict):
                n = a.get("dp", {}).get("name", {}).get("value", [])
                if n: authors.append(n[0])

        journal = None
        for s in op.get("hasSource", {}).get("value", []):
            if isinstance(s, dict):
                n = s.get("dp", {}).get("name", {}).get("value", [])
                if n: journal = n[0]; break

        sid = _v("sysId") or ""
        return RedundantMeta(sysId=sid, title=_v("title"), authors=authors,
                             year=str(_v("year")) if _v("year") else None,
                             journal=journal, doi=_v("doi"), keywords=_vl("keyword"))

    def _extract_meta_flat(self, article: Dict) -> RedundantMeta:
        def _first(val):
            if isinstance(val, list):
                return val[0] if val else None
            if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                try:
                    fixed = val.replace(chr(39), chr(34))
                    parsed = json.loads(fixed)
                except Exception:
                    try:
                        import ast; parsed = ast.literal_eval(val)
                    except Exception:
                        return val
                if isinstance(parsed, list) and parsed:
                    return str(parsed[0]) if parsed[0] is not None else None
                return val
            return val

        # 作者
        authors = []
        has_author = article.get("hasAuthor", [])
        if isinstance(has_author, str):
            try: has_author = json.loads(has_author)
            except: has_author = []
        if isinstance(has_author, list):
            for a in has_author:
                if isinstance(a, dict):
                    name = a.get("name", a.get("fullName", ""))
                    if name: authors.append(str(name))
                elif isinstance(a, str):
                    authors.append(a)

        # hasFullTextFilename
        ft_files = article.get("hasFullTextFilename", [])
        if isinstance(ft_files, str):
            try: ft_files = json.loads(ft_files)
            except: ft_files = []
        source_file = ft_files[0] if isinstance(ft_files, list) and ft_files else ""

        sid = str(_first(article.get("sysId", "")) or "")
        title = _first(article.get("title"))
        year = str(_first(article.get("year")) or "")
        keywords = article.get("keyword", [])
        if isinstance(keywords, str):
            keywords = [keywords]

        return RedundantMeta(
            sysId=sid, title=str(title) if title else None,
            authors=authors, year=year if year else None,
            journal=None, doi=None,
            keywords=list(keywords) if isinstance(keywords, list) else [],
            sourceFileName=str(source_file) if source_file else "")

    # ======================== 文件处理 ========================

    def _process_local_only(self, sys_id: str) -> SplitResult:
        files = self._scan_local_mmd_files(sys_id, global_fallback=False)
        if not files:
            return SplitResult(sysId=sys_id, success=False,
                               errorMessage=f"No .mmd files found for sysId={sys_id}")
        meta = RedundantMeta(sysId=sys_id)
        mmd_text = self._load_mmd_file(files[0], sys_id)
        if mmd_text:
            title = MmdParser.get_heading_title(mmd_text)
            if title: meta.title = title
        return self._process_fulltext_files(sys_id, files, meta)

    def _process_fulltext_files(self, sys_id, fulltext_files, meta):
        all_chunks = []
        source_files = []
        for ft in fulltext_files:
            mmd_text = self._load_mmd_file(ft, sys_id)
            if mmd_text is None:
                continue
            meta.sourceFileName = ft.fileName

            parsed_doc = MmdParser.parse(mmd_text)
            self._save_parse_result(parsed_doc, sys_id, ft.fileName)

            chunks = self.splitter.split_parsed_document(parsed_doc, sys_id, meta)
            all_chunks.extend(chunks)
            source_files.append(ft.fileName)
        if not all_chunks:
            return SplitResult(sysId=sys_id, success=False, errorMessage="No chunks extracted")
        chunks_by_type = {}
        for c in all_chunks:
            ct = c.chunkType.value if isinstance(c.chunkType, ChunkType) else str(c.chunkType)
            chunks_by_type[ct] = chunks_by_type.get(ct, 0) + 1
        result = SplitResult(sysId=sys_id, success=True, totalChunks=len(all_chunks),
                             chunksByType=chunks_by_type, chunks=all_chunks,
                             sourceFile=", ".join(source_files))
        jsonl_path = self.writer.save_chunks(result)
        print(f"[Pipeline] Chunks saved (JSONL, {result.totalChunks} lines): {jsonl_path}")
        if self.cleanup_attachments:
            self._cleanup_temp_files()
        return result

    def _scan_local_mmd_files(self, sys_id, global_fallback=True):
        files = []
        def _add_from_dir(directory):
            if not os.path.isdir(directory):
                return
            for fname in sorted(os.listdir(directory)):
                if fname.endswith(".mmd"):
                    fpath = os.path.join(directory, fname)
                    if fpath not in {f.url for f in files}:
                        files.append(FullTextFileInfo(fileName=fname, url=fpath, fileType="mmd",
                                                      fileSize=os.path.getsize(fpath)))
        _add_from_dir(self.mmd_dir)
        sys_dir = os.path.join(self.base_dir, sys_id)
        _add_from_dir(sys_dir)
        if global_fallback and os.path.isdir(self.base_dir):
            for root, dirs, filenames in os.walk(self.base_dir):
                for fname in filenames:
                    if fname.endswith(".mmd"):
                        fpath = os.path.join(root, fname)
                        if fpath not in {f.url for f in files}:
                            files.append(FullTextFileInfo(fileName=fname, url=fpath, fileType="mmd"))
        return files

    def _load_mmd_file(self, ft_info: FullTextFileInfo, sys_id=""):
        """加载 MMD 文件: 先尝试本地，再尝试 HTTP 下载"""
        text = self._load_mmd_local(ft_info, sys_id)
        if text:
            return text
        # 本地未找到 -> 尝试 HTTP 下载
        if self.download_base_url and ft_info.url:
            return self._download_mmd(ft_info)
        return None

    def _download_mmd(self, ft_info: FullTextFileInfo) -> Optional[str]:
        """HTTP 下载 .mmd 文件"""
        url = ft_info.url
        if isinstance(url, list):
            url = url[0] if url else ""
        if not url:
            return None
        # 拼接完整下载 URL
        if url.startswith("http"):
            full_url = url
        else:
            full_url = self.download_base_url.rstrip("/") + "/" + url.lstrip("/")
        try:
            import urllib.request
            print(f"[Pipeline] Downloading MMD: {full_url}")
            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")
            if text.strip():
                # 缓存到 mmd 目录
                cache_path = os.path.join(self.mmd_dir, ft_info.fileName)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[Pipeline] Downloaded MMD: {len(text)} chars, cached to {cache_path}")
                return text
        except Exception as e:
            print(f"[Pipeline] Download failed: {e}")
        return None

    def _load_mmd_local(self, ft_info: FullTextFileInfo, sys_id=""):
        candidates = []
        # mmd 下载目录 (优先)
        candidates.append(os.path.join(self.mmd_dir, ft_info.fileName))
        # 直接路径
        if ft_info.url and os.path.isabs(ft_info.url):
            candidates.append(ft_info.url)
        # sysId 专属目录
        if sys_id:
            sys_dir = os.path.join(self.base_dir, sys_id)
            if os.path.isdir(sys_dir):
                candidates.append(os.path.join(sys_dir, ft_info.fileName))
                for fname in sorted(os.listdir(sys_dir)):
                    if fname.endswith(".mmd"):
                        candidates.append(os.path.join(sys_dir, fname))
        # base_dir
        candidates.append(os.path.join(self.base_dir, ft_info.fileName))
        # url 拼接
        if ft_info.url:
            clean = ft_info.url.lstrip("/")
            candidates.append(os.path.join(self.base_dir, clean))
            candidates.append(os.path.join(self.base_dir, os.path.basename(ft_info.url)))
        # mmd_dir 全局扫描
        if os.path.isdir(self.mmd_dir):
            for root, dirs, filenames in os.walk(self.mmd_dir):
                for fname in filenames:
                    if fname.endswith(".mmd"):
                        candidates.append(os.path.join(root, fname))
        # base_dir 全局扫描
        if os.path.isdir(self.base_dir):
            for root, dirs, filenames in os.walk(self.base_dir):
                for fname in filenames:
                    if fname.endswith(".mmd"):
                        candidates.append(os.path.join(root, fname))
        seen = set()
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if candidate in seen: continue
            seen.add(candidate)
            try:
                if os.path.isfile(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        text = f.read()
                    if text.strip():
                        print(f"[Pipeline] Loaded MMD: {candidate} ({len(text)} chars)")
                        return text
            except Exception as e:
                print(f"[Pipeline] Read error {candidate}: {e}")
        print(f"[Pipeline] MMD not found: {ft_info.fileName}")
        return None

    def _save_metadata_json(self, sys_id: str, article_data: Dict):
        """将向量库查询结果保存到 sample_json/{sys_id}.json"""
        metadata_path = os.path.join(self.base_dir, f"{sys_id}.json")
        os.makedirs(self.base_dir, exist_ok=True)
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(article_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"[Pipeline] Metadata saved: {metadata_path}")
        except Exception as e:
            print(f"[Pipeline] Failed to save metadata: {e}")

    def _save_parse_result(self, parsed_doc, sys_id: str, file_name: str):
        """将 MMD AST 解析结果保存为 result.json"""
        result_filename = f"result_{sys_id}_{file_name}.json"
        result_path = os.path.join(self.output_dir, result_filename)
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(parsed_doc.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"[Pipeline] AST parse result saved: {result_path}")
        except Exception as e:
            print(f"[Pipeline] Failed to save parse result: {e}")

    def _cleanup_temp_files(self):
        """mmd 文件作为持久化缓存，不再清理"""
        pass
