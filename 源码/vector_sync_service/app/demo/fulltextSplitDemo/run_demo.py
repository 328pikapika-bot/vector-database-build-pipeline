"""全文切分 Demo 测试入口 (Milvus 数据源 + 本地回退)"""
import json, os, sys

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    from app.demo.fulltextSplitDemo.pipeline import FullTextPipeline

    print("=" * 70)
    print("  全文数据三级切分 Demo (Milvus + 本地回退)")
    print("=" * 70)

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(demo_dir, "sample_json")
    output_dir = os.path.join(demo_dir, "output")
    mmd_dir = os.path.join(demo_dir, "mmd")
    sample_json = os.path.join(base_dir, "sample.json")

    os.makedirs(mmd_dir, exist_ok=True)

    TARGET_SYS_ID = "b85b160c5f0c9778f85fbab812bf1050"
    MILVUS_URI = "http://127.0.0.1:19530"
    MILVUS_COLLECTION = "scientific_documents_v30"
    DOWNLOAD_BASE_URL = "http://127.0.0.1:12001/"

    print(f"\n[Config]")
    print(f"  Target sysId: {TARGET_SYS_ID}")
    print(f"  Milvus: {MILVUS_URI} / {MILVUS_COLLECTION}")
    print(f"  Download URL: {DOWNLOAD_BASE_URL}")
    print(f"  MMD cache: {mmd_dir}")
    print(f"  Local base: {base_dir}")

    # ==== Step 1: 先探测 Milvus，查看原始数据格式 ====
    print(f"\n[Step 1] Probing Milvus for sysId={TARGET_SYS_ID} ...")
    milvus_ok = False
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri=MILVUS_URI, timeout=10)
        if client.has_collection(MILVUS_COLLECTION):
            results = client.query(
                collection_name=MILVUS_COLLECTION,
                filter=f'ID == "{TARGET_SYS_ID}"',
                output_fields=["*"], limit=1)
            if results:
                milvus_ok = True
                rec = results[0]
                print(f"[Milvus] Record found! Fields ({len(rec)}): {list(rec.keys())}")
                for key in ["ID", "title", "year", "hasFullTextFile"]:
                    val = rec.get(key)
                    if val is not None:
                        print(f"  {key}: {repr(val)[:200]}")
                # 详细看 hasFullTextFile 的类型和结构
                has_ft = rec.get("hasFullTextFile")
                if has_ft:
                    print(f"\n  hasFullTextFile type: {type(has_ft).__name__}")
                    if isinstance(has_ft, str):
                        print(f"  hasFullTextFile (first 500 chars): {has_ft[:500]}")
                        try:
                            parsed = json.loads(has_ft)
                            print(f"  Parsed type: {type(parsed).__name__}")
                            if isinstance(parsed, dict):
                                print(f"  Parsed keys: {list(parsed.keys())}")
                                if "value" in parsed:
                                    vals = parsed["value"]
                                    print(f"  value count: {len(vals)}")
                                    for i, v in enumerate(vals[:3]):
                                        print(f"    [{i}] type={type(v).__name__}, keys={list(v.keys()) if isinstance(v, dict) else 'N/A'}")
                                        if isinstance(v, dict):
                                            print(f"        {json.dumps(v, ensure_ascii=False)[:300]}")
                            elif isinstance(parsed, list):
                                print(f"  Parsed list length: {len(parsed)}")
                                for i, v in enumerate(parsed[:3]):
                                    print(f"    [{i}] {repr(v)[:200]}")
                        except Exception as pe:
                            print(f"  JSON parse failed: {pe}")
                    elif isinstance(has_ft, dict):
                        print(f"  Dict keys: {list(has_ft.keys())}")
                    elif isinstance(has_ft, list):
                        print(f"  List length: {len(has_ft)}")
                        for i, v in enumerate(has_ft[:3]):
                            if isinstance(v, dict):
                                print(f"    [{i}] keys={list(v.keys())}")
                            else:
                                print(f"    [{i}] {repr(v)[:200]}")
            else:
                print(f"[Milvus] sysId not found in collection")
        else:
            print(f"[Milvus] Collection not found")
        client.close()
    except Exception as e:
        print(f"[Milvus] Connection failed: {e}")

    # ==== Step 2: 执行 Pipeline ====
    print(f"\n[Step 2] Running fulltext split pipeline ...")
    print("-" * 70)

    pipeline = FullTextPipeline(
        metadata_source=sample_json if os.path.exists(sample_json) else "",
        milvus_uri=MILVUS_URI if milvus_ok else "",
        milvus_collection=MILVUS_COLLECTION,
        base_dir=base_dir, output_dir=output_dir, mmd_dir=mmd_dir,
        max_chars=1500, overlap_chars=200, vector_dim=2560,
        cleanup_attachments=False,
        download_base_url=DOWNLOAD_BASE_URL,
    )

    result = pipeline.process_by_sys_id(TARGET_SYS_ID)

    # ==== Step 3: 输出 ====
    print("\n" + "=" * 70)
    print("  Result")
    print("=" * 70)
    if result.success:
        print(f"\n  Status: SUCCESS")
        print(f"  Total chunks: {result.totalChunks}")
        print(f"  By type: {result.chunksByType}")
        print(f"  Source: {result.sourceFile}")
        print(f"\n  Sample chunks (first 5):")
        for chunk in result.chunks[:5]:
            ct = chunk.chunkType.value if hasattr(chunk.chunkType, 'value') else chunk.chunkType
            preview = chunk.content[:80].replace("\n", " ")
            print(f"    [{ct}] {chunk.chunkId}: {preview}...")
        meta = result.chunks[0].redundantMeta if result.chunks else None
        if meta:
            print(f"\n  Redundant metadata:")
            print(f"    title: {meta.title}")
            print(f"    authors: {meta.authors}")
            print(f"    year: {meta.year}")
            print(f"    keywords: {meta.keywords}")
    else:
        print(f"\n  Status: FAILED")
        print(f"  Error: {result.errorMessage}")

    if os.path.isdir(output_dir):
        print(f"\n  Output files:")
        for fname in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath):
                size_kb = os.path.getsize(fpath) / 1024
                if fname.endswith(".jsonl"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        line_count = sum(1 for _ in f)
                    print(f"    {fname} ({line_count} lines, {size_kb:.1f} KB)")
                else:
                    print(f"    {fname} ({size_kb:.1f} KB)")

    pipeline.close()
    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)

if __name__ == "__main__":
    main()
