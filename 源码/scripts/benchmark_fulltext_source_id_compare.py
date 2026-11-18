"""Authorized bounded comparison: 10,000 real chunks, one added field only.

ABBA rounds use 2,500 different chunks each: single 231 / parallel three /
parallel three / single 231. Then 230 and 232 receive the single-mode group
separately, so all three end with the identical 10,000 field values.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

from benchmark_fulltext_source_id_100 import (
    COLLECTION, FIELD, HOSTS, ROOT, SOLR, SOLR_FIELD, TZ,
    connect, digest, plain, read_rows, snapshot,
)


TOTAL = 10000
ROUND_SIZE = 2500
BATCH = 500
POOL_SIZE = 12000


def log(event, **data):
    print(json.dumps({"at": datetime.now(TZ).isoformat(timespec="seconds"),
                      "event": event, **data}, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("--apply is required for the user-authorized bounded benchmark")
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise RuntimeError("Insufficient local disk for the original-record backup")
    folder = ROOT / "outputs" / ("fulltext_source_id_compare_" + datetime.now(TZ).strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    report = {"started_at": datetime.now(TZ).isoformat(), "collection": COLLECTION,
              "field": FIELD, "rows_per_host": TOTAL, "batch_size": BATCH,
              "status": "preflight", "hosts": {}, "batches": [], "rounds": []}
    def save():
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    clients = {host: connect(host) for host in HOSTS}
    before_hashes = {}
    try:
        for host, client in clients.items():
            baseline = snapshot(client)
            schema = baseline[COLLECTION]["fields"]
            target = next((f for f in schema if f["name"] == FIELD), None)
            if not target or not target.get("nullable") or len(schema) != 19:
                raise RuntimeError(f"Unexpected target schema on {host}")
            filled = int(client.query(COLLECTION, filter=f"{FIELD} is not null",
                output_fields=["count(*)"], consistency_level="Strong", timeout=30)[0]["count(*)"])
            report["hosts"][host] = {"before": baseline, "filled_before": filled}
            log("baseline", host=host, count=baseline[COLLECTION]["count"], filled=filled)
        if len({digest(h["before"]) for h in report["hosts"].values()}) != 1:
            raise RuntimeError("Collection identities/schema/counts differ across hosts")
        fields = [f["name"] for f in report["hosts"][HOSTS[0]]["before"][COLLECTION]["fields"]]
        original_fields = [f for f in fields if f != FIELD]

        started = time.perf_counter()
        # The initial null-field iterator timed out on the unindexed new field.
        # Select a bounded set of document seeds, then use the existing doc_id
        # INVERTED index; inspect null values locally instead of scanning them.
        seeds = clients[HOSTS[0]].query(COLLECTION, filter="chunkIndex == 0",
            output_fields=["doc_id"], limit=2000, consistency_level="Strong", timeout=30)
        seed_ids = list(dict.fromkeys(row["doc_id"] for row in seeds))
        candidates = []
        for offset in range(0, len(seed_ids), 50):
            wanted = seed_ids[offset:offset+50]
            rows = clients[HOSTS[0]].query(COLLECTION,
                filter="doc_id in " + json.dumps(wanted),
                output_fields=["id", "doc_id", FIELD], limit=16000,
                consistency_level="Strong", timeout=30)
            if len(rows) >= 16000:
                raise RuntimeError("Document lookup exceeded the bounded result limit")
            candidates.extend({"id":row["id"], "doc_id":row["doc_id"]}
                              for row in rows if row.get(FIELD) is None)
            if len(candidates) >= POOL_SIZE:
                candidates = candidates[:POOL_SIZE]
                break
        report["selection_method"] = "chunkIndex=0 seeds followed by indexed doc_id lookups; null checks local"
        report["source_read_seconds"] = time.perf_counter() - started
        if len({r["id"] for r in candidates}) != len(candidates):
            raise RuntimeError("Duplicate selected chunk IDs")
        doc_ids = sorted({r["doc_id"] for r in candidates})
        if any(not isinstance(i, str) or len(i) != 32 or any(c not in "0123456789abcdef" for c in i) for i in doc_ids):
            raise RuntimeError("Unexpected document ID format")
        log("selection_read", chunks=len(candidates), documents=len(doc_ids), seconds=report["source_read_seconds"])
        session = requests.Session()
        session.trust_env = False
        values = {}
        request_timings = []
        started = time.perf_counter()
        for offset in range(0, len(doc_ids), BATCH):
            wanted = doc_ids[offset:offset+BATCH]
            req_started = time.perf_counter()
            response = session.post(SOLR, data={"q": "{!terms f=id}" + ",".join(wanted),
                "fl": "id,"+SOLR_FIELD, "rows": len(wanted), "wt": "json", "timeAllowed": 10000},
                timeout=(6, 20))
            response.raise_for_status()
            body = response.json()
            if body.get("responseHeader", {}).get("partialResults"):
                raise RuntimeError("Partial Solr response")
            found = body["response"]["docs"]
            for row in found:
                value = row.get(SOLR_FIELD)
                if isinstance(value, list) and value and all(isinstance(v, str) and v.strip() for v in value):
                    values[row["id"]] = value
            request_timings.append({"requested_docs": len(wanted), "found_docs": len(found),
                                    "seconds": time.perf_counter()-req_started})
        report["solr_seconds"] = time.perf_counter() - started
        report["solr_requests"] = request_timings
        report["candidate_chunks"] = len(candidates)
        report["candidate_documents"] = len(doc_ids)
        report["documents_with_source_id"] = len(values)
        eligible = [r for r in candidates if r["doc_id"] in values]
        if len(eligible) < TOTAL:
            raise RuntimeError(f"Only {len(eligible)} eligible chunks; no writes performed")
        selected = eligible[:TOTAL]
        random.Random(20261113).shuffle(selected)
        ids = [r["id"] for r in selected]
        patches = [{"id": r["id"], FIELD: values[r["doc_id"]]} for r in selected]
        expected = {r["id"]: r[FIELD] for r in patches}
        report["selected_documents"] = len({r["doc_id"] for r in selected})
        report["patches_sha256"] = digest(patches)
        (folder / "selected_and_patches.json").write_text(json.dumps({"selected": selected, "patches": patches},ensure_ascii=False), encoding="utf-8")
        log("solr_ready", eligible=len(eligible), selected=TOTAL, selected_docs=report["selected_documents"], seconds=report["solr_seconds"])

        # Read/hash every selected original record on every host in bounded batches.
        # Only the 231 originals need a full local backup after hashes match.
        with gzip.open(folder / "original_231_10000.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as backup, ThreadPoolExecutor(max_workers=3) as pool:
            for offset in range(0, TOTAL, BATCH):
                batch_ids = ids[offset:offset+BATCH]
                futures = {pool.submit(read_rows, clients[host], batch_ids, fields): host for host in HOSTS}
                batch_hashes = {}
                for future in as_completed(futures):
                    host = futures[future]
                    rows = future.result()
                    if any(row.get(FIELD) is not None for row in rows.values()):
                        raise RuntimeError(f"A selected new field already has a value on {host}")
                    hashes = {i: digest({k: row[k] for k in original_fields}) for i, row in rows.items()}
                    batch_hashes[host] = hashes
                    if host == HOSTS[0]:
                        before_hashes.update(hashes)
                        for i in batch_ids:
                            backup.write(json.dumps(rows[i],ensure_ascii=False)+"\n")
                if len({digest(h) for h in batch_hashes.values()}) != 1:
                    raise RuntimeError("Selected records/vectors differ across hosts")
                if (offset+BATCH) % 2000 == 0:
                    log("originals_verified", chunks_per_host=offset+BATCH)
        (folder / "original_record_hashes.json").write_text(json.dumps(before_hashes), encoding="utf-8")
        report["status"] = "preflight_passed"
        save()
        log("preflight_passed", chunks_per_host=TOTAL, report=str(folder / "report.json"))

        def update(host, batch, barrier=None):
            if barrier is not None:
                barrier.wait(timeout=15)
            started = time.perf_counter()
            result = clients[host].upsert(COLLECTION, data=batch, partial_update=True, timeout=90)
            elapsed = time.perf_counter()-started
            if int(result.get("upsert_count", -1)) != len(batch):
                raise RuntimeError(f"Unexpected upsert count on {host}")
            return host, elapsed

        single_patches = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            for index, mode in enumerate(("single", "parallel", "parallel", "single")):
                group = patches[index*ROUND_SIZE:(index+1)*ROUND_SIZE]
                if mode == "single":
                    single_patches.extend(group)
                started = time.perf_counter()
                wall_sum = 0.0
                per_host = {host: 0.0 for host in (HOSTS if mode == "parallel" else HOSTS[:1])}
                for offset in range(0, ROUND_SIZE, BATCH):
                    batch = group[offset:offset+BATCH]
                    batch_started = time.perf_counter()
                    if mode == "parallel":
                        barrier = threading.Barrier(3)
                        results = [future.result() for future in [pool.submit(update, host, batch, barrier) for host in HOSTS]]
                    else:
                        results = [update(HOSTS[0], batch)]
                    wall = time.perf_counter()-batch_started
                    wall_sum += wall
                    for host, elapsed in results:
                        per_host[host] += elapsed
                    report["batches"].append({"round": index+1, "mode": mode, "offset": offset,
                        "rows_per_host": len(batch), "wall_seconds": wall,
                        "host_seconds": dict(results), "ids_sha256": digest([r["id"] for r in batch])})
                    save()
                result = {"round": index+1, "mode": mode, "rows_per_host": ROUND_SIZE,
                          "wall_seconds": time.perf_counter()-started,
                          "write_seconds_sum": wall_sum, "host_seconds_sum": per_host}
                report["rounds"].append(result)
                save()
                log("round_completed", **result)

        # Complete the single-mode group on each remaining host, measured separately.
        report["isolated_followup"] = {}
        for host in HOSTS[1:]:
            started = time.perf_counter()
            seconds = 0.0
            for offset in range(0, len(single_patches), BATCH):
                batch = single_patches[offset:offset+BATCH]
                _, elapsed = update(host, batch)
                seconds += elapsed
                report["batches"].append({"mode": "isolated_followup", "host": host,
                    "rows_per_host": len(batch), "seconds": elapsed,
                    "ids_sha256": digest([r["id"] for r in batch])})
                save()
            report["isolated_followup"][host] = {"rows": len(single_patches),
                "write_seconds_sum": seconds, "wall_seconds": time.perf_counter()-started}
            save()
            log("followup_completed", host=host, **report["isolated_followup"][host])

        report["status"] = "verifying"
        save()
        with ThreadPoolExecutor(max_workers=3) as pool:
            for offset in range(0, TOTAL, BATCH):
                batch_ids = ids[offset:offset+BATCH]
                futures = {pool.submit(read_rows, clients[host], batch_ids, fields):host for host in HOSTS}
                for future in as_completed(futures):
                    host = futures[future]
                    rows = future.result()
                    for i, row in rows.items():
                        if row.get(FIELD) != expected[i]:
                            raise RuntimeError(f"Wrong new value on {host}/{i}")
                        if digest({k:row[k] for k in original_fields}) != before_hashes[i]:
                            raise RuntimeError(f"Original record/vector changed on {host}/{i}")
                if (offset+BATCH) % 2000 == 0:
                    log("post_update_verified", chunks_per_host=offset+BATCH)
        for host, client in clients.items():
            after = snapshot(client)
            if after != report["hosts"][host]["before"]:
                raise RuntimeError(f"Collection counts/schema/index inventory/load states changed on {host}")
            filled = int(client.query(COLLECTION,filter=f"{FIELD} is not null",output_fields=["count(*)"],
                                     consistency_level="Strong",timeout=30)[0]["count(*)"])
            if filled != report["hosts"][host]["filled_before"]+TOTAL:
                raise RuntimeError(f"Unexpected filled count on {host}: {filled}")
            report["hosts"][host].update({"after":after,"filled_after":filled,
                "original_10000_unchanged":True,"source_values_match":True})
            save()

        single_seconds = sum(r["write_seconds_sum"] for r in report["rounds"] if r["mode"] == "single")
        parallel_seconds = sum(r["write_seconds_sum"] for r in report["rounds"] if r["mode"] == "parallel")
        followup_seconds = sum(r["write_seconds_sum"] for r in report["isolated_followup"].values())
        n = report["hosts"][HOSTS[0]]["before"][COLLECTION]["count"]
        common = (report["source_read_seconds"]+report["solr_seconds"])/len(candidates)
        summary = {
            "comparison_rows_per_mode_per_host":5000,
            "single_231_write_seconds":single_seconds,
            "single_231_rows_per_second":5000/single_seconds,
            "parallel_three_write_seconds":parallel_seconds,
            "parallel_rows_per_second_per_host":5000/parallel_seconds,
            "sequential_three_write_seconds":single_seconds+followup_seconds,
            "sequential_over_parallel_speedup":(single_seconds+followup_seconds)/parallel_seconds,
            "three_vs_one_time_ratio":parallel_seconds/single_seconds,
            "shared_source_seconds_per_candidate_chunk":common,
            "full_count":n,
            "estimated_single_231_with_shared_source_hours":n*(common+single_seconds/5000)/3600,
            "estimated_parallel_three_with_shared_source_hours":n*(common+parallel_seconds/5000)/3600,
            "estimated_sequential_three_with_shared_source_hours":n*(common+(single_seconds+followup_seconds)/5000)/3600,
            "physical_copy_break_even_hours":n*(parallel_seconds-single_seconds)/5000/3600,
            "physical_copy_speed_not_measured":True,
            "extrapolation_note":"Linear short-run extrapolation; shared Solr lookups deduplicated by doc_id; long-run compaction/indexing/disk/cache effects not included.",
        }
        report["summary"] = summary
        report["three_hosts_10000_identical"] = True
        report["finished_at"] = datetime.now(TZ).isoformat()
        report["status"] = "completed"
        save()
        log("completed", **summary, report=str(folder / "report.json"))
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        save()
        log("stopped", error=repr(exc), report=str(folder / "report.json"))
        raise
    finally:
        for client in clients.values():
            client.close()


if __name__ == "__main__":
    main()
