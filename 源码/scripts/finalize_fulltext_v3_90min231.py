"""Read-only final observation and local result report for the90minute experiment."""
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time

from pymilvus import MilvusClient
from run_fulltext_v3_90min231 import Guard,SOURCE_URI,SOURCE_COLLECTION,TARGET_URI,write_json,now


ROOT=Path(__file__).resolve().parents[1]/'outputs/fulltext_v3_90min_20261116_115305'


def main():
    state=json.loads((ROOT/'status.json').read_text())
    assert state['status']=='completed_real_samples_and_verified_no_production_cutover'
    observation=ROOT/'final_observation';observation.mkdir(exist_ok=False)
    guard=Guard(observation)
    try:
        baseline=json.loads((ROOT/'server_baseline.json').read_text())
        while True:
            sample=guard.sample(force=True)
            if sample['services']!=baseline['services']:raise RuntimeError('Original service configuration/state changed')
            elapsed=(datetime.fromisoformat(now())-datetime.fromisoformat(state['started_at'])).total_seconds()
            print(json.dumps({'at':now(),'elapsed_seconds':elapsed,'cpu':sample['cpu_percent'],
                  'index_pending':sample['target_index_pending'],'index_running':sample['target_index_running']}),flush=True)
            if elapsed>=5400:break
            time.sleep(min(30,5401-elapsed))
        source=MilvusClient(uri=SOURCE_URI,timeout=20);target=MilvusClient(uri=TARGET_URI,timeout=20)
        try:
            def count(expr):return int(source.query(SOURCE_COLLECTION,filter=expr,output_fields=['count(*)'],consistency_level='Strong',timeout=90)[0]['count(*)'])
            counts={'total':count(''),'filled':count('hasSourcesysSourceId is not null')}
            assert counts=={'total':195893017,'filled':11659216}
            inventories={'production':source.list_collections(),'isolated':target.list_collections()}
            source_indexes=[source.describe_index(SOURCE_COLLECTION,n) for n in source.list_indexes(SOURCE_COLLECTION)]
            assert all(i.get('state')=='Finished' and i.get('pending_index_rows',0)==0 for i in source_indexes)
        finally:source.close();target.close()
        metrics_command="python3 -c \"import urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); print(o.open('http://127.0.0.1:9099/metrics',timeout=5).read().decode())\""
        _,out,err=guard.ssh.exec_command(metrics_command,timeout=15)
        metrics=out.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError(err.read().decode())
        selected=[line for line in metrics.splitlines() if line.startswith(('milvus_datacoord_task_num_in_scheduler{','milvus_datacoord_compaction_task_num{'))]
        write_json(observation/'background_metrics.json',selected)
        reports=[];ids=set();duplicates_across_samples=0;uncertain=unverified_confirmed=0
        for file in sorted(ROOT.glob('cycle_*/report.json')):
            report=json.loads(file.read_text());reports.append(report)
            db=sqlite3.connect('file:'+str(file.parent/'audit.sqlite').replace('\\','/')+'?mode=ro',uri=True)
            uncertain+=db.execute("SELECT count(*) FROM batches WHERE status='submitted_outcome_not_yet_confirmed'").fetchone()[0]
            unverified_confirmed+=db.execute("SELECT count(*) FROM expected_rows WHERE verified=0 AND seq IN (SELECT seq FROM batches WHERE status IN ('inserted_pending_verification','completed_and_verified'))").fetchone()[0]
            for key, in db.execute('SELECT id FROM expected_rows WHERE verified=1'):
                if key in ids:duplicates_across_samples+=1
                ids.add(key)
            db.close()
        assert uncertain==unverified_confirmed==duplicates_across_samples==0
        best=[c for c in state['cycles'] if c['workers']==4]
        rows=sum(c['rows'] for c in best);copy_seconds=sum(c['copy_seconds'] for c in best);complete_seconds=sum(c['complete_seconds'] for c in best)
        resources=[json.loads(line) for line in (ROOT/'resources.jsonl').read_text().splitlines()]
        resources += [json.loads(line) for line in (observation/'resources.jsonl').read_text().splitlines()]
        summary={'at':now(),'started_at':state['started_at'],'observed_elapsed_seconds':elapsed,
            'status':'90minute_real_test_completed_and_verified','no_source_writes':True,'no_existing_data_deleted':True,
            'existing_services_configuration_and_start_state_unchanged':True,'source_counts':counts,
            'source_indexes_finished':True,'inventories':inventories,
            'formal_verified_rows':state['verified_rows'],'formal_verified_new_source_ids':state['verified_new_source_ids'],
            'diagnostic_verified_rows':6000,'diagnostic_verified_new_source_ids':2000,
            'all_this_run_verified_rows':len(ids),'verified_unique_primary_keys':len(ids),
            'duplicate_primary_keys_across_test_cycles':duplicates_across_samples,
            'ambiguous_insert_batches':uncertain,'confirmed_rows_awaiting_verification':unverified_confirmed,
            'best_tested':{'batch_size':200,'workers':4,'repeat_count':len(best),'verified_rows':rows,
                'copy_rows_per_second':rows/copy_seconds,'complete_cycle_rows_per_second':rows/complete_seconds,
                'new_source_ids_per_complete_second':sum(c['new_source_ids'] for c in best)/complete_seconds,
                'copy_only_linear_days_for_195893017_rows':195893017/(rows/copy_seconds)/86400,
                'complete_cycle_linear_days_for_195893017_rows':195893017/(rows/complete_seconds)/86400},
            'max_sampled_host_cpu_percent':max(r['cpu_percent'] for r in resources),
            'max_sampled_target_memory_GiB':max(r['target_memory'] for r in resources)/1024**3,
            'min_sampled_host_available_GiB':min(r['available_memory'] for r in resources)/1024**3,
            'ending_target_index_pending':sample['target_index_pending'],'ending_target_index_running':sample['target_index_running'],
            'excluded_duplicate_keys':sum(r['excluded_duplicate_keys'] for r in reports),
            'excluded_physical_versions':sum(r['excluded_physical_versions'] for r in reports),
            'all_source_sample_rows_unchanged':sum(r['source_sample_rows_unchanged'] for r in reports),
            'all_target_original_fields_vectors_source_values_verified':True,
            'cycles':state['cycles'],'limitation':state['limitation'],
            'overall_formal_verified_rows_per_second_including_diagnosis_and_observation':state['verified_rows']/elapsed,
            'overall_formal_new_source_ids_per_second_including_diagnosis_and_observation':state['verified_new_source_ids']/elapsed,
            'full_migration_ready':False,'performance_not_absolute_maximum':True}
        write_json(ROOT/'final_summary.json',summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    finally:guard.ssh.close()


if __name__=='__main__':main()
