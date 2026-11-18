"""Coordinate one collection-property A/B between existing bounded trials.

No data, indexes, or collections are deleted. The original property inheritance
is restored exactly if this run added a temporary property key.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from benchmark_fulltext_source_id_100 import COLLECTION, connect

ROOT=Path(__file__).resolve().parents[1]/'outputs/fulltext_source_id_stable_trial_231_20261115'
WORK=Path(__file__).resolve().parents[1]/'outputs/fulltext_source_id_local231_continuous_20261113'
KEY='collection.autocompaction.enabled'


def now():return datetime.now(timezone.utc).isoformat()


def main():
    path=ROOT/'mix_ab_state.json'
    if path.exists():raise RuntimeError('Existing A/B state must not be overwritten')
    state={'pid':os.getpid(),'started_at':now(),'status':'waiting_for_baseline_quiet_80_seconds','collection':COLLECTION,'property':KEY}
    client=connect('192.0.2.15')
    baseline=dict(client.describe_collection(COLLECTION).get('properties') or {})
    state['original_properties']=baseline
    changed=False
    restored=False
    def save():
        state['at']=now()
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
    def restore():
        nonlocal restored
        state['status']='restoring_original_property';save()
        current=dict(client.describe_collection(COLLECTION).get('properties') or {})
        if current!=baseline:
            if KEY in baseline:
                client.alter_collection_properties(collection_name=COLLECTION,properties={KEY:baseline[KEY]},timeout=30)
            else:
                client.drop_collection_properties(collection_name=COLLECTION,property_keys=[KEY],timeout=30)
        properties=dict(client.describe_collection(COLLECTION).get('properties') or {})
        state['properties_after_restore']=properties
        if properties!=baseline:raise RuntimeError('Collection properties differ after restoration')
        restored=True;state['property_restored_at']=now();state['status']='property_restored_waiting_for_deferred_background';save()
    save()
    try:
        while True:
            report=json.loads((ROOT/'status.json').read_text(encoding='utf-8'))
            stages=report.get('stages',[])
            state['main_wrapper_status']=report['status']
            if not changed:
                if len(stages)>1:raise RuntimeError('Second trial started before the planned A/B boundary')
                host=report.get('host',{})
                idle=not any(host.get(k,0) for k in ['index_pending','index_running','compaction_pending','compaction_running'])
                if report['status']=='draining_index_after_stage1' and report.get('background_quiet_seconds',0)>=80 and idle:
                    properties=dict(client.describe_collection(COLLECTION).get('properties') or {})
                    if properties!=baseline:raise RuntimeError('Collection properties changed before A/B')
                    state['baseline_stage_snapshot']=stages[0]
                    state['baseline_boundary_at']=now();state['baseline_quiet_seconds']=report['background_quiet_seconds']
                    state['status']='applying_mix_off';save()
                    changed=True
                    client.alter_collection_properties(collection_name=COLLECTION,properties={KEY:'false'},timeout=30)
                    after=dict(client.describe_collection(COLLECTION).get('properties') or {})
                    expected=dict(baseline);expected[KEY]='false'
                    if after!=expected:raise RuntimeError('Only the target property was expected to change')
                    state['properties_during_trial']=after;state['mix_off_applied_at']=now();state['status']='mix_off_waiting_for_second_trial';save()
            elif not restored and len(stages)>1 and stages[1].get('index_drained_at'):
                state['mix_off_stage_snapshot']=stages[1]
                restore()
            if report.get('error') or report['status']=='failed':
                raise RuntimeError('Main bounded trial wrapper failed: '+str(report.get('error')))
            if changed and restored and report.get('cpu_quota_restored_at'):
                state['deferred_background_ready_at']=report['at']
                state['main_final_index_drain_seconds']=report.get('final_index_drain_seconds')
                state['status']='completed_property_restored';save();break
            save();time.sleep(1)
    except BaseException as exc:
        state['error']=repr(exc);state['status']='ab_error_stopping_dispatch';save()
        if not (WORK/'STOP_REQUESTED').exists():(WORK/'STOP_REQUESTED').write_text('Mix A/B coordinator error: stop dispatch and preserve completed batches.\n')
        if changed and not restored:
            while not restored:
                try:restore()
                except Exception as problem:
                    state['restore_error']=repr(problem);state['status']='retrying_original_property_restore';save();time.sleep(15)
                    try:client.close()
                    except Exception:pass
                    try:client=connect('192.0.2.15')
                    except Exception:pass
        state['status']='failed_original_property_restored' if restored else 'failed_no_property_change';save()
    finally:
        client.close()


if __name__=='__main__':main()
