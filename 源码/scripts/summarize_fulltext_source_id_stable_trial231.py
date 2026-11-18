"""Summarize bounded trial evidence; reads local files only."""
from datetime import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'outputs/fulltext_source_id_stable_trial_231_20261115'


def ts(value):
    return datetime.fromisoformat(value).timestamp()


def main():
    source=json.loads((ROOT/'status.json').read_text(encoding='utf-8'))
    samples=[json.loads(line) for line in (ROOT/'host_observations.jsonl').read_text(encoding='utf-8').splitlines() if line]
    result={'source_status':source['status'],'source_updated_at':source['at'],'stages':[],
            'note':'Full readiness wall time includes historical patch export and count checks; regular continuous operation does not export all historical patches every ten minutes.'}
    for stage in source['stages']:
        trial=(stage.get('runner_final') or {}).get('trial') or stage.get('progress')
        if not trial:continue
        entry={'ordinal':stage['ordinal'],'rate_ceiling':stage['rate_ceiling'],'trial':trial,
               'index_drained_at':stage.get('index_drained_at')}
        end=stage.get('index_drained_at')
        begin=trial.get('started_at')
        if begin:
            stop=ts(end) if end else ts(source['at'])
            begin=ts(begin)
            observations=[s for s in samples if begin<=ts(s['at'])<=stop and 'host_cpu_percent' in s]
            if observations:
                entry['host_cpu_peak_percent']=max(s['host_cpu_percent'] for s in observations)
                entry['host_cpu_mean_sample_percent']=sum(s['host_cpu_percent'] for s in observations)/len(observations)
                for name in ['index_pending','index_running','compaction_pending','compaction_running']:
                    entry[name+'_peak']=max(s.get(name,0) for s in observations)
            if end:
                rows=trial['new_counts']['newly_filled']
                wall=stop-begin
                entry['full_wall_seconds_until_background_ready']=wall
                entry['verified_rows_per_second_until_background_ready']=rows/max(wall,1)
                entry['post_write_additional_wall_seconds']=wall-trial['elapsed_seconds_including_inflight_waits']
                entry['counts_after']=stage.get('counts_after')
        result['stages'].append(entry)
    target=ROOT/'stage_performance_summary.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
