"""Execute the predeclared diagnostics after the primary run releases the model."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from openrubrics_lab.io import file_hash, now, write_json

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / 'experiments/m1-cpu-training-20261116'
OUT = ROOT / 'experiments/supplementary'

def main():
    primary_pid = int(sys.argv[1])
    while True:
        m = json.loads((PRIMARY / 'manifest.json').read_text())
        if m.get('status') == 'completed_small_scale_protocol':
            break
        if any(s['status'] == 'failed' for s in m['stages'].values()):
            raise RuntimeError('Primary protocol failed; supplementary jobs were not started')
        try:
            os.kill(primary_pid, 0)
        except ProcessLookupError:
            raise RuntimeError('Primary process exited before completion') from None
        time.sleep(10)
    python = sys.executable
    jobs = [
        ('long-decode-sft', [python, '-m', 'openrubrics_lab', 'evaluate',
         '--config', str(OUT/'long-decode-config.json'), '--data', str(OUT/'validation-pairs.jsonl'),
         '--rubric-adapter', str(PRIMARY/'02-sft-rubric'), '--judge-adapter', str(PRIMARY/'03-sft-judge'),
         '--output', str(OUT/'long-decode-sft')]),
        ('long-two-stage-sft', [python, '-m', 'openrubrics_lab', 'evaluate',
         '--config', str(OUT/'long-two-stage-config.json'), '--data', str(OUT/'validation-pairs.jsonl'),
         '--rubric-adapter', str(PRIMARY/'02-sft-rubric'), '--judge-adapter', str(PRIMARY/'03-sft-judge'),
         '--output', str(OUT/'long-two-stage-sft')]),
    ]
    checkpoints = m['stages']['07-arm']['result']
    for name, rubric, judge in [('sft', PRIMARY/'02-sft-rubric', PRIMARY/'03-sft-judge'),
                                ('arm', ROOT/checkpoints['rubric'], ROOT/checkpoints['judge'])]:
        jobs.append((f'order-{name}', [python, '-m', 'openrubrics_lab', 'evaluate',
            '--data', str(OUT/'swapped-validation-pairs.jsonl'),
            '--rubric-adapter', str(rubric), '--judge-adapter', str(judge),
            '--output', str(OUT/f'order-{name}')]))
    for name in ('helpsteer3', 'ppe-ifeval', 'ifbench'):
        jobs.append((f'eval-{name}', [python, '-m', 'openrubrics_lab', 'evaluate',
            '--data', str(OUT/'additional-data-v2'/f'{name}.jsonl'),
            '--rubric-adapter', str(ROOT/checkpoints['rubric']),
            '--judge-adapter', str(ROOT/checkpoints['judge']),
            '--output', str(OUT/f'eval-{name}')]))
    policies = [('base', None), ('source-dpo', PRIMARY/'13-dpo-source-baseline'),
                ('online-grpo', PRIMARY/'15-online-grpo')]
    if m['stages']['12-dpo-arm']['result'].get('steps', 0):
        policies.append(('arm-dpo', PRIMARY/'12-dpo-arm/adapter'))
    iterative = json.loads((PRIMARY/'14-iterative-dpo/summary.json').read_text())
    if iterative['final_adapter']:
        policies.append(('iterative-dpo', ROOT/iterative['final_adapter']))
    write_json(OUT/'policy-variants.json', {'policies': [
        {'name': name, 'adapter': str(adapter) if adapter else None} for name, adapter in policies],
        'rule': 'Evaluate every available trained policy; empty ARM/IterDPO stages have no policy score.'})
    for name, adapter in policies:
        cmd = [python, 'scripts/evaluate_ifeval.py', '--samples', '4', '--max-tokens', '512',
               '--output', str(OUT/f'ifeval-{name}')]
        if adapter:
            cmd += ['--adapter', str(adapter)]
        jobs.append((f'ifeval-{name}', cmd))
    jobs.append(('analysis', [python, str(OUT/'analyze.py')]))
    jobs.append(('demo-recording', [python, 'scripts/record_demo.py', '--run', str(PRIMARY)]))
    receipt = {'started_utc': now(), 'jobs': [],
               'code_hashes': {p.name: file_hash(p) for p in sorted(OUT.glob('*.py'))}}
    for name, command in jobs:
        item = {'name': name, 'command': command, 'started_utc': now(), 'status': 'running'}
        receipt['jobs'].append(item)
        write_json(OUT/'execution.json', receipt)
        print(f'=== Supplementary: {name} ===', flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, check=False)
        item.update(status='completed' if result.returncode == 0 else 'failed',
                    seconds=time.perf_counter()-started, returncode=result.returncode)
        write_json(OUT/'execution.json', receipt)
        if result.returncode:
            raise SystemExit(result.returncode)
    receipt['finished_utc'] = now()
    write_json(OUT/'execution.json', receipt)

if __name__ == '__main__':
    main()
