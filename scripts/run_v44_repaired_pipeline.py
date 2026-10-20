"""Verify/rebuild repaired v44; --train-ppo explicitly runs new policies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'config/hierarchical_asset_itransformer_v44_accounting_fixed.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-ppo', action='store_true', help='Train new policies under corrected reward/accounting')
    parser.add_argument('--baselines-only', action='store_true')
    args = parser.parse_args()
    stages = [
        ('regression_tests',['scripts/176_test_accounting_repairs.py']),
        ('rebuild_frozen_mechanical_anchors',['scripts/177_prepare_v44_repaired_anchors.py']),
        ('recompute_A0_A1',['scripts/171_run_v43_control_variable_baselines.py']),
    ]
    if not args.baselines_only:
        stages.append(('train_new_PPO' if args.train_ppo else 'replay_signed_PPO_or_mechanical_fallback',
                       ['scripts/136_run_hierarchical_asset_itransformer_ppo_v7.py','--config',CONFIG]
                       + ([] if args.train_ppo else ['--skip-training'])))
        stages.append(('validate_repaired_outputs',['scripts/178_validate_v44_repaired_outputs.py']))
    rows = []
    for name,command in stages:
        print(f'[{datetime.now():%H:%M:%S}] starting {name}',flush=True)
        result = subprocess.run([sys.executable,'-X','utf8',*command],cwd=ROOT)
        rows.append({'stage':name,'command':command,'returncode':result.returncode,'finished':datetime.now().isoformat(timespec='seconds')})
        (ROOT/'output/reports/v44_accounting_fixed_pipeline_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        if result.returncode:
            raise SystemExit(f'Stopped at {name}; exit={result.returncode}')
    print('Completed. Original v43 artifacts retained; v44 outputs are separate.',flush=True)


if __name__=='__main__':
    main()
