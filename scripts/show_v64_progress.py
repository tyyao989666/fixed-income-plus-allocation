import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_v45_research_pipeline import process_alive


def main():
    out = ROOT / 'output/reports/task_ablation_v64'
    path = out / 'progress.json'
    data = json.loads(path.read_text()) if path.exists() else dict(status='not_started')
    data['pid_alive'] = process_alive(data.get('pid'))
    data['selected'] = []
    for path in sorted((ROOT / 'output/models/task_ablation_v64').glob('*/*/*/selection.json')):
        row = json.loads(path.read_text())
        data['selected'].append({k: v for k, v in row.items() if k != 'identity'})
    for key in ['process_exit', 'audit_exit', 'artifact_audit']:
        path = out / (key+'.json')
        if path.exists():
            data[key] = json.loads(path.read_text(encoding='utf-8-sig'))
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
