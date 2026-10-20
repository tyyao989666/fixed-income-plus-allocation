"""Read actual V63 process identity, per-task epochs and saved status."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_v45_research_pipeline import process_alive


def main():
    report = ROOT / 'output/reports/residual_learning_v63'
    progress = report / 'progress.json'
    state = json.loads(progress.read_text()) if progress.exists() else {'status': 'not_started'}
    if 'pid' in state:
        state['pid_alive'] = process_alive(state['pid'])
    state['tasks'] = []
    for year in [2023, 2024]:
        for seed in [17, 29, 47]:
            folder = ROOT / f'output/models/residual_learning_v63/{year}/{seed}'
            row = dict(year=year, seed=seed, completed=(folder / 'summary.json').exists())
            if row['completed']:
                data = json.loads((folder / 'summary.json').read_text())
                row.update({k: data[k] for k in ['selected_epoch', 'trained_epochs', 'validation_MSE_ratio']})
            state['tasks'].append(row)
    exit_path = report / 'process_exit.json'
    if exit_path.exists():
        state['process_exit'] = json.loads(exit_path.read_text(encoding='utf-8-sig'))
    for key, filename in [('original_automatic_audit_exit', 'audit_exit.json'),
                          ('postfix_artifact_audit', 'artifact_audit.json'),
                          ('selected_inference_audit', 'inference_audit.json')]:
        path = report / filename
        if path.exists():
            state[key] = json.loads(path.read_text(encoding='utf-8-sig'))
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
