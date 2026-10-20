"""Re-infer fixed checkpoints and compare single tasks with joint on identical V cells."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine
from src.v49_masked_forecast import training_implementation
from src.cuda_finite_check import noncompacting_finite_check
from src.v61_state_momentum import StateRanker
from src.v63_residual_learning import fit_gain
from src.v64_task_ablation import active_tasks
from scripts.run_v64_task_ablation import bind, load_context, parts_for, raw_outputs, MODELS, REPORT
from scripts.build_v63_residual_targets import sha
from scripts.run_v59_raw_mean import score


def main():
    identity, target = bind(False)
    status = json.loads((REPORT / 'progress.json').read_text())
    assert status['status'] == 'completed_diagnostic_only' and status['completed'] == 9
    assert not status['formal_promotion'] and not status['ppo_training_started']
    frozen = json.loads((REPORT / 'all_E_selected_before_CV.json').read_text())
    assert len(frozen) == 9
    for name, expected in frozen.items():
        assert sha(ROOT / name) == expected
    config = json.loads((ROOT / 'config/v64_task_ablation.json').read_text())
    parent = json.loads((ROOT / config['parent_config']).read_text())
    old = engine.configuration(ROOT / 'config/v45_research.yaml')
    a, _ = load_context()
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    results = pd.read_csv(REPORT / 'V_task_results.csv')
    assert len(results) == 18
    for _, block in results.groupby(['year', 'seed', 'task']):
        assert block.valid_cells.nunique() == 1 and block.active_entities.nunique() == 1
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('cuda')
    assert torch.cuda.is_available()
    audit_rows, predictions = [], {}
    with training_implementation(), noncompacting_finite_check(), torch.no_grad():
        for definition in target['splits']:
            if not definition['eligible']:
                continue
            year = definition['year']
            with np.load(ROOT / f'data_processed/task_ablation_v64/{year}.npz') as z:
                r = {k: z[k].copy() for k in z.files}
            parts = parts_for(r)
            fit, early, ci, vi = [r[k+'_indices'] for k in ['fit', 'early', 'calibration', 'validation']]
            ends = engine.end_times(a)
            assert (ends[fit].max(1) < a['dates'][early[0]]).all()
            assert (ends[early].max(1) < a['dates'][ci[0]]).all()
            assert (ends[ci].max(1) < a['dates'][vi[0]]).all()
            query = np.concatenate([fit, early, ci, vi])
            for t in groups:
                latest = r[t+'_latest_maturity']
                assert np.all((latest == 0) | (latest < a['dates'][query, None]))
            clipped, _ = engine.clip_fold_features(a, fit, old)
            for mode in config['modes']:
                active = active_tasks(mode)
                for seed in config['seeds']:
                    folder = MODELS / str(year) / mode / str(seed)
                    chosen = json.loads((folder / 'selection.json').read_text())
                    assert chosen['identity']['manifest'] == identity
                    assert chosen['checkpoint_chosen_on'] == 'inner_E_only' and not chosen['C_or_V_used_for_selection']
                    losses = pd.read_csv(folder / 'loss.csv')
                    assert not losses.V_opened.any()
                    recorded = losses[losses.epoch == chosen['selected_epoch']].iloc[0]
                    np.testing.assert_allclose(recorded.early_MSE_ratio, chosen['early_MSE_ratio'], atol=1e-10)
                    checkpoint = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=False)
                    assert checkpoint['identity'] == chosen['identity']
                    torch.manual_seed(seed)
                    model = StateRanker(engine.ForecastNetwork(clipped, old), old['d_model'])
                    initial = model.state_dict()
                    for prefix in ['network.core.local_projection', 'network.core.encoder.layers.0', 'network.core.encoder.layers.1']:
                        assert any(k.startswith(prefix) and not torch.equal(v, checkpoint['state'][k]) for k, v in initial.items()), (mode, seed, prefix)
                    for task in groups:
                        for key, value in model.heads[task].state_dict().items():
                            if task not in active:
                                assert torch.equal(value, checkpoint['state']['heads.'+task+'.'+key])
                    model.load_state_dict(checkpoint['state'])
                    model.to(device)
                    raw = raw_outputs(model, clipped, np.concatenate([ci, vi]), old, device, parent, checkpoint['scale'])
                    with np.load(folder / 'validation_predictions.npz') as z:
                        stored = {k: z[k].copy() for k in z.files}
                    predictions[year, mode, seed] = stored
                    for task in groups:
                        gain = fit_gain(raw[task][:len(ci)], r[task+'_residual'][parts['calibration']],
                            r[task+'_mask'][parts['calibration']], groups[task], ridge=parent['calibration_ridge']) if task in active else np.zeros_like(checkpoint['scale'][task])
                        np.testing.assert_allclose(gain, stored[task+'_gain'], atol=1e-6, rtol=1e-4)
                        inc = raw[task][len(ci):]*gain if task in active else np.zeros_like(raw[task][len(ci):])
                        base = r[task+'_base'][parts['validation']]
                        np.testing.assert_allclose(inc, stored[task+'_increment'], atol=1e-6, rtol=1e-4)
                        np.testing.assert_allclose(base+inc, stored[task], atol=1e-6, rtol=1e-4, equal_nan=True)
                        masks = r[task+'_mask'][parts['validation']]
                        truth = a[task+'_raw_relative_target'][vi]
                        ratio, _ = score({task: stored[task]}, {task: base}, {task: truth}, {task: masks}, groups)
                        row = results[(results.year == year) & (results['mode'] == mode) & (results.seed == seed) & (results.task == task)].iloc[0]
                        np.testing.assert_allclose(ratio, row.V_MSE_ratio, atol=1e-10)
                        if task not in active:
                            assert not stored[task+'_increment'].any()
                            np.testing.assert_allclose(ratio, 1.)
                    audit_rows.append(dict(year=year, mode=mode, seed=seed, E_selection_verified=True,
                        inactive_head_preserved=True, selected_core_updated=True, C_gain_reproduced=True, V_means_reproduced=True))
                    del model, checkpoint
                    torch.cuda.empty_cache()
                    print(json.dumps(audit_rows[-1]), flush=True)
    comparisons = []
    for year in results.year.unique():
        for task, single in [('asset', 'asset_only'), ('sleeve', 'sleeve_only')]:
            for seed in config['seeds']:
                block = results[(results.year == year) & (results.seed == seed) & (results.task == task)].set_index('mode')
                comparisons.append(dict(year=int(year), task=task, seed=seed,
                    joint_MSE_ratio=float(block.loc['joint', 'V_MSE_ratio']),
                    single_MSE_ratio=float(block.loc[single, 'V_MSE_ratio']),
                    single_minus_joint_MSE=float(block.loc[single, 'V_MSE_ratio']-block.loc['joint', 'V_MSE_ratio']),
                    single_minus_joint_delta_IC=float(block.loc[single, 'V_delta_rank_ic']-block.loc['joint', 'V_delta_rank_ic']),
                    same_support=True, exploratory_single_period=True))
    pd.DataFrame(comparisons).to_csv(REPORT / 'paired_task_comparison.csv', index=False)
    pd.DataFrame(audit_rows).to_csv(REPORT / 'inference_audit.csv', index=False)
    summary = results[results.trained].groupby(['year', 'mode', 'task'], as_index=False)[['V_MSE_ratio', 'V_delta_rank_ic']].mean()
    summary.to_csv(REPORT / 'three_seed_task_summary.csv', index=False)
    bind(False)
    artifact = dict(passed=True, reinferred_models=9, inactive_heads_preserved=True,
        early_selection_before_CV=True, paired_task_support_equal=True, source_hashes_passed=True,
        formal_promotion=False, ppo_training_started=False, independent_OOS_claim=False,
        audit_script_sha256=sha(Path(__file__)))
    (REPORT / 'artifact_audit.json').write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    lines = ['# V64 多任务消融完成报告', '',
        '9组完整训练完成，E选权重、C校准、V诊断；9组实际推断与存档复现通过。单一已观察2022年研究V，不是全新样本外。', '',
        '| 模式 | 实际训练任务 | 三种子平均误差比 | 三种子平均RankIC增量 |', '| --- | --- | ---: | ---: |']
    for _, row in summary.iterrows():
        lines.append(f"| {row['mode']} | {row.task} | {row.V_MSE_ratio:.6f} | {row.V_delta_rank_ic:+.6f} |")
    lines += ['', '这里平均的是三个种子的指标，不是集成收益或组合夏普。比较单任务与joint必须在同一task、同一日期标签支持上进行。详见paired_task_comparison.csv，所有种子完整保留。', '',
        '固定实验结束即停止，不以获胜模式晋升，不继续改参数，不启动正式PPO。38资产与九宫格保留，现金流/执行认证限制不变。']
    (REPORT / 'completion_review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(artifact), flush=True)


if __name__ == '__main__':
    main()
