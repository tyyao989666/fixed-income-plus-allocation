"""Independent completed-run accounting, coverage and gradient-state review."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine
from src.v61_state_momentum import StateRanker
from scripts.build_v63_residual_targets import sha
from scripts.run_v62_bridge_diagnostic import metric_rows
from scripts.run_v59_raw_mean import score
from scripts.run_v61_state_ranker import rank_metrics
from src.research_rank_metrics import rank_metrics_named


def main():
    out = ROOT / 'output/reports/residual_learning_v63'
    manifest = json.loads((out / 'training_manifest.json').read_text())
    for name, expected in manifest.items():
        path = ROOT / ('data_processed/corporate_action_v58/dataset.npz' if name == 'dataset_sha256' else name)
        assert sha(path) == expected, name
    status = json.loads((out / 'progress.json').read_text())
    assert status['status'] == 'completed_validation_only' and status['completed'] == 6
    decision = json.loads((out / 'decision.json').read_text())
    assert not decision['formal_promotion'] and not decision['ppo_training_started'] and not decision['G_or_test_used']
    cfg = json.loads((ROOT / 'config/v63_residual_learning.json').read_text())
    old = engine.configuration(ROOT / 'config/v45_research.yaml')
    with np.load(ROOT / 'data_processed/corporate_action_v58/dataset.npz') as z:
        a = {k: z[k].copy() for k in z.files}
    with np.load(ROOT / 'data_processed/state_momentum_v61/signals.npz') as z:
        signals = {k: z[k].copy() for k in z.files}
    extras = np.stack([2*signals['cohort']-1, np.tanh(signals['extension']/3), signals['value_boundary']], -1)
    a['local_values'] = np.concatenate([a['local_values'], np.nan_to_num(extras)], -1).astype(np.float32)
    a['local_masks'] = np.concatenate([a['local_masks'], np.isfinite(extras)], -1)
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    rows, gradients, source_rows, corrections = [], [], [], []
    torch.set_num_threads(2)
    for year in cfg['years']:
        with np.load(ROOT / f'data_processed/residual_learning_v63/{year}.npz') as z:
            r = {k: z[k].copy() for k in z.files}
        ti, ci, vi = [r[k + '_indices'] for k in ['train', 'calibration', 'validation']]
        query = np.concatenate([ti, ci, vi])
        vp = slice(len(ti)+len(ci), None)
        truth = {t: a[t + '_raw_relative_target'][vi] for t in groups}
        baseline = {t: r[t+'_base'][vp] for t in groups}
        masks = {t: r[t+'_mask'][vp] for t in groups}
        controls = {'historical_control': baseline}
        for name in ['momentum20', 'cohort', 'price_soft']:
            x, w = signals[name], a['sleeve_reference_weights']
            sleeve = np.sum(w*np.where(np.isfinite(x[:, None]), x[:, None], 0), -1)
            good = (w.sum(-1) > 0) & ((w == 0) | np.isfinite(x[:, None])).all(-1)
            controls[name] = dict(asset=np.repeat(x[:, None], 3, 1)[vi], sleeve=np.repeat(np.where(good, sleeve, np.nan)[:, None], 3, 1)[vi])
        common = {t: masks[t].copy() for t in groups}
        for base in controls.values():
            for task in groups:
                common[task] &= np.isfinite(base[task])
        values = []
        for task in groups:
            latest = r[task+'_latest_maturity']
            assert np.all((latest == 0) | (latest < a['dates'][query, None]))
            assert np.isfinite(r[task+'_scale']).all() and (r[task+'_scale'] > 0).all()
            source_rows.append(dict(year=year, task=task, maturity_passed=True,
                source_last_day=str(pd.Timestamp(a['dates'][ti[-1]]).date()),
                validation_start=str(pd.Timestamp(a['dates'][vi[0]]).date()),
                validation_end=str(pd.Timestamp(a['dates'][vi[-1]]).date())))
        for seed in cfg['seeds']:
            folder = ROOT / f'output/models/residual_learning_v63/{year}/{seed}'
            summary = json.loads((folder / 'summary.json').read_text())
            assert summary['identity']['manifest'] == manifest
            assert summary['validation_not_tradeable'] and not summary['formal_promotion']
            selected = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=False)
            current = torch.load(folder / 'current.pt', map_location='cpu', weights_only=False)
            assert selected['identity'] == current['identity'] == summary['identity']
            torch.manual_seed(seed)
            model = StateRanker(engine.ForecastNetwork(a, old), old['d_model'])
            named = dict(model.named_parameters())
            # Adam's serialized parameter IDs follow the explicit head/core groups.
            head_names = [k for k, _ in model.named_parameters() if k.startswith('heads.')]
            core_names = ['network.core.' + k for k, _ in model.network.core.named_parameters()]
            mapping = {}
            for group, names in zip(current['optimizer']['param_groups'], [head_names, core_names]):
                assert len(group['params']) == len(names)
                mapping.update(dict(zip(group['params'], names)))
            for label, prefix in [('local', 'network.core.local_projection'), ('layer_0', 'network.core.encoder.layers.0'), ('layer_1', 'network.core.encoder.layers.1'), ('heads', 'heads.')]:
                evidence = [v['exp_avg'].abs().sum().item() for key, v in current['optimizer']['state'].items() if mapping[key].startswith(prefix)]
                assert evidence and sum(evidence) > 0, (year, seed, label)
                changed = [name for name in named if name.startswith(prefix) and not torch.equal(named[name].detach(), selected['state'][name])]
                assert changed, (year, seed, label, 'selected parameters unchanged')
                gradients.append(dict(year=year, seed=seed, component=label, optimizer_moment_abs_sum=sum(evidence), selected_changed_parameters=len(changed)))
            with np.load(folder / 'validation_predictions.npz') as z:
                pred = {k: z[k].copy() for k in z.files}
            for task in groups:
                gain, scale = selected['gain'][task], selected['scale'][task]
                assert np.isfinite(gain).all() and ((gain >= 0) & (gain <= 1)).all()
                np.testing.assert_array_equal(scale, r[task+'_scale'])
                np.testing.assert_allclose(pred[task], baseline[task] + pred[task+'_increment'], atol=1e-8, equal_nan=True)
                limit = cfg['increment_bound_sigma']*scale*gain
                assert np.all(np.abs(pred[task+'_increment']) <= limit[None]+1e-7)
            mean = {t: pred[t] for t in groups}
            ratio, delta = score(mean, baseline, truth, masks, groups)
            np.testing.assert_allclose(ratio, summary['validation_MSE_ratio'], atol=1e-10)
            np.testing.assert_allclose(delta, summary['validation_delta_rank_ic'], atol=1e-10)
            values.append(pred)
            del model, current, selected
        mean = {t: np.mean([p[t] for p in values], 0) for t in groups}
        ratio, _ = score(mean, baseline, truth, masks, groups)
        minimum, delta, legacy_ic = rank_metrics(mean, controls, truth, common, groups)
        fixed_minimum, fixed_delta, ic = rank_metrics_named(mean, controls, truth, common, groups, reference='historical_control')
        np.testing.assert_allclose(fixed_minimum, minimum, atol=1e-10)
        for key in delta:
            np.testing.assert_allclose(fixed_delta[key], delta[key], atol=1e-10)
        detail = next(d for d in decision['details'] if d['year'] == year)
        np.testing.assert_allclose([ratio, minimum], [detail['own_support_MSE_ratio'], detail['min_rank_increment']], atol=1e-10)
        assert detail['common_support_rank_ic'] is None and not np.isfinite(legacy_ic)
        assert np.isfinite(ic)
        corrections.append(dict(year=year, original_absolute_rank_ic=None, corrected_absolute_rank_ic=ic,
            MSE_ratio_unchanged=ratio, minimum_increment_unchanged=minimum,
            qualification_changed=False, convention='legacy_group_weighted_absolute_IC; do not mix with equal_task_component_summary'))
        rows.extend(metric_rows(year, 'residual_mean', mean, baseline, truth, masks, controls, groups))
    pd.DataFrame(rows).to_csv(out / 'coverage_component_metrics.csv', index=False)
    pd.DataFrame(gradients).to_csv(out / 'full_encoder_gradient_audit.csv', index=False)
    pd.DataFrame(source_rows).to_csv(out / 'maturity_audit.csv', index=False)
    correction = dict(original_decision_sha256=sha(out / 'decision.json'),
        audit_script_sha256=sha(Path(__file__)), helper_sha256=sha(ROOT / 'src/research_rank_metrics.py'),
        original_automatic_audit_failed=True, original_audit_exit_preserved=True,
        cause='legacy rank_metrics collected absolute IC only for a baseline literally named prior',
        changes=corrections, gates_unchanged=True, retraining=False)
    (out / 'reporting_correction.json').write_text(json.dumps(correction, indent=2), encoding='utf-8')
    result = dict(passed=True, completed_models=6, full_encoder_gradient_components=len(gradients),
        source_hashes_passed=True, means_reconstructed=True, bounded_raw_return_units=True,
        original_MSE_and_delta_metrics_reproduced=True, absolute_IC_reporting_repaired=True,
        original_failed_audit_preserved=True, exploratory_validation_only=True,
        formal_promotion=False, ppo_training_started=False)
    save_path = out / 'artifact_audit.json'
    save_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# V63 成熟标签残差学习完成报告', '',
        '6组完整监督训练完成，独立产物复核通过。V为已观察研究验证，不是全新样本外；未训练PPO，未产生新的三产品收益或夏普。', '',
        '| 折编号 | 收益MSE/基准MSE | 四对照共同覆盖最小RankIC增量 |', '| --- | ---: | ---: |']
    for d in decision['details']:
        lines.append(f"| {d['year']} | {d['own_support_MSE_ratio']:.6f} | {d['min_rank_increment']:+.6f} |")
    lines += ['', '下一阶段研究资格：' + str(decision['research_followup_allowed']) + '。', '',
        '失败项：' + '、'.join(k for k, v in decision['checks'].items() if not v), '',
        '不同覆盖口径不得混比。coverage_component_metrics.csv单列模型自身、每个基准成对及四基准共同覆盖。训练数据和现金流/执行认证限制继续保留；本轮完成即停止，不自动增加实验或放松门槛。', '',
        '原自动审计退出码1：旧报告函数仅当基准名为prior时累计绝对RankIC，V63使用historical_control故原字段为空。修正脚本显式传入参考名，重算并确认MSE、相对增量及门控不变；原decision.json、audit_exit.json与失败日志均保留。修正详情见reporting_correction.json。', '',
        '| 折编号 | 修正后共同覆盖绝对RankIC（沿用旧权重） |', '| --- | ---: |']
    for c in corrections:
        lines.append(f"| {c['year']} | {c['corrected_absolute_rank_ic']:.6f} |")
    (out / 'completion_review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
