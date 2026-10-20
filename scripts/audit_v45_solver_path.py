"""Reconstruct v44 monthly inputs and audit strict v45 sequential feasibility.

The repaired prefix stops at the first conflict. Later archived-transition
probes are diagnostics only, never deployable targets or continued backtests.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.v1_strategy import (load_config, apply_selected_profile_overrides,
    V1Paths, build_sleeve_returns, month_end_views, calibrate_market_view_confidence,
    load_itransformer_predictions, _build_overlay_sleeve_weights, sleeve_bounds)
from src.ep_skfolio_optimizer import (RISK_SLEEVES, effective_ep_config,
    build_prior_mu, posterior_from_views, trend_scores)
from src.rienet_covariance import estimate_covariance
from src.v45_solver import solve, minimum_turnover, causal_anchor, SolverContractError, CONTRACT_VERSION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor-policy',choices=['strict_raw','causal_projection'],default='strict_raw')
    args=parser.parse_args()
    prefix='v45_anchor_smoothing' if args.anchor_policy=='causal_projection' else 'v45_solver'
    reports=ROOT/'output/reports'
    paths=V1Paths(returns=ROOT/'data_processed/v44_frozen_returns_monthly.parquet',
        views=ROOT/'data_processed/v44_mechanical_views_daily.parquet',
        market_state_cards=ROOT/'data_processed/v44_state_cards_daily.parquet')
    rows, targets, conflicts, profiles, anchor_rows=[],[],[],[],[]
    dependencies=[paths.returns,paths.views,paths.market_state_cards,
        ROOT/'config/v1_strategy_config.yaml', ROOT/'src/v45_solver.py']
    for profile in ['low_vol','balanced','high_vol']:
        cfg=copy.deepcopy(load_config())
        cfg['product']['selected_profile']=profile
        cfg['strategy_version']='no_agent'
        cfg['synthetic_assets']['rate_30y_curve_proxy']['enabled']=False
        cfg=apply_selected_profile_overrides(cfg)
        ep=effective_ep_config(cfg)
        ret=build_sleeve_returns(paths,cfg)
        views=calibrate_market_view_confidence(month_end_views(paths,cfg),ret,cfg)
        pred=load_itransformer_predictions(cfg,paths)
        base_path=ROOT/f'data_processed/v44_{profile}_risk_budget_base_monthly.parquet'
        archived_path=ROOT/f'data_processed/v44_{profile}_sleeve_weights_monthly.parquet'
        dependencies.extend([base_path,archived_path])
        base=pd.read_parquet(base_path)
        overlay=_build_overlay_sleeve_weights(base,views,ret,pred,cfg)
        archived=pd.read_parquet(archived_path).set_index('date')
        # The legacy config includes cash:(0,0); v45 solves five sleeves only.
        all_bounds=sleeve_bounds(cfg)
        if all_bounds.get('cash',(0,0)) != (0,0):
            raise ValueError('v45 no-cash contract does not permit nonzero cash bounds')
        bounds={s:all_bounds[s] for s in RISK_SLEEVES}
        lo=np.array([bounds[s][0] for s in RISK_SLEEVES])
        hi=np.array([bounds[s][1] for s in RISK_SLEEVES])
        previous=None
        first_conflict=None
        for row in overlay.itertuples(index=False):
            date=pd.Timestamp(row.date)
            b=pd.Series({s:getattr(row,s) for s in RISK_SLEEVES})
            hist=ret.loc[:date].tail(int(ep['cov_lookback_months']))
            if len(hist)<int(ep['min_history_months']):
                previous=b.copy()
                rows.append(dict(profile=profile,date=date,mode='warmup',status='not_enough_history'))
                continue
            mu,_=posterior_from_views(build_prior_mu(hist,cfg),
                views if cfg['signal_overlays']['view_application'].get('ep_mean_enabled',True) else views.iloc[:0],
                pred,trend_scores(hist,cfg),date,cfg)
            cov=estimate_covariance(hist,RISK_SLEEVES,ep,
                default_shrinkage=float(ep['covariance_shrinkage_to_diag']),
                floor=float(cfg['risk_budget']['diagonal_vol_floor'])**2,
                jitter=float(ep['cov_jitter'])).covariance
            # Diagnose each old transition without treating it as a valid path.
            older=archived.loc[archived.index<date].tail(1)
            p_old=b if older.empty else older.iloc[0][RISK_SLEEVES].astype(float)
            probe=dict(profile=profile,date=date,mode='archived_transition_probe')
            try:
                needed,_=minimum_turnover(b.to_numpy(),p_old.to_numpy(),lo,hi,float(ep['max_active_deviation']))
                probe.update(minimum_required_l1=needed,configured_l1_limit=float(ep['max_sleeve_turnover']),
                    status='feasible' if needed<=float(ep['max_sleeve_turnover'])+1e-6 else 'infeasible')
            except SolverContractError as exc:
                probe.update(exc.detail)
            rows.append(probe)
            if first_conflict is not None:
                continue
            try:
                effective=b.copy()
                anchor_diag={}
                if args.anchor_policy=='causal_projection':
                    projected=causal_anchor(b,previous if previous is not None else b,bounds,
                        float(ep['max_active_deviation']),float(ep['max_sleeve_turnover']))
                    effective,anchor_diag=projected.weights,projected.diagnostics
                result=solve(effective,mu,cov,previous,bounds,hist,ep)
                rows.append(dict(profile=profile,date=date,mode='strict_sequential_prefix',**result.diagnostics,**anchor_diag))
                anchor_rows.extend(dict(profile=profile,date=date,sleeve=s,
                    raw_anchor=float(b[s]),effective_anchor=float(effective[s]),**anchor_diag)
                    for s in RISK_SLEEVES)
                previous=result.weights.copy()
                targets.extend(dict(profile=profile,date=date,sleeve=s,weight=float(w))
                    for s,w in result.weights.items())
            except SolverContractError as exc:
                first_conflict=str(date.date())
                conflict=dict(profile=profile,date=first_conflict,mode='strict_sequential_prefix',**exc.detail)
                rows.append(conflict); conflicts.append(conflict)
                print(json.dumps(conflict,ensure_ascii=False),flush=True)
        profiles.append(dict(profile=profile,first_conflict=first_conflict,
            prefix_only=first_conflict is not None))
        print(profile,'scan complete; first_conflict=',first_conflict,flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(reports/f'{prefix}_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(targets).to_parquet(ROOT/f'data_processed/{prefix}_validated_targets.parquet',index=False)
    pd.DataFrame(anchor_rows).to_csv(reports/f'{prefix}_anchor_ledger.csv',index=False,encoding='utf-8-sig')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    state=dict(contract=CONTRACT_VERSION,stage='solver',anchor_policy=args.anchor_policy,passed=not conflicts,
        profiles=profiles,conflicts=conflicts,training_started=False,
        next_stage_allowed=not conflicts,inputs_sha256=hashes,
        scan_scope='monthly core solver reconstructed from frozen v44 inputs; no post-solver rule-speed/daily guardrails or asset-path certification',
        probe_note='archived_transition_probe does not continue the repaired backtest')
    (reports/f'{prefix}_stage_gate.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# v45 第零阶段执行结果','',
        '本次是求解器实现与可行性扫描，不是iTransformer/PPO正式训练。',
        f'全历史顺序路径通过：{not conflicts}。','',
        '## 三产品首个阻断点','',pd.DataFrame(profiles).to_markdown(index=False),'',
        '## 扫描范围','',
        '仅认证冻结v44输入的月度核心求解器路径。未完成日频guardrails、最终资产分配全路径、数据发布日台账和训练阶段。低波/高波此处通过不等于整个v45通过。','',
        '## 冲突证据','',
        pd.DataFrame(conflicts)[['profile','date','status','minimum_required_l1','configured_l1_limit','active_limit','anchor_previous_l1']].to_markdown(index=False) if conflicts else '无顺序路径冲突。','',
        '## 处理原则','',
        '不放宽额度、不跳过日期、不把已验证前缀称为完整回测。存在阻断点时后续数据/模型/训练阶段不启动。',
        'archived_transition_probe仅检查旧权重过渡的数学可行性，不生成可部署目标。',
        f'详细证据见{prefix}_audit.csv和{prefix}_stage_gate.json。','',
        '## 锚点政策','',
        f'本轮政策：{args.anchor_policy}。用户2026-10-15要求解决冲突，新增因果锚点投影，不调整总L1或主动偏离上限。原strict_raw失败证据另存。',
        '原始锚点与有效锚点逐日保留；策略行为改变后的业绩必须重新计算，不能复用v44业绩证明改善。',
        '不能以此保证所有后续日期都可行；若仍有资格/上下限冲突，继续输出冲突而非自动豁免。',
        '另一种方案是明确允许底仓变动时的换手例外，但这意味着8%不再是所有日期的硬上限。',
        '仅本阶段通过不能代表后续数据、模型或全策略接入已完成。']
    (reports/f'{prefix}_stage_results.md').write_text('\n'.join(lines),encoding='utf-8')
    if conflicts:
        raise SystemExit(2)


if __name__=='__main__': main()
