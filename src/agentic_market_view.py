from __future__ import annotations

import json
import operator
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any, TypedDict

import numpy as np
import pandas as pd
import yaml
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.market_skill import (
    MarketSkillPaths,
    _base_view,
    _compact_card_for_prompt,
    _extract_json_object,
    _glm_chat_completion,
    _json_dumps,
    _json_loads,
    _view_asset_allowed,
    build_ep_views,
    generate_executable_mechanical_views,
    load_config as load_market_skill_config,
    validate_view,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "agentic_market_view.yaml"


class ExpertFinding(BaseModel):
    expert: str
    conclusion: str
    relevant_assets: list[str] = Field(default_factory=list)
    direction: str = "neutral"
    confidence: float = 0.5
    evidence: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    fact_observations: list[str] = Field(default_factory=list)
    inference_chain: list[str] = Field(default_factory=list)
    contrarian_checks: list[str] = Field(default_factory=list)
    suggested_view_candidates: list[str] = Field(default_factory=list)
    direction_strength: float = 0.5
    evidence_quality: float = 0.5
    cross_module_confirmation: float = 0.5
    contradiction_penalty: float = 0.0
    liquidity_penalty: float = 0.0

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @field_validator("direction_strength", "evidence_quality", "cross_module_confirmation", "contradiction_penalty", "liquidity_penalty")
    @classmethod
    def clamp_score(cls, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))


class ReviewedFinding(BaseModel):
    expert: str
    status: str
    confidence: float
    accepted_evidence: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    sufficiency_score: float = 0.0
    missing_checks: list[str] = Field(default_factory=list)
    usable_for_view: bool = False
    direction_strength: float = 0.5
    evidence_quality: float = 0.5
    cross_module_confirmation: float = 0.5
    contradiction_penalty: float = 0.0
    liquidity_penalty: float = 0.0


def _evidence_calibrated_confidence(
    direction_strength: float,
    evidence_quality: float,
    confirmation: float,
    contradiction_penalty: float,
    liquidity_penalty: float,
    lo: float = 0.45,
    hi: float = 0.75,
) -> float:
    raw = (
        0.35 * float(direction_strength)
        + 0.25 * float(evidence_quality)
        + 0.20 * float(confirmation)
        - 0.15 * float(contradiction_penalty)
        - 0.05 * float(liquidity_penalty)
    )
    return float(np.clip(raw, lo, hi))


class ResearchTask(BaseModel):
    expert: str
    main_question: str
    research_goal: str
    required_checks: list[str] = Field(default_factory=list)
    asset_focus: list[str] = Field(default_factory=list)
    expected_output: str = ""


class MarketThesisPlan(BaseModel):
    as_of_date: str
    dominant_contradictions: list[str] = Field(default_factory=list)
    regime_hypotheses: list[str] = Field(default_factory=list)
    cross_asset_questions: list[str] = Field(default_factory=list)
    expert_tasks: list[ResearchTask] = Field(default_factory=list)
    do_not_use: list[str] = Field(default_factory=list)
    fallback_required: bool = False


class AgentView(BaseModel):
    source_asset: str
    target_asset: str
    relation: str = "overweight"
    confidence: float
    horizon_days: int = 20
    evidence_ids: list[str] = Field(default_factory=list)
    reason_tags: list[str] = Field(default_factory=list)
    model_alignment: str = "mixed"
    risk_flags: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    direction_strength: float = 0.5
    evidence_quality: float = 0.5
    cross_module_confirmation: float = 0.5
    contradiction_penalty: float = 0.0
    liquidity_penalty: float = 0.0

    @field_validator("confidence", "direction_strength", "evidence_quality", "cross_module_confirmation", "contradiction_penalty", "liquidity_penalty")
    @classmethod
    def clamp_score(cls, value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))


def _mechanical_candidate_views_for_prompt(card: pd.Series, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    filters = _json_loads(card["view_filters"])
    candidates: list[dict[str, Any]] = []
    for idx, view in enumerate(generate_executable_mechanical_views(card, cfg), start=1):
        status, reason = validate_view(view, filters, cfg)
        tags = view.get("reason_tags", [])
        if isinstance(tags, str):
            tags = _json_loads(tags)
        candidates.append(
            {
                "candidate_id": f"mechanical_{idx}",
                "source_asset": view.get("source_asset"),
                "target_asset": view.get("target_asset"),
                "relation": view.get("relation", "overweight"),
                "confidence": float(view.get("confidence", 0.0)),
                "raw_confidence": float(view.get("raw_confidence", view.get("confidence", 0.0))),
                "confidence_calibration": view.get("confidence_calibration", ""),
                "reason_tags": [str(x) for x in tags],
                "validator_status": status,
                "validator_reason": reason,
                "audit_instruction": "Agent must explicitly agree, lower confidence, veto, or replace with better-evidenced relative view.",
            }
        )
    return candidates


def _attach_mechanical_candidates(card: pd.Series, payload: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["mechanical_view_candidates"] = _mechanical_candidate_views_for_prompt(card, cfg)
    out["agentic_task_mode"] = {
        "primary_task": "audit_and_calibrate_mechanical_views",
        "secondary_task": "add_new_view_only_when_evidence_is_stronger_than_mechanical_rule",
        "calendar_date_hidden": True,
        "no_future_data": True,
    }
    return out


class SupervisorOutput(BaseModel):
    market_regime: str = "neutral"
    views: list[AgentView] = Field(default_factory=list)
    rejected_views: list[Any] = Field(default_factory=list)
    fallback_required: bool = False


class AgentState(TypedDict, total=False):
    card: dict[str, Any]
    plan: dict[str, Any]
    findings: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]
    reviewed: list[dict[str, Any]]
    supervisor: dict[str, Any]
    raw_responses: Annotated[list[dict[str, Any]], operator.add]


GLOBAL_RESEARCH_DISCIPLINE = """
你在一个固收+多资产配置系统里工作。这个系统不是聊天分析，而是历史回测与实盘投研都要能审计的 Market View 生产线。

共同研究纪律：
1. 只使用 Market State Card 中当前 decision_id 可见的信息；真实日历日期已隐藏，不许猜测日期、不许补写实时行情、不许引用今天的新闻解释历史。
2. 每个判断必须拆成四层：输入事实、状态/模型信号、分析推断、可交易含义。四层不能混在一句话里。
3. 精确数字只能来自输入字段。没有字段就写 data_gaps，不能用“市场一般认为”“大概”“近期”等词伪装数据。
4. 先识别一到两个主矛盾，再判断资产相对强弱。不要把所有框架维度机械铺开。
5. 任何正向观点都必须同时写反证：什么条件出现，该观点撤销或降级。
6. 禁止输出目标权重、仓位比例、突破约束建议、确定性收益预测、cash view。
7. 如果输入证据不足，宁可返回 neutral/mixed 和低 confidence，也不要为了“有观点”而编造。
8. 观点最终只会进入 EP/skfolio；你的任务是提供可审计的 relative view evidence，不是替优化器做决定。
9. 如果 Market State Card 中提供 mechanical_view_candidates，你的第一职责是审核这些候选：agree、lower_confidence、veto 或 replace；只有证据明显更强时才新增 view。
10. confidence 不能凭感觉给。必须先给 direction_strength、evidence_quality、cross_module_confirmation、contradiction_penalty、liquidity_penalty 五项 0-1 评分，再让主管按固定公式校准。
"""


PLANNER_SYSTEM_PROMPT = """
你是固收+多资产 Market View 的研究规划者，职责类似 DeepResearch 的 Planner。

你的目标不是直接给观点，而是把当前 Market State Card 拆成一棵小型研究问题树，分派给后续领域专家。

必须遵守：
1. 先读 decision_id、states、key_metrics、state_tags、view_filters、eligible_universe。as_of_date 只能是隐藏后的 decision_id，不是真实日历日期。
2. 找出本期最多 3 个主矛盾，例如“信用票息补偿 vs 融资压力”“利率下行趋势 vs 财政供给压力”“权益宽度修复 vs 估值压力”。
3. 优先围绕 mechanical_view_candidates 设计审核任务：哪些候选需要确认、降级、否决，哪些主矛盾可能支持新增替代 view。
4. 把每个主矛盾翻译为专家任务。任务必须可回答、可验证、可映射到 allowed_assets。
5. 禁止过度规划。每个专家 3-6 个 required_checks 即可，但必须具体。
6. 不能要求专家查询外部实时数据；历史回测模式只能用 Market State Card。
7. 明确 do_not_use：不能用哪些缺失数据、不能输出哪些资产、不能做哪些结论。
8. 不得要求专家输出目标权重、仓位比例、超配百分比、低于基准多少、XX% 等任何组合数字。

输出严格 JSON，字段：
{
  "as_of_date": "decision_id",
  "dominant_contradictions": ["主矛盾1", "主矛盾2"],
  "regime_hypotheses": ["可能的市场状态假设"],
  "cross_asset_questions": ["需要最终Supervisor回答的跨资产问题"],
  "expert_tasks": [
    {
      "expert": "rates_fiscal|credit|equity_cb|gold|fx_us_macro|ust",
      "main_question": "该专家要回答的核心问题",
      "research_goal": "为什么这个问题影响组合Market View",
      "required_checks": ["必须检查的输入字段/状态/逻辑"],
      "asset_focus": ["rates_long_duration", "credit"],
      "expected_output": "希望该专家产出什么证据"
    }
  ],
  "do_not_use": ["禁止事项"],
  "fallback_required": false
}
"""


EXPERT_PROMPTS = {
    "rates_fiscal": (
        """
角色：利率、财政与流动性专家。
你吸收财政政策 skill 的“卖方功底、买方视角、抓主要矛盾、并行取证”原则，也吸收利率衍生品 skill 对因果符号和交易时点的谨慎要求。

分析顺序：
1. 增长/通胀/风险预算状态对利率方向的含义。
2. 财政发力如果存在，是增长预期改善、债券供给压力，还是流动性投放/财政支出节奏。
3. 资金面是否支持久期，是否存在流动性冲击。
4. 曲线与久期赔率：长久期是趋势机会、避险资产，还是供给/波动风险。
5. 映射到 rates_long_duration、credit、equity 的相对强弱。

必须给出：
- fact_observations：只引用输入字段或状态码。
- inference_chain：从事实到利率/久期判断的推理链。
- contrarian_checks：什么情况会让久期观点失效。

禁止：
- 把宽财政简单等同于债券利空。
- 把技术走势当成基本面。
- 用状态卡外的政策新闻或当日资金价格。
"""
    ),
    "credit": (
        """
角色：信用策略专家。
你按信用策略 skill 的华泰固收五维框架工作：基本面与利率趋势、信用债供需与机构行为、信用风险、收益率与利差水平、资金面。

分析顺序：
1. 利率环境：久期资产强弱会不会压制信用相对价值。
2. 信用补偿：利差分位、利差变化、票息补偿是否足以覆盖尾部风险。
3. 供需与机构行为：输入里若没有机构数据，必须写 data_gaps，不能假设理财/保险配置行为。
4. 信用风险：融资压力、利差走阔、评级下沉、低流动性是否出现。
5. 资金面：杠杆和流动性是否支持信用 carry。

必须回答：
- credit 相对 rates_long_duration 是 carry 占优，还是防御性不足。
- credit 相对 convertible_bonds 是更稳的票息，还是被转债性价比压制。

禁止：
- 只因票息高就给 credit overweight。
- 在利差过薄或资金压力上升时忽略尾部风险。
"""
    ),
    "equity_cb": (
        """
角色：权益与可转债专家。
你吸收 A 股策略 skill 的盈利、流动性、风险偏好三因子，也吸收可转债 skill 的股性、债性、估值、条款和生命周期框架。

分析顺序：
1. 权益：盈利/景气、估值、宽度、趋势、风险偏好分别给什么信号。
2. 转债：正股环境、价格中枢、估值/溢价、赎回压力、债底保护分别给什么信号。
3. 风险预算：当前组合是否应承担权益 beta，还是只用转债表达温和进攻。
4. 映射：equity vs credit、convertible_bonds vs credit、convertible_bonds vs equity。

必须回答：
- 权益机会是基本面、估值修复、流动性还是纯趋势。
- 转债当前是增强收益工具，还是估值拥挤/赎回压力资产。

禁止：
- 看到科技成长叙事就提高权益结论。
- 忽略宽度、估值和最大回撤。
"""
    ),
    "gold": (
        """
角色：黄金与商品专家。
你按 gold-skill 的定价框架工作：实际利率/Fed 预期、央行购金与去美元化、供给成本、美元、风险偏好、持仓拥挤。

分析顺序：
1. 黄金的驱动来自避险、实际利率、美元、通胀，还是趋势动量。
2. 商品/有色/能化是否代表通胀交易、需求修复，还是单纯波动。
3. 黄金与 rates_long_duration 是同向防御，还是存在替代关系。
4. 黄金相对 equity 是否能降低回撤，还是风险偏好修复会削弱其价值。

必须回答：
- gold vs equity 是否成立。
- alternatives 内部是黄金主导还是商品通胀主导。

禁止：
- 用“避险”二字直接推出超配。
- 忽略实际利率上行和美元走强的失效条件。
"""
    ),
    "fx_us_macro": (
        """
角色：汇率与美国宏观专家。
你按 FX skill 的四层框架工作：中间价、CFETS 篮子、三驾马车、政策工具；同时按美国宏观框架检查增长、通胀和政策路径。

分析顺序：
1. 输入是否提供美元、中美利差、汇率压力；没有就列 data_gaps。
2. 外部环境是 risk-off、美元压力、流动性收紧，还是风险偏好修复。
3. 外部变量如何传导到 gold、rates_long_duration、equity。
4. 对国内固收+组合是提高防御、增加黄金，还是允许风险资产。

必须回答：
- 外部宏观变量是否支持黄金/久期。
- 是否有足够证据影响国内 equity 或 convertible_bonds。

禁止：
- 使用状态卡外的当日 USDCNY、CNH、DXY、政策工具状态。
- 把美元方向与人民币政策态度混为一谈。
"""
    ),
    "ust": (
        """
角色：美债与全球利率专家。
你按 UST framework 的“五碗面”综合：基本面、政策、在岸流动性、离岸美元流动性、供需、情绪/仓位。

分析顺序：
1. 美国增长和通胀状态对实际利率的含义。
2. Fed 反应函数和降息/加息预期如何影响全球久期。
3. 期限溢价、供给和风险偏好是否支持美债或黄金。
4. 全球利率信号如何映射到国内 rates_long_duration、gold、equity。

必须回答：
- 外部利率信号是支持防御，还是支持风险资产。
- 当前判断的胜率、赔率和止损条件。

禁止：
- 把美债技术面信号当作宏观方向。
- 在输入没有美债数据时硬给方向。
"""
    ),
}


def load_agent_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DEFAULT_CONFIG).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _model_cfg(agent_cfg: dict[str, Any], market_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(market_cfg)
    cfg["llm_views"] = dict(agent_cfg.get("model", {}))
    return cfg


def _call_model(messages: list[dict[str, str]], model_cfg: dict[str, Any], agent_cfg: dict[str, Any]) -> str:
    attempts = int(agent_cfg.get("request_retry_attempts", 3))
    base = float(agent_cfg.get("request_retry_base_seconds", 3.0))
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return _glm_chat_completion(messages, model_cfg)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            transient_markers = [
                "429",
                "Too Many Requests",
                "UNEXPECTED_EOF",
                "EOF occurred",
                "timed out",
                "Connection reset",
                "Remote end closed",
                "temporarily unavailable",
            ]
            if not any(marker in message for marker in transient_markers):
                raise
            if attempt + 1 >= attempts:
                break
            time.sleep(base * (attempt + 1))
    raise last_error or RuntimeError("model_call_failed")


class StructuredCallError(Exception):
    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


def _call_structured_model(
    messages: list[dict[str, str]],
    model_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    parse_fn,
) -> tuple[Any, str]:
    attempts = int(agent_cfg.get("request_retry_attempts", 3))
    base = float(agent_cfg.get("request_retry_base_seconds", 3.0))
    last_error: Exception | None = None
    last_raw = ""
    for attempt in range(max(1, attempts)):
        try:
            raw = _call_model(messages, model_cfg, {**agent_cfg, "request_retry_attempts": 1})
            last_raw = raw
            if not raw.strip():
                raise ValueError("empty_model_response")
            return parse_fn(raw), raw
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(base * (attempt + 1))
    raise StructuredCallError(str(last_error or "structured_model_call_failed"), raw=last_raw)


def _planner_messages(card: dict[str, Any], agent_cfg: dict[str, Any]) -> list[dict[str, str]]:
    experts = [name for name, cfg in agent_cfg.get("experts", {}).items() if cfg.get("enabled", True)]
    user = {
        "task": "为本期Market View生成研究计划。不要直接给views，只拆主矛盾、专家任务和禁止事项。",
        "available_experts": experts,
        "allowed_assets": [
            "rates_long_duration",
            "credit",
            "convertible_bonds",
            "equity",
            "gold",
            "alternatives",
        ],
        "market_state_card": card,
    }
    return [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _planner_node(model_cfg: dict[str, Any], agent_cfg: dict[str, Any]):
    def node(state: AgentState) -> dict[str, Any]:
        raw = ""
        try:
            plan, raw = _call_structured_model(
                _planner_messages(state["card"], agent_cfg),
                model_cfg,
                agent_cfg,
                lambda text: MarketThesisPlan.model_validate(_extract_json_object(text)),
            )
            return {"plan": plan.model_dump(), "raw_responses": [{"stage": "planner", "status": "ok", "raw_response": raw}]}
        except Exception as exc:
            raw = getattr(exc, "raw", raw)
            fallback_tasks = [
                ResearchTask(
                    expert=name,
                    main_question=f"{name} should evaluate its domain signals from the Market State Card.",
                    research_goal="Fallback plan because planner failed.",
                    required_checks=["states", "key_metrics", "state_tags", "view_filters"],
                    asset_focus=[],
                    expected_output="Return conservative evidence or data gaps.",
                ).model_dump()
                for name, cfg in agent_cfg.get("experts", {}).items()
                if cfg.get("enabled", True)
            ]
            plan = MarketThesisPlan(
                as_of_date=str(state["card"].get("decision_id", "")),
                dominant_contradictions=["planner_failed"],
                expert_tasks=fallback_tasks,
                fallback_required=False,
                do_not_use=["external_live_data", "cash_views", "target_weights"],
            )
            return {
                "plan": plan.model_dump(),
                "errors": [{"stage": "planner", "error": str(exc)}],
                "raw_responses": [{"stage": "planner", "status": "error", "raw_response": raw or str(exc)}],
            }

    return node


def _task_for_expert(plan: dict[str, Any], name: str) -> dict[str, Any]:
    for task in plan.get("expert_tasks", []) or []:
        if task.get("expert") == name:
            return task
    return {
        "expert": name,
        "main_question": f"Evaluate {name} signals from this Market State Card.",
        "research_goal": "No explicit planner task found; provide conservative evidence.",
        "required_checks": ["states", "key_metrics", "state_tags", "view_filters"],
        "asset_focus": [],
        "expected_output": "Evidence, inference, contrarian checks, candidate views.",
    }


def _expert_messages(name: str, card: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "你是固收+多资产配置系统中的领域专家。你承接Planner分派的研究任务，像DeepResearch的Researcher一样工作：围绕任务取证、推理、反证，而不是泛泛点评。"
        + GLOBAL_RESEARCH_DISCIPLINE
        + EXPERT_PROMPTS[name]
        + "输出严格JSON，不要Markdown，不要解释JSON以外内容。"
    )
    schema = {
        "expert": name,
        "conclusion": "结论：主矛盾 + 资产方向 + 置信度来源",
        "relevant_assets": ["rates_long_duration"],
        "direction": "positive|neutral|negative|mixed",
        "confidence": 0.55,
        "direction_strength": 0.55,
        "evidence_quality": 0.60,
        "cross_module_confirmation": 0.50,
        "contradiction_penalty": 0.20,
        "liquidity_penalty": 0.10,
        "fact_observations": ["只引用输入字段、状态码、state_tags、view_filters"],
        "inference_chain": ["事实 -> 状态解释 -> 资产相对含义"],
        "evidence": ["可被Reviewer复核的证据ID或字段名"],
        "contrarian_checks": ["反证或失效条件"],
        "risk_flags": [],
        "invalidation_conditions": [],
        "data_gaps": [],
        "suggested_view_candidates": ["source_asset overweight vs target_asset, or neutral"],
    }
    user = {
        "research_plan": plan,
        "assigned_task": _task_for_expert(plan, name),
        "task": "围绕assigned_task形成领域意见，供审稿人交叉验证。必须按schema输出。",
        "output_contract": [
            "fact_observations只写输入中看得到的事实/状态。",
            "inference_chain必须逐步解释，不许跳到结论。",
            "如果market_state_card.mechanical_view_candidates中有与你领域相关的候选，必须在conclusion或evidence中明确写agree/lower_confidence/veto/replace之一。",
            "suggested_view_candidates可以提出0-2条候选relative view，但不许给权重。",
            "新增候选必须解释为什么比机械候选更好；否则优先校准机械候选confidence。",
            "如果assigned_task依赖缺失数据，写data_gaps并降低confidence。",
            "必须写contrarian_checks和invalidation_conditions。",
            "必须写五项评分：direction_strength/evidence_quality/cross_module_confirmation/contradiction_penalty/liquidity_penalty。",
            "评分口径：direction_strength看方向清晰度；evidence_quality看是否有字段支撑；cross_module_confirmation看其他模块是否同向；contradiction_penalty看反证强度；liquidity_penalty看流动性或交易约束压力。",
        ],
        "required_schema": schema,
        "market_state_card": card,
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _expert_node(name: str, model_cfg: dict[str, Any], agent_cfg: dict[str, Any]):
    def node(state: AgentState) -> dict[str, Any]:
        raw = ""
        try:
            finding, raw = _call_structured_model(
                _expert_messages(name, state["card"], state.get("plan", {})),
                model_cfg,
                agent_cfg,
                lambda text: ExpertFinding.model_validate(_extract_json_object(text)),
            )
            finding.expert = name
            return {
                "findings": [finding.model_dump()],
                "raw_responses": [{"stage": name, "status": "ok", "raw_response": raw}],
            }
        except Exception as exc:
            raw = getattr(exc, "raw", raw)
            return {
                "errors": [{"stage": name, "error": str(exc)}],
                "raw_responses": [{"stage": name, "status": "error", "raw_response": raw or str(exc)}],
            }

    return node


def _review_messages(card: dict[str, Any], plan: dict[str, Any], findings: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = (
        """
你是固收+投研审稿人和未来函数风控，工作方式参考 DeepResearch Reviewer。
你的任务不是让观点更多，而是判断每个专家节点是否“足以进入观点综合”。

逐条检查：
1. 知识覆盖度：是否回答了 Planner 分派的问题，是否遗漏 required_checks。
2. 信息质量：证据是否来自 Market State Card，是否有具体字段/状态码支撑。
3. 时效性：是否猜测或使用真实日历日期，是否使用 decision_id 之后的信息，或把缺失实时数据写成事实。
4. 推理质量：fact_observations -> inference_chain -> conclusion 是否连贯，因果方向是否正确。
5. 组合相关性：是否能映射到 allowed_assets 的相对观点，还是只有宏观空话。
6. 冲突和重复：是否与其他专家高度共线，是否存在无法解释的冲突。
7. 风险纪律：是否写了 contrarian_checks 和 invalidation_conditions。

评级规则：
- accept：证据明确、推理完整、可进入Supervisor。
- downgrade：方向有用但证据弱、数据缺口明显或与其他专家部分冲突。
- reject：证据不在输入中、未来函数、无法映射资产、推理跳跃或违反view_filters。

输出严格JSON，不要Markdown。
"""
    )
    user = {
        "task": "逐条审核专家意见。",
        "allowed_status": ["accept", "downgrade", "reject"],
        "required_schema": {
            "reviewed": [{
                "expert": "credit",
                "status": "accept|downgrade|reject",
                "confidence": 0.55,
                "sufficiency_score": 0.0,
                "usable_for_view": True,
                "direction_strength": 0.55,
                "evidence_quality": 0.60,
                "cross_module_confirmation": 0.50,
                "contradiction_penalty": 0.20,
                "liquidity_penalty": 0.10,
                "accepted_evidence": [],
                "issues": [],
                "missing_checks": []
            }]
        },
        "research_plan": plan,
        "market_state_card": card,
        "findings": findings,
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _review_node(model_cfg: dict[str, Any], agent_cfg: dict[str, Any]):
    def extract_reviewed_items(obj: dict[str, Any]) -> list[Any]:
        direct = obj.get("reviewed")
        if isinstance(direct, list) and direct:
            return direct
        nested = obj.get("required_schema", {})
        if isinstance(nested, dict):
            nested_reviewed = nested.get("reviewed")
            if isinstance(nested_reviewed, list) and nested_reviewed:
                return nested_reviewed
        return []

    def fallback_review(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reviewed: list[dict[str, Any]] = []
        for item in findings:
            evidence = item.get("evidence", []) or item.get("fact_observations", [])
            candidates = item.get("suggested_view_candidates", [])
            data_gaps = item.get("data_gaps", [])
            contrarian = item.get("contrarian_checks", []) or item.get("invalidation_conditions", [])
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            has_evidence = bool(evidence)
            has_candidate = bool(candidates)
            has_risk_discipline = bool(contrarian)
            usable = has_evidence and has_candidate and confidence >= 0.45
            if usable and has_risk_discipline and len(data_gaps) <= 2:
                status = "accept"
            elif usable:
                status = "downgrade"
            else:
                status = "reject"
            issues: list[str] = []
            if not has_evidence:
                issues.append("missing_evidence")
            if not has_candidate:
                issues.append("missing_view_candidate")
            if not has_risk_discipline:
                issues.append("missing_contrarian_checks")
            if data_gaps:
                issues.extend([f"data_gap:{x}" for x in data_gaps[:3]])
            sufficiency = float(np.clip(confidence - 0.05 * len(issues), 0.0, 1.0))
            direction_strength = float(np.clip(item.get("direction_strength", confidence), 0.0, 1.0))
            evidence_quality = float(np.clip(item.get("evidence_quality", 0.65 if has_evidence else 0.25), 0.0, 1.0))
            confirmation = float(np.clip(item.get("cross_module_confirmation", 0.50), 0.0, 1.0))
            contradiction = float(np.clip(item.get("contradiction_penalty", 0.20 + 0.08 * len(issues)), 0.0, 1.0))
            liquidity_penalty = float(np.clip(item.get("liquidity_penalty", 0.20 if data_gaps else 0.05), 0.0, 1.0))
            reviewed.append(
                ReviewedFinding(
                    expert=str(item.get("expert", "")),
                    status=status,
                    confidence=_evidence_calibrated_confidence(direction_strength, evidence_quality, confirmation, contradiction, liquidity_penalty, 0.0, 1.0),
                    sufficiency_score=sufficiency,
                    usable_for_view=status in {"accept", "downgrade"},
                    direction_strength=direction_strength,
                    evidence_quality=evidence_quality,
                    cross_module_confirmation=confirmation,
                    contradiction_penalty=contradiction,
                    liquidity_penalty=liquidity_penalty,
                    accepted_evidence=[str(x) for x in evidence[:6]],
                    issues=issues,
                    missing_checks=[str(x) for x in data_gaps[:6]],
                ).model_dump()
            )
        return reviewed

    def node(state: AgentState) -> dict[str, Any]:
        if not state.get("findings"):
            return {"reviewed": [], "errors": [{"stage": "reviewer", "error": "no_expert_findings"}]}
        raw = ""
        try:
            def parse_review(text: str) -> list[dict[str, Any]]:
                obj = _extract_json_object(text)
                reviewed_items = extract_reviewed_items(obj)
                if not reviewed_items:
                    raise ValueError("reviewer_returned_no_reviewed_items")
                parsed = []
                for x in reviewed_items:
                    item = ReviewedFinding.model_validate(x)
                    item.confidence = _evidence_calibrated_confidence(
                        item.direction_strength,
                        item.evidence_quality,
                        item.cross_module_confirmation,
                        item.contradiction_penalty,
                        item.liquidity_penalty,
                        0.0,
                        1.0,
                    )
                    parsed.append(item.model_dump())
                return parsed

            reviewed, raw = _call_structured_model(
                _review_messages(state["card"], state.get("plan", {}), state["findings"]),
                model_cfg,
                agent_cfg,
                parse_review,
            )
            return {"reviewed": reviewed, "raw_responses": [{"stage": "reviewer", "status": "ok", "raw_response": raw}]}
        except Exception as exc:
            raw = getattr(exc, "raw", raw)
            return {
                "reviewed": fallback_review(state["findings"]),
                "errors": [{"stage": "reviewer", "error": str(exc)}],
                "raw_responses": [{"stage": "reviewer", "status": "error", "raw_response": raw or str(exc)}],
            }

    return node


def _supervisor_messages(card: dict[str, Any], plan: dict[str, Any], findings: list[dict[str, Any]], reviewed: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, str]]:
    allowed = [x for x in cfg.get("view_generation", {}).get("allowed_assets", []) if x != "cash"]
    system = (
        """
你是固收+ Market View 主管，工作方式参考 DeepResearch 的 Knowledge Organizer + Supervisor。
你不做新的事实判断，只综合 Planner、专家意见和 Reviewer 结果。

综合流程：
1. 先读取 market_state_card.mechanical_view_candidates，把它们当成本期规则基准。
2. 只使用 reviewer.status 为 accept 或 downgrade 且 usable_for_view=true 的意见。
3. 对每条机械候选做四选一：agree_keep、raise_confidence、lower_confidence、veto_mechanical。
4. 只有当专家证据明显强于机械规则，且不违反 view_filters，才允许输出 new_agentic_override。
5. 最终输出不超过配置上限的非共线 relative views。若机械候选足够好，优先输出校准后的机械候选；若证据支持，也可以输出同袖套内部的资产级 view。
6. 每条 view 必须是 allowed_assets 中两个资产的 relative view，relation 固定 overweight。
7. 每条 view 必须输出五项评分：direction_strength、evidence_quality、cross_module_confirmation、contradiction_penalty、liquidity_penalty。最终 confidence 会按固定公式重算，不以模型口头 confidence 为准。
8. reason_tags 必须来自 state_tags、状态码、mechanical candidate reason_tags、accepted_evidence 或专家 evidence。
9. 允许在同一袖套内部输出具体资产相对观点，例如 equity_chinext vs equity_dividend_utility、alt_gold vs alt_chemical、rate_30y vs rate_7_10y；但不得用具体资产 view 绕开大类 view_filters。
9. model_alignment 必须写成 agree_mechanical、raise_mechanical_confidence、lower_mechanical_confidence、veto_mechanical_replacement、new_agentic_override、mixed 之一。

禁止：
- 输出 cash、目标权重、绝对收益预测。
- 用“看好/不看好”替代 source_asset vs target_asset。
- 为了凑满上限而输出弱观点。
- 把两个同向防御观点重复表达。
- 在没有比机械规则更强证据时，输出与机械候选方向相反的 view。

如果没有足够证据，fallback_required=true。
输出严格JSON。
"""
    )
    user = {
        "task": "审核机械MarketSkill候选并校准为可供EP使用的相对观点。",
        "allowed_assets": allowed,
        "required_schema": {
            "market_regime": "risk_on|neutral|risk_off",
            "views": [{
                "source_asset": "rates_long_duration", "target_asset": "equity", "relation": "overweight",
                "confidence": 0.6, "horizon_days": 20, "evidence_ids": ["R1"], "reason_tags": ["R1"],
                "model_alignment": "agree_mechanical|raise_mechanical_confidence|lower_mechanical_confidence|veto_mechanical_replacement|new_agentic_override|mixed",
                "direction_strength": 0.60,
                "evidence_quality": 0.65,
                "cross_module_confirmation": 0.55,
                "contradiction_penalty": 0.15,
                "liquidity_penalty": 0.05,
                "risk_flags": [], "invalidation_conditions": []
            }],
            "rejected_views": [],
            "fallback_required": False,
        },
        "market_state_card": card,
        "mechanical_view_candidates": card.get("mechanical_view_candidates", []),
        "research_plan": plan,
        "findings": findings,
        "reviewed": reviewed,
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _supervisor_node(model_cfg: dict[str, Any], market_cfg: dict[str, Any], agent_cfg: dict[str, Any]):
    def deterministic_supervisor(card: dict[str, Any], findings: list[dict[str, Any]], reviewed: list[dict[str, Any]]) -> SupervisorOutput:
        by_expert = {str(x.get("expert", "")): x for x in findings}
        by_review = {str(x.get("expert", "")): x for x in reviewed}
        usable = [x for x in reviewed if x.get("status") in {"accept", "downgrade"} and x.get("usable_for_view")]
        filters = card.get("view_filters", {}) or {}
        views: list[AgentView] = []

        def add(source: str, target: str, confidence: float, evidence: list[str], tags: list[str], risk_flags: list[str] | None = None) -> None:
            if len(views) >= int(agent_cfg.get("max_views_per_date", 3)):
                return
            if source == target:
                return
            if source == "credit" and filters.get("block_credit_overweight"):
                return
            if source == "convertible_bonds" and filters.get("block_cb_overweight"):
                return
            if source == "rates_long_duration" and filters.get("block_long_rate_overweight"):
                return
            if source == "equity" and filters.get("block_equity_overweight"):
                return
            if source not in market_cfg.get("view_generation", {}).get("allowed_assets", []):
                return
            if target not in market_cfg.get("view_generation", {}).get("allowed_assets", []):
                return
            expert_keys = [str(x) for x in tags if str(x) in by_review]
            if expert_keys:
                scored = [by_review[x] for x in expert_keys]
                direction_strength = float(np.mean([float(x.get("direction_strength", 0.5)) for x in scored]))
                evidence_quality = float(np.mean([float(x.get("evidence_quality", 0.5)) for x in scored]))
                confirmation = float(np.mean([float(x.get("cross_module_confirmation", 0.5)) for x in scored]))
                contradiction = float(np.mean([float(x.get("contradiction_penalty", 0.0)) for x in scored]))
                liquidity = float(np.mean([float(x.get("liquidity_penalty", 0.0)) for x in scored]))
                confidence = _evidence_calibrated_confidence(direction_strength, evidence_quality, confirmation, contradiction, liquidity, 0.0, 1.0)
            else:
                direction_strength = float(np.clip(confidence, 0.0, 1.0))
                evidence_quality = 0.55 if evidence else 0.30
                confirmation = 0.50
                contradiction = 0.20
                liquidity = 0.05
                confidence = _evidence_calibrated_confidence(direction_strength, evidence_quality, confirmation, contradiction, liquidity, 0.0, 1.0)
            views.append(
                AgentView(
                    source_asset=source,
                    target_asset=target,
                    relation="overweight",
                    confidence=float(np.clip(confidence, agent_cfg.get("confidence_min", 0.45), agent_cfg.get("confidence_max", 0.75))),
                    horizon_days=20,
                    evidence_ids=[str(x) for x in evidence[:6]],
                    reason_tags=[str(x) for x in tags[:8]],
                    model_alignment="deterministic_supervisor",
                    direction_strength=direction_strength,
                    evidence_quality=evidence_quality,
                    cross_module_confirmation=confirmation,
                    contradiction_penalty=contradiction,
                    liquidity_penalty=liquidity,
                    risk_flags=[str(x) for x in (risk_flags or [])[:6]],
                    invalidation_conditions=["reviewed_evidence_no_longer_holds", "state_filters_change"],
                )
            )

        usable_experts = {str(x.get("expert", "")) for x in usable}
        rates = by_expert.get("rates_fiscal", {})
        ust = by_expert.get("ust", {})
        credit = by_expert.get("credit", {})
        equity_cb = by_expert.get("equity_cb", {})
        gold = by_expert.get("gold", {})

        rates_positive = rates.get("direction") == "positive" or ust.get("direction") == "positive"
        credit_not_positive = credit.get("direction") in {"neutral", "negative", "mixed"} or filters.get("block_credit_overweight")
        equity_not_positive = equity_cb.get("direction") in {"neutral", "negative", "mixed"} or filters.get("block_equity_overweight")
        cb_positive = equity_cb.get("direction") == "positive" and not filters.get("block_cb_overweight")
        gold_positive = gold.get("direction") == "positive"

        if rates_positive and ({"rates_fiscal", "ust"} & usable_experts):
            target = "credit" if credit_not_positive else "equity"
            add(
                "rates_long_duration",
                target,
                max(float(rates.get("confidence", 0.55)), float(ust.get("confidence", 0.0) or 0.0)),
                (rates.get("evidence", []) or rates.get("fact_observations", [])) + (ust.get("evidence", []) or []),
                ["deterministic_supervisor", "rates_positive"] + (["rates_fiscal"] if "rates_fiscal" in usable_experts else []) + (["ust"] if "ust" in usable_experts else []) + list(card.get("state_tags", []) or []),
                rates.get("risk_flags", []) or ust.get("risk_flags", []),
            )
        if cb_positive and "equity_cb" in usable_experts:
            add(
                "convertible_bonds",
                "credit",
                float(equity_cb.get("confidence", 0.50)),
                equity_cb.get("evidence", []) or equity_cb.get("fact_observations", []),
                ["deterministic_supervisor", "cb_positive", "equity_cb"] + list(card.get("state_tags", []) or []),
                equity_cb.get("risk_flags", []),
            )
        if gold_positive and "gold" in usable_experts:
            add(
                "gold",
                "equity",
                float(gold.get("confidence", 0.50)),
                gold.get("evidence", []) or gold.get("fact_observations", []),
                ["deterministic_supervisor", "gold_positive", "gold"] + list(card.get("state_tags", []) or []),
                gold.get("risk_flags", []),
            )
        if not views and rates_positive and equity_not_positive:
            add(
                "rates_long_duration",
                "equity",
                0.50,
                rates.get("evidence", []) or rates.get("fact_observations", []),
                ["deterministic_supervisor", "rates_vs_equity", "rates_fiscal"] + list(card.get("state_tags", []) or []),
                rates.get("risk_flags", []),
            )
        return SupervisorOutput(
            market_regime=str(card.get("cross_asset_regime", {}).get("stock_bond_regime", "neutral")),
            views=views,
            rejected_views=[],
            fallback_required=not bool(views),
        )

    def node(state: AgentState) -> dict[str, Any]:
        accepted = [x for x in state.get("reviewed", []) if x.get("status") in {"accept", "downgrade"}]
        if not accepted:
            return {"supervisor": SupervisorOutput(fallback_required=True).model_dump()}
        raw = ""
        try:
            output, raw = _call_structured_model(
                _supervisor_messages(state["card"], state.get("plan", {}), state["findings"], accepted, market_cfg),
                model_cfg,
                agent_cfg,
                lambda text: SupervisorOutput.model_validate(_extract_json_object(text)),
            )
            return {"supervisor": output.model_dump(), "raw_responses": [{"stage": "supervisor", "status": "ok", "raw_response": raw}]}
        except (ValidationError, Exception) as exc:
            raw = getattr(exc, "raw", raw)
            deterministic = deterministic_supervisor(state["card"], state.get("findings", []), accepted)
            return {
                "supervisor": deterministic.model_dump(),
                "errors": [{"stage": "supervisor", "error": str(exc)}],
                "raw_responses": [{"stage": "supervisor", "status": "error", "raw_response": raw or str(exc)}],
            }

    return node


def build_graph(agent_cfg: dict[str, Any], market_cfg: dict[str, Any]):
    model_cfg = _model_cfg(agent_cfg, market_cfg)
    graph = StateGraph(AgentState)
    enabled = [name for name, cfg in agent_cfg.get("experts", {}).items() if cfg.get("enabled", True)]
    parallel = bool(agent_cfg.get("parallel_experts", False))
    graph.add_node("planner", _planner_node(model_cfg, agent_cfg))
    for name in enabled:
        graph.add_node(name, _expert_node(name, model_cfg, agent_cfg))
    graph.add_node("reviewer", _review_node(model_cfg, agent_cfg))
    graph.add_node("supervisor", _supervisor_node(model_cfg, market_cfg, agent_cfg))
    if parallel:
        for name in enabled:
            graph.add_edge("planner", name)
            graph.add_edge(name, "reviewer")
    elif enabled:
        graph.add_edge("planner", enabled[0])
        for prev, nxt in zip(enabled, enabled[1:]):
            graph.add_edge(prev, nxt)
        graph.add_edge(enabled[-1], "reviewer")
    else:
        graph.add_edge("planner", "reviewer")
    graph.add_edge(START, "planner")
    graph.add_edge("reviewer", "supervisor")
    graph.add_edge("supervisor", END)
    return graph.compile()


def _single_card_scope(payload: dict[str, Any]) -> dict[str, Any]:
    decision_id = str(payload.get("decision_id", ""))
    if not decision_id:
        raise ValueError("agentic_market_view_requires_hidden_decision_id")
    forbidden_payload_keys = {"cards", "history", "future", "full_panel", "all_rows", "dataframe"}
    leaked_keys = sorted(forbidden_payload_keys.intersection(payload))
    if leaked_keys:
        raise ValueError(f"agentic_market_view_payload_not_single_card: {leaked_keys}")
    if "date" in payload:
        raise ValueError("agentic_market_view_payload_must_hide_calendar_date")
    return {
        "decision_id": decision_id,
        "as_of_date": decision_id,
        "calendar_date_hidden": True,
        "visible_card_count": 1,
        "future_data_allowed": False,
        "history_table_allowed": False,
        "payload_keys": sorted(payload.keys()),
    }


def _calibrate_raw_views_with_mechanical(
    card: pd.Series,
    raw_views: list[Any],
    market_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    enh = agent_cfg.get("agentic_enhancement", {}) or {}
    if not bool(enh.get("mechanical_audit_calibration", True)):
        return [x.model_dump() if isinstance(x, BaseModel) else dict(x) for x in raw_views if isinstance(x, (dict, BaseModel))]

    filters = _json_loads(card["view_filters"])
    mechanical: dict[tuple[str, str], dict[str, Any]] = {}
    for view in generate_executable_mechanical_views(card, market_cfg):
        status, _ = validate_view(view, filters, market_cfg)
        if status == "valid":
            mechanical[(str(view["source_asset"]), str(view["target_asset"]))] = view

    agree_boost = float(enh.get("agree_confidence_boost", 0.03))
    raise_boost = float(enh.get("raise_confidence_boost", 0.06))
    lower_penalty = float(enh.get("lower_confidence_penalty", 0.06))
    override_min_conf = float(enh.get("new_override_min_confidence", 0.58))
    keep_limit = int(enh.get("max_calibrated_views", agent_cfg.get("max_views_per_date", 3)))
    lo = float(agent_cfg.get("confidence_min", 0.45))
    hi = float(agent_cfg.get("confidence_max", 0.75))
    evidence_formula = bool(enh.get("evidence_confidence_formula", True))

    calibrated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_views:
        raw = item.model_dump() if isinstance(item, BaseModel) else dict(item) if isinstance(item, dict) else {}
        source = str(raw.get("source_asset", ""))
        target = str(raw.get("target_asset", ""))
        if not source or not target or source == target:
            continue
        pair = (source, target)
        if pair in seen:
            continue
        seen.add(pair)
        alignment = str(raw.get("model_alignment", "mixed") or "mixed")
        input_conf = float(np.clip(float(raw.get("confidence", 0.0)), 0.0, 1.0))
        direction_strength = float(np.clip(float(raw.get("direction_strength", input_conf)), 0.0, 1.0))
        evidence_quality = float(np.clip(float(raw.get("evidence_quality", 0.5)), 0.0, 1.0))
        confirmation = float(np.clip(float(raw.get("cross_module_confirmation", 0.5)), 0.0, 1.0))
        contradiction = float(np.clip(float(raw.get("contradiction_penalty", 0.0)), 0.0, 1.0))
        liquidity = float(np.clip(float(raw.get("liquidity_penalty", 0.0)), 0.0, 1.0))
        raw_conf = (
            _evidence_calibrated_confidence(direction_strength, evidence_quality, confirmation, contradiction, liquidity, 0.0, 1.0)
            if evidence_formula
            else input_conf
        )
        tags = raw.get("reason_tags", []) or []
        if isinstance(tags, str):
            tags = [tags]
        calibration_steps: list[str] = [
            "agentic_evidence_formula" if evidence_formula else "agentic_raw_confidence",
            f"input={input_conf:.3f}",
            f"evidence_formula={raw_conf:.3f}",
        ]

        if pair in mechanical:
            base = mechanical[pair]
            base_conf = float(base.get("confidence", raw_conf))
            base_tags = base.get("reason_tags", []) or []
            if isinstance(base_tags, str):
                base_tags = _json_loads(base_tags)
            if alignment == "raise_mechanical_confidence":
                conf = max(raw_conf, base_conf + raise_boost)
                calibration_steps.append(f"raise_mechanical_base={base_conf:.3f}")
            elif alignment == "lower_mechanical_confidence":
                conf = min(raw_conf, base_conf - lower_penalty)
                calibration_steps.append(f"lower_mechanical_base={base_conf:.3f}")
            else:
                conf = max(raw_conf, base_conf + agree_boost)
                if alignment in {"mixed", ""}:
                    alignment = "agree_mechanical"
                calibration_steps.append(f"agree_mechanical_base={base_conf:.3f}")
            tags = sorted(set([str(x) for x in tags] + [str(x) for x in base_tags] + ["mechanical_audit_calibrated"]))
        else:
            reverse = (target, source)
            if reverse in mechanical:
                if alignment not in {"veto_mechanical_replacement", "new_agentic_override"} or raw_conf < override_min_conf:
                    continue
                tags = sorted(set([str(x) for x in tags] + ["agentic_veto_mechanical"]))
                conf = raw_conf
                calibration_steps.append("veto_reverse_mechanical")
            else:
                if alignment not in {"new_agentic_override", "mixed"} or raw_conf < override_min_conf:
                    continue
                tags = sorted(set([str(x) for x in tags] + ["agentic_new_override"]))
                alignment = "new_agentic_override" if alignment == "mixed" else alignment
                conf = raw_conf
                calibration_steps.append("agentic_new_override")

        raw["confidence"] = float(np.clip(conf, lo, hi))
        raw["raw_confidence"] = input_conf
        raw["direction_strength"] = direction_strength
        raw["evidence_quality"] = evidence_quality
        raw["cross_module_confirmation"] = confirmation
        raw["contradiction_penalty"] = contradiction
        raw["liquidity_penalty"] = liquidity
        raw["confidence_calibration"] = _json_dumps(
            {
                "method": "agentic_evidence_plus_mechanical_audit",
                "input_confidence": round(input_conf, 3),
                "evidence_formula_confidence": round(raw_conf, 3),
                "final_confidence": round(float(raw["confidence"]), 3),
                "direction_strength": round(direction_strength, 3),
                "evidence_quality": round(evidence_quality, 3),
                "cross_module_confirmation": round(confirmation, 3),
                "contradiction_penalty": round(contradiction, 3),
                "liquidity_penalty": round(liquidity, 3),
                "alignment": alignment,
                "steps": calibration_steps,
            }
        )
        raw["reason_tags"] = tags[:10]
        raw["model_alignment"] = alignment
        calibrated.append(raw)

    calibrated.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return calibrated[:keep_limit]


def _agent_rows(card: pd.Series, result: AgentState, market_cfg: dict[str, Any], agent_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    filters = _json_loads(card["view_filters"])
    supervisor = result.get("supervisor", {})
    fallback = bool(supervisor.get("fallback_required", False)) or not supervisor.get("views")
    raw_views = supervisor.get("views", [])
    if fallback and agent_cfg.get("fallback_to_mechanical", True):
        raw_views = generate_executable_mechanical_views(card, market_cfg)
    elif raw_views:
        raw_views = _calibrate_raw_views_with_mechanical(card, raw_views, market_cfg, agent_cfg)
        if not raw_views and agent_cfg.get("fallback_to_mechanical", True):
            fallback = True
            raw_views = generate_executable_mechanical_views(card, market_cfg)
    rows: list[dict[str, Any]] = []
    lo = float(agent_cfg.get("confidence_min", 0.45))
    hi = float(agent_cfg.get("confidence_max", 0.75))
    limit = len(raw_views) if fallback else int(agent_cfg.get("max_views_per_date", 3))
    for raw in raw_views[:limit]:
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        source = str(raw.get("source_asset", ""))
        target = str(raw.get("target_asset", ""))
        confidence = float(np.clip(raw.get("confidence", 0.0), lo, hi))
        raw_confidence = float(np.clip(raw.get("raw_confidence", confidence), 0.0, 1.0))
        tags = raw.get("reason_tags", [])
        if isinstance(tags, str):
            tags = [tags]
        view = _base_view(card["date"], source, target, str(raw.get("relation", "overweight")), confidence, [str(x) for x in tags])
        view["generator"] = "agentic_fallback_mechanical_v1" if fallback else "agentic_market_view_v1"
        view["raw_confidence"] = round(raw_confidence, 3)
        if raw.get("confidence_calibration"):
            view["confidence_calibration"] = raw.get("confidence_calibration")
        elif not fallback:
            view["confidence_calibration"] = _json_dumps(
                {
                    "method": "agentic_evidence_formula",
                    "input_confidence": round(raw_confidence, 3),
                    "final_confidence": round(confidence, 3),
                }
            )
        if not _view_asset_allowed(source, market_cfg, allow_cash=False) or not _view_asset_allowed(target, market_cfg, allow_cash=False):
            status, reason = "blocked", "asset_not_allowed_for_agent"
        else:
            status, reason = validate_view(view, filters, market_cfg)
        view["validator_status"] = status
        view["validator_reason"] = reason
        view["reason_tags"] = _json_dumps(sorted(set(view["reason_tags"])))
        view["horizon_days"] = int(raw.get("horizon_days", 20))
        view["evidence_ids"] = _json_dumps(raw.get("evidence_ids", []))
        view["model_alignment"] = str(raw.get("model_alignment", "fallback" if fallback else "mixed"))
        view["risk_flags"] = _json_dumps(raw.get("risk_flags", []))
        view["invalidation_conditions"] = _json_dumps(raw.get("invalidation_conditions", []))
        for score_col in [
            "direction_strength",
            "evidence_quality",
            "cross_module_confirmation",
            "contradiction_penalty",
            "liquidity_penalty",
        ]:
            view[score_col] = round(float(np.clip(raw.get(score_col, 0.5 if "penalty" not in score_col else 0.0), 0.0, 1.0)), 3)
        rows.append(view)
    return rows, fallback


def _fallback_diagnostics(result: AgentState) -> dict[str, Any]:
    reviewed = result.get("reviewed", []) or []
    supervisor = result.get("supervisor", {}) or {}
    accepted = [
        x
        for x in reviewed
        if x.get("status") in {"accept", "downgrade"} and bool(x.get("usable_for_view", False))
    ]
    supervisor_views = supervisor.get("views", []) or []
    fallback_required = bool(supervisor.get("fallback_required", False))
    if supervisor_views:
        reason = "agentic_views_available"
    elif result.get("errors"):
        reason = "node_errors"
    elif not accepted:
        reason = "no_accepted_reviewed_findings"
    elif fallback_required:
        reason = "supervisor_requested_fallback"
    elif not supervisor_views:
        reason = "supervisor_empty_views"
    return {
        "fallback_reason": reason,
        "accepted_reviewed": len(accepted),
        "reviewed_total": len(reviewed),
        "supervisor_views": len(supervisor_views),
        "supervisor_fallback_required": fallback_required,
    }


COMPRESSED_AGENT_SYSTEM_PROMPT = """
你是固收+多资产动态配置系统中的“压缩 Agentic Market View”研究节点。
你要在一次调用内完成原 LangGraph 多节点流程：Planner -> 六专家 -> Reviewer -> Supervisor。
但你仍然必须按可审计流程输出结构化 JSON，不能变成聊天分析。

核心纪律：
1. 只能使用输入的单张 Market State Card；真实日历日期已隐藏，只能看到 decision_id。
2. 禁止使用外部实时行情、新闻、真实日期、未来收益、全历史表。
3. 禁止输出目标仓位、权重比例、绝对收益预测、cash view。
4. 优先审核 mechanical_view_candidates：agree、raise_confidence、lower_confidence、veto、replace。
5. 只有证据明显强于机械规则，且不违反 view_filters，才允许 new_agentic_override。
6. 每条 view 必须是 allowed_assets 内两个资产的 relative view，relation 固定 overweight。
7. 每条 view 必须先给五项评分，再给 confidence：direction_strength、evidence_quality、cross_module_confirmation、contradiction_penalty、liquidity_penalty。
8. reason_tags 必须来自状态码、state_tags、mechanical candidate tags 或 evidence 字段，不许凭空造叙事标签。
9. 如果证据不足，fallback_required=true 或只保留低置信度机械候选校准，不要凑满配置上限。
10. 必须极度精简：每个 conclusion 不超过 40 个中文字符；每个 evidence/contrarian_checks/suggested_view_candidates 最多 2 条；不要输出长段解释。

研究步骤必须体现在 JSON 里：
- plan：一到两个主矛盾、需要检查的专家视角。
- expert_notes：六个专家视角的简短审阅。专家包括 rates_fiscal、credit、equity_cb、gold、fx_us_macro、ust。
- reviewed：对专家意见的审稿结果，accept/downgrade/reject。
- supervisor：最终不超过配置上限的 relative views。

输出必须是严格 JSON，不要 Markdown，不要 JSON 以外文本。
"""


def _compressed_agent_messages(card: dict[str, Any], agent_cfg: dict[str, Any], market_cfg: dict[str, Any]) -> list[dict[str, str]]:
    allowed = [x for x in market_cfg.get("view_generation", {}).get("allowed_assets", []) if x != "cash"]
    user = {
        "task": "在一次调用内完成Planner、六专家审阅、Reviewer和Supervisor，输出可供EP/skfolio使用的relative views。输出必须短小，避免长文本。",
        "allowed_assets": allowed,
        "required_output_schema": {
            "plan": {
                "dominant_contradictions": ["主矛盾1", "主矛盾2"],
                "expert_tasks": ["rates_fiscal", "credit", "equity_cb", "gold", "fx_us_macro", "ust"],
                "do_not_use": ["external_live_data", "calendar_date", "future_returns", "target_weights"],
            },
            "expert_notes": [
                {
                    "expert": "credit",
                    "status": "accept|downgrade|reject",
                "conclusion": "40字内结论",
                    "direction": "positive|neutral|negative|mixed",
                    "confidence": 0.55,
                    "direction_strength": 0.55,
                    "evidence_quality": 0.60,
                    "cross_module_confirmation": 0.50,
                    "contradiction_penalty": 0.20,
                    "liquidity_penalty": 0.10,
                    "evidence": ["字段名或状态码"],
                    "contrarian_checks": ["失效条件"],
                    "suggested_view_candidates": ["source_asset overweight target_asset"],
                    "data_gaps": [],
                }
            ],
            "reviewed": [
                {
                    "expert": "credit",
                    "status": "accept|downgrade|reject",
                    "confidence": 0.55,
                    "sufficiency_score": 0.60,
                    "usable_for_view": True,
                    "direction_strength": 0.55,
                    "evidence_quality": 0.60,
                    "cross_module_confirmation": 0.50,
                    "contradiction_penalty": 0.20,
                    "liquidity_penalty": 0.10,
                    "accepted_evidence": ["字段名或状态码"],
                    "issues": [],
                    "missing_checks": [],
                }
            ],
            "supervisor": {
                "market_regime": "risk_on|neutral|risk_off|mixed",
                "views": [
                    {
                        "source_asset": "rates_long_duration",
                        "target_asset": "equity",
                        "relation": "overweight",
                        "confidence": 0.60,
                        "horizon_days": 20,
                        "evidence_ids": ["R2", "EQ4"],
                        "reason_tags": ["R2", "EQ4"],
                        "model_alignment": "agree_mechanical|raise_mechanical_confidence|lower_mechanical_confidence|veto_mechanical_replacement|new_agentic_override|mixed",
                        "direction_strength": 0.60,
                        "evidence_quality": 0.65,
                        "cross_module_confirmation": 0.55,
                        "contradiction_penalty": 0.15,
                        "liquidity_penalty": 0.05,
                        "risk_flags": [],
                    "invalidation_conditions": ["状态反转"],
                    }
                ],
                "rejected_views": [],
                "fallback_required": False,
            },
        },
        "market_state_card": card,
        "length_limits": {
            "max_expert_notes": 6,
            "max_evidence_per_expert": 2,
            "max_views": int(agent_cfg.get("max_views_per_date", 3)),
            "max_chars_per_conclusion": 40,
            "max_items_per_list": 2,
        },
    }
    return [
        {"role": "system", "content": COMPRESSED_AGENT_SYSTEM_PROMPT + GLOBAL_RESEARCH_DISCIPLINE},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _json_repair_messages(raw_text: str) -> list[dict[str, str]]:
    clipped = raw_text[:12000]
    return [
        {
            "role": "system",
            "content": (
                "你是JSON修复器。只修复用户提供的损坏JSON，使其成为严格合法JSON。"
                "不要新增事实，不要改变字段含义，不要输出Markdown，不要解释。"
                "如果某个字符串或列表未闭合，请用最短方式闭合；如果内容过长，请截断该字段但保持JSON合法。"
            ),
        },
        {"role": "user", "content": clipped},
    ]


def _parse_compressed_agent_output(text: str) -> dict[str, Any]:
    try:
        obj = _extract_json_object(text)
    except Exception:
        import json_repair

        obj = json_repair.loads(text)
    plan = obj.get("plan", {}) if isinstance(obj.get("plan", {}), dict) else {}
    notes = obj.get("expert_notes", []) if isinstance(obj.get("expert_notes", []), list) else []
    reviewed_raw = obj.get("reviewed", []) if isinstance(obj.get("reviewed", []), list) else []
    supervisor_raw = obj.get("supervisor", {}) if isinstance(obj.get("supervisor", {}), dict) else {}

    findings: list[dict[str, Any]] = []
    for item in notes:
        if not isinstance(item, dict):
            continue
        try:
            finding = ExpertFinding(
                expert=str(item.get("expert", "")),
                conclusion=str(item.get("conclusion", "")),
                relevant_assets=[str(x) for x in item.get("relevant_assets", []) or []],
                direction=str(item.get("direction", "neutral")),
                confidence=float(item.get("confidence", 0.5)),
                direction_strength=float(item.get("direction_strength", item.get("confidence", 0.5))),
                evidence_quality=float(item.get("evidence_quality", 0.5)),
                cross_module_confirmation=float(item.get("cross_module_confirmation", 0.5)),
                contradiction_penalty=float(item.get("contradiction_penalty", 0.0)),
                liquidity_penalty=float(item.get("liquidity_penalty", 0.0)),
                evidence=[str(x) for x in item.get("evidence", []) or []],
                fact_observations=[str(x) for x in item.get("fact_observations", []) or item.get("evidence", []) or []],
                inference_chain=[str(x) for x in item.get("inference_chain", []) or []],
                contrarian_checks=[str(x) for x in item.get("contrarian_checks", []) or []],
                suggested_view_candidates=[str(x) for x in item.get("suggested_view_candidates", []) or []],
                data_gaps=[str(x) for x in item.get("data_gaps", []) or []],
                risk_flags=[str(x) for x in item.get("risk_flags", []) or []],
                invalidation_conditions=[str(x) for x in item.get("invalidation_conditions", []) or []],
            )
            findings.append(finding.model_dump())
        except Exception:
            continue

    reviewed: list[dict[str, Any]] = []
    for item in reviewed_raw:
        if not isinstance(item, dict):
            continue
        try:
            review = ReviewedFinding.model_validate(item)
            review.confidence = _evidence_calibrated_confidence(
                review.direction_strength,
                review.evidence_quality,
                review.cross_module_confirmation,
                review.contradiction_penalty,
                review.liquidity_penalty,
                0.0,
                1.0,
            )
            reviewed.append(review.model_dump())
        except Exception:
            continue

    valid_views: list[dict[str, Any]] = []
    for raw_view in supervisor_raw.get("views", []) or []:
        if not isinstance(raw_view, dict):
            continue
        try:
            view = AgentView.model_validate(raw_view)
            valid_views.append(view.model_dump())
        except Exception:
            continue
    supervisor = SupervisorOutput(
        market_regime=str(supervisor_raw.get("market_regime", "neutral")),
        views=valid_views,
        rejected_views=supervisor_raw.get("rejected_views", []) if isinstance(supervisor_raw.get("rejected_views", []), list) else [],
        fallback_required=bool(supervisor_raw.get("fallback_required", False)),
    ).model_dump()
    return {"plan": plan, "findings": findings, "reviewed": reviewed, "supervisor": supervisor}


def _compressed_empty_supervisor_repair(card: dict[str, Any], parsed: dict[str, Any], agent_cfg: dict[str, Any]) -> dict[str, Any]:
    supervisor = parsed.get("supervisor", {}) or {}
    if supervisor.get("views") or supervisor.get("fallback_required"):
        return parsed
    reviewed = parsed.get("reviewed", []) or []
    usable = [x for x in reviewed if x.get("status") in {"accept", "downgrade"} and bool(x.get("usable_for_view", False))]
    if not usable:
        return parsed
    candidates = card.get("mechanical_view_candidates", []) or []
    valid_candidates = [x for x in candidates if x.get("validator_status") == "valid"]
    if not valid_candidates:
        return parsed
    avg_direction = float(np.mean([float(x.get("direction_strength", 0.5)) for x in usable]))
    avg_evidence = float(np.mean([float(x.get("evidence_quality", 0.5)) for x in usable]))
    avg_confirm = float(np.mean([float(x.get("cross_module_confirmation", 0.5)) for x in usable]))
    avg_contra = float(np.mean([float(x.get("contradiction_penalty", 0.2)) for x in usable]))
    avg_liq = float(np.mean([float(x.get("liquidity_penalty", 0.05)) for x in usable]))
    limit = int(agent_cfg.get("max_views_per_date", 3))
    repaired_views: list[dict[str, Any]] = []
    for cand in sorted(valid_candidates, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)[:limit]:
        tags = cand.get("reason_tags", []) or []
        if isinstance(tags, str):
            tags = _json_loads(tags)
        conf = _evidence_calibrated_confidence(avg_direction, avg_evidence, avg_confirm, avg_contra, avg_liq, 0.0, 1.0)
        base_conf = float(cand.get("confidence", conf))
        repaired_views.append(
            {
                "source_asset": str(cand.get("source_asset", "")),
                "target_asset": str(cand.get("target_asset", "")),
                "relation": "overweight",
                "confidence": float(np.clip(max(conf, min(base_conf + 0.01, 0.78)), 0.0, 1.0)),
                "horizon_days": 20,
                "evidence_ids": [str(x.get("expert", "")) for x in usable[:2]],
                "reason_tags": sorted(set([str(x) for x in tags] + ["compressed_empty_supervisor_repair"])),
                "model_alignment": "agree_mechanical",
                "direction_strength": avg_direction,
                "evidence_quality": avg_evidence,
                "cross_module_confirmation": avg_confirm,
                "contradiction_penalty": avg_contra,
                "liquidity_penalty": avg_liq,
                "risk_flags": ["supervisor_empty_repaired"],
                "invalidation_conditions": ["state_filters_change"],
            }
        )
    if repaired_views:
        parsed["supervisor"] = {
            "market_regime": str(card.get("cross_asset_regime", {}).get("stock_bond_regime", "neutral")),
            "views": repaired_views,
            "rejected_views": [{"reason": "llm_supervisor_empty_repaired_with_mechanical_candidates"}],
            "fallback_required": False,
        }
    return parsed


def build_compressed_graph(agent_cfg: dict[str, Any], market_cfg: dict[str, Any]):
    model_cfg = _model_cfg(agent_cfg, market_cfg)

    def single_agent_node(state: AgentState) -> dict[str, Any]:
        raw = ""
        try:
            parsed, raw = _call_structured_model(
                _compressed_agent_messages(state["card"], agent_cfg, market_cfg),
                model_cfg,
                agent_cfg,
                _parse_compressed_agent_output,
            )
            parsed = _compressed_empty_supervisor_repair(state["card"], parsed, agent_cfg)
            return {
                **parsed,
                "raw_responses": [{"stage": "compressed_single_agent", "status": "ok", "raw_response": raw}],
            }
        except Exception as exc:
            raw = getattr(exc, "raw", raw)
            if raw:
                try:
                    repaired, repair_raw = _call_structured_model(
                        _json_repair_messages(raw),
                        model_cfg,
                        {**agent_cfg, "request_retry_attempts": 1},
                        _parse_compressed_agent_output,
                    )
                    repaired = _compressed_empty_supervisor_repair(state["card"], repaired, agent_cfg)
                    return {
                        **repaired,
                        "raw_responses": [
                            {"stage": "compressed_single_agent", "status": "repair_source_error", "raw_response": raw},
                            {"stage": "compressed_json_repair", "status": "ok", "raw_response": repair_raw},
                        ],
                    }
                except Exception as repair_exc:
                    raw = getattr(repair_exc, "raw", raw)
            return {
                "supervisor": SupervisorOutput(fallback_required=True).model_dump(),
                "errors": [{"stage": "compressed_single_agent", "error": str(exc)}],
                "raw_responses": [{"stage": "compressed_single_agent", "status": "error", "raw_response": raw or str(exc)}],
            }

    graph = StateGraph(AgentState)
    graph.add_node("compressed_single_agent", single_agent_node)
    graph.add_edge(START, "compressed_single_agent")
    graph.add_edge("compressed_single_agent", END)
    return graph.compile()


def run_compressed_agentic_market_views(cards: pd.DataFrame, frequency: str | None = None, config_path: Path | None = None) -> pd.DataFrame:
    agent_cfg = load_agent_config(config_path)
    market_cfg = load_market_skill_config()
    from src.market_skill import cards_for_llm_frequency

    selected = cards_for_llm_frequency(cards, frequency or str(agent_cfg.get("frequency", "month_end"))).sort_values("date").reset_index(drop=True)
    graph = build_compressed_graph(agent_cfg, market_cfg)
    all_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    paths_cfg = agent_cfg["paths"]
    views_path = ROOT / paths_cfg["agentic_views"]
    ep_path = ROOT / paths_cfg["agentic_ep_views"]
    audit_path = ROOT / paths_cfg["audit_jsonl"]
    raw_path = ROOT / paths_cfg["raw_responses"]
    report_path = ROOT / paths_cfg["report"]
    progress_path = report_path.with_name("agentic_market_view_compressed_progress.log")
    for path in [views_path, ep_path, audit_path, raw_path, report_path, progress_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        f"Compressed Agentic Market View run started: frequency={frequency or agent_cfg.get('frequency', 'month_end')}, total_dates={len(selected)} hidden_calendar_date=true requests_per_date=1\n",
        encoding="utf-8",
    )
    decision_ids = {
        str(pd.Timestamp(row["date"]).date()): f"decision_{i:04d}"
        for i, (_, row) in enumerate(selected.iterrows(), start=1)
    }

    def materialize_outputs() -> pd.DataFrame:
        out = pd.DataFrame(all_rows)
        mechanical_path = MarketSkillPaths().mechanical_views
        if mechanical_path.exists():
            mechanical = pd.read_parquet(mechanical_path)
            mechanical["date"] = pd.to_datetime(mechanical["date"], errors="coerce")
            nonfallback_dates = {
                pd.Timestamp(item["date"])
                for item in audits
                if not bool(item.get("fallback")) and item.get("date")
            }
            replace_months = {date + pd.offsets.MonthEnd(0) for date in nonfallback_dates}
            if not out.empty and "date" in out.columns:
                out["date"] = pd.to_datetime(out["date"], errors="coerce")
                out_month = out["date"] + pd.offsets.MonthEnd(0)
                agentic_rows = out[out["date"].isin(nonfallback_dates)].copy()
            else:
                agentic_rows = pd.DataFrame()
            mechanical_month = mechanical["date"] + pd.offsets.MonthEnd(0)
            untouched = mechanical[~mechanical_month.isin(replace_months)].copy()
            out = pd.concat([untouched, agentic_rows], ignore_index=True, sort=False)
        if not out.empty:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.sort_values(["date", "validator_status", "source_asset", "target_asset"]).reset_index(drop=True)
        out.to_parquet(views_path, index=False)
        ep = build_ep_views(out, paths=MarketSkillPaths(ep_views=ep_path))
        audit_path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in sorted(audits, key=lambda x: x["date"])),
            encoding="utf-8",
        )
        pd.DataFrame(raw_rows).to_csv(raw_path, index=False, encoding="utf-8-sig")
        valid = int(out["validator_status"].eq("valid").sum()) if not out.empty else 0
        fallback_count = sum(bool(x["fallback"]) for x in audits)
        report_path.write_text(
            "# Compressed Agentic Market View Run\n\n"
            f"- completed_dates: {len(audits)}\n- total_dates: {len(selected)}\n"
            f"- requests_per_date: 1\n- hidden_calendar_date: true\n"
            f"- views: {len(out)}\n- valid_views: {valid}\n"
            f"- fallback_dates: {fallback_count}\n- ep_views: {len(ep)}\n",
            encoding="utf-8",
        )
        return out

    for idx, (_, card_row) in enumerate(selected.iterrows(), start=1):
        card = pd.Series(card_row)
        date = str(pd.Timestamp(card["date"]).date())
        payload = _compact_card_for_prompt(card, decision_id=decision_ids[date])
        payload = _attach_mechanical_candidates(card, payload, market_cfg)
        card_scope = _single_card_scope(payload)
        result: AgentState = graph.invoke({"card": payload, "findings": [], "errors": [], "raw_responses": []})
        result["card_scope"] = card_scope
        result["decision_id"] = decision_ids[date]
        rows, fallback = _agent_rows(card, result, market_cfg, agent_cfg)
        diagnostics = _fallback_diagnostics(result)
        all_rows.extend(rows)
        audits.append(
            {
                "date": date,
                "decision_id": result.get("decision_id"),
                "fallback": fallback,
                "mode": "compressed_single_call_langgraph",
                **diagnostics,
                "card_scope": result.get("card_scope", {}),
                "plan": result.get("plan", {}),
                "findings": result.get("findings", []),
                "reviewed": result.get("reviewed", []),
                "supervisor": result.get("supervisor", {}),
                "errors": result.get("errors", []),
            }
        )
        for raw in result.get("raw_responses", []):
            raw_rows.append({"date": date, **raw})
        status = "fallback" if fallback else "agentic"
        materialize_outputs()
        line = (
            f"{idx}/{len(selected)} date={date} status={status} "
            f"views={len(rows)} errors={len(result.get('errors', []))} "
            f"reason={diagnostics['fallback_reason']} supervisor_views={diagnostics['supervisor_views']}\n"
        )
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        print(line.strip(), flush=True)
        delay = float(agent_cfg.get("compressed_request_delay_seconds", 0.0))
        if delay > 0 and idx < len(selected):
            time.sleep(delay)
    return materialize_outputs()


def run_agentic_market_views(cards: pd.DataFrame, frequency: str | None = None, config_path: Path | None = None) -> pd.DataFrame:
    agent_cfg = load_agent_config(config_path)
    market_cfg = load_market_skill_config()
    from src.market_skill import cards_for_llm_frequency

    selected = cards_for_llm_frequency(cards, frequency or str(agent_cfg.get("frequency", "month_end")))
    graph = build_graph(agent_cfg, market_cfg)
    all_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    paths_cfg = agent_cfg["paths"]
    views_path = ROOT / paths_cfg["agentic_views"]
    ep_path = ROOT / paths_cfg["agentic_ep_views"]
    audit_path = ROOT / paths_cfg["audit_jsonl"]
    raw_path = ROOT / paths_cfg["raw_responses"]
    report_path = ROOT / paths_cfg["report"]
    progress_path = report_path.with_name("agentic_market_view_progress.log")
    for path in [views_path, ep_path, audit_path, raw_path, report_path, progress_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        f"Agentic Market View run started: frequency={frequency or agent_cfg.get('frequency', 'month_end')}, total_dates={len(selected)}\n",
        encoding="utf-8",
    )

    def materialize_outputs() -> pd.DataFrame:
        out = pd.DataFrame(all_rows)
        mechanical_path = MarketSkillPaths().mechanical_views
        if mechanical_path.exists():
            mechanical = pd.read_parquet(mechanical_path)
            mechanical["date"] = pd.to_datetime(mechanical["date"], errors="coerce")
            if audits and all(bool(item["fallback"]) for item in audits):
                out = mechanical.copy()
            else:
                replaced_dates = set(pd.to_datetime(out["date"], errors="coerce")) if not out.empty else set()
                untouched = mechanical[~mechanical["date"].isin(replaced_dates)].copy()
                out = pd.concat([untouched, out], ignore_index=True, sort=False)
        if not out.empty:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.sort_values(["date", "validator_status", "source_asset", "target_asset"]).reset_index(drop=True)
        out.to_parquet(views_path, index=False)
        custom_paths = MarketSkillPaths(ep_views=ep_path)
        ep = build_ep_views(out, paths=custom_paths)
        audit_path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in sorted(audits, key=lambda x: x["date"])),
            encoding="utf-8",
        )
        pd.DataFrame(raw_rows).to_csv(raw_path, index=False, encoding="utf-8-sig")
        valid = int(out["validator_status"].eq("valid").sum()) if not out.empty else 0
        fallback_count = sum(bool(x["fallback"]) for x in audits)
        report_path.write_text(
            "# Agentic Market View Run\n\n"
            f"- completed_dates: {len(audits)}\n- total_dates: {len(selected)}\n"
            f"- views: {len(out)}\n- valid_views: {valid}\n"
            f"- fallback_dates: {fallback_count}\n- ep_views: {len(ep)}\n",
            encoding="utf-8",
        )
        return out

    decision_ids = {
        str(pd.Timestamp(row["date"]).date()): f"decision_{i:04d}"
        for i, (_, row) in enumerate(selected.sort_values("date").iterrows(), start=1)
    }

    def execute(card_dict: dict[str, Any]) -> tuple[pd.Series, AgentState]:
        card = pd.Series(card_dict)
        date_key = str(pd.Timestamp(card["date"]).date())
        payload = _compact_card_for_prompt(card, decision_id=decision_ids[date_key])
        payload = _attach_mechanical_candidates(card, payload, market_cfg)
        card_scope = _single_card_scope(payload)
        result = graph.invoke({"card": payload, "findings": [], "errors": [], "raw_responses": []})
        result["card_scope"] = card_scope
        result["decision_id"] = decision_ids[date_key]
        return card, result

    workers = min(int(agent_cfg.get("date_workers", 1)), max(1, len(selected)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, row.to_dict()) for _, row in selected.iterrows()]
        total = len(futures)
        for idx, future in enumerate(as_completed(futures), start=1):
            card, result = future.result()
            rows, fallback = _agent_rows(card, result, market_cfg, agent_cfg)
            diagnostics = _fallback_diagnostics(result)
            all_rows.extend(rows)
            date = str(pd.Timestamp(card["date"]).date())
            errors = result.get("errors", [])
            audits.append({
                "date": date,
                "decision_id": result.get("decision_id"),
                "fallback": fallback,
                **diagnostics,
                "card_scope": result.get("card_scope", {}),
                "plan": result.get("plan", {}),
                "findings": result.get("findings", []),
                "reviewed": result.get("reviewed", []),
                "supervisor": result.get("supervisor", {}),
                "errors": errors,
            })
            for raw in result.get("raw_responses", []):
                raw_rows.append({"date": date, **raw})
            status = "fallback" if fallback else "agentic"
            out = materialize_outputs()
            progress_line = (
                f"{idx}/{total} date={date} status={status} "
                f"views={len(rows)} errors={len(errors)} "
                f"reason={diagnostics['fallback_reason']} "
                f"accepted={diagnostics['accepted_reviewed']}/{diagnostics['reviewed_total']} "
                f"supervisor_views={diagnostics['supervisor_views']} "
                f"completed_dates={len(audits)}\n"
            )
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(progress_line)
            print(progress_line.strip(), flush=True)
    out = materialize_outputs()
    return out
