"""
src/thoughts.py  –  GĐ6: thought của ToT.

Bước 6.2 (file này, phần 1): schema thought + kiểm tra hợp lệ.
Bước 6.3 (phần 2): prompt tầng 1 (đề xuất) và tầng 2 (tinh chỉnh) + hàm gọi LLM.

Một thought (theo thiết kế 1.6, đã đổi `config` cho khớp thực đơn SEARCH_SPACE_V2):
{
  "thought_id": "system_2007-01-15_d1_t2",
  "parent_id": null,                      # tầng 1 không có cha
  "depth": 1,
  "hypothesis": {"category": "level_shift",
                 "statement": "Load jumped ~3x four days ago; only recent data is relevant.",
                 "evidence": ["level_7d_vs_365d", "days_since_jump_over_20pct"]},
  "config": {"model": "lgbm", "num_leaves": 31, "target": "ratio", "train_years": 1,
             "n_estimators": 400, "lags": "short"},
  "change_from_parent": null              # tầng 2: sửa gì so với cha và vì sao
}

Nguyên tắc kiểm tra: SỬA những lỗi nhỏ (thiếu núm -> mặc định, giá trị lệch -> giá trị gần nhất
trong thực đơn) thay vì vứt cả thought; mọi chỗ đã sửa đều ghi vào `issues` để báo cáo.
Chỉ coi là KHÔNG HỢP LỆ khi không xác định được mô hình.
"""
import json

from src.evidence import DESCRIPTIONS
from src.models import SEARCH_SPACE_V2

CATEGORIES = ["stable_seasonal", "level_shift", "temp_sensitive", "calendar_anomaly",
              "trend_change", "other"]

# Cấu hình mặc định của từng mô hình = cấu hình baseline ở GĐ4 (núm mới để tắt)
DEFAULT_CONFIG = {
    "snaive": {"lag": 24},
    "mlr": {"train_years": None, "poly": 3, "month_temp": True, "holiday": False, "recent_bias": False},
    "mstl_ets": {"window_weeks": 8},
    "lgbm": {"num_leaves": 31, "target": "ratio", "train_years": None, "n_estimators": 400, "lags": "full"},
}

# Tên LLM hay viết khác đi -> tên chuẩn
MODEL_ALIASES = {"seasonal_naive": "snaive", "naive": "snaive", "snaive": "snaive",
                 "mlr": "mlr", "linear_regression": "mlr", "tao_hong": "mlr",
                 "mstl_ets": "mstl_ets", "ets_mstl": "mstl_ets", "mstl": "mstl_ets", "ets": "mstl_ets",
                 "lgbm": "lgbm", "lightgbm": "lgbm", "gbm": "lgbm"}


# ----------------------------------------------------------------------------
# Chuẩn hóa từng giá trị núm
# ----------------------------------------------------------------------------
def _to_python(v):
    """'true' -> True, 'null'/'all'/'none' -> None, '2' -> 2."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes"):
            return True
        if s in ("false", "no"):
            return False
        if s in ("null", "none", "all", "full_history", "all_history"):
            return None
        try:
            return int(s)
        except ValueError:
            return s
    return v


def _snap(value, allowed):
    """Đưa value về giá trị gần nhất trong danh sách cho phép. Trả về (giá trị, đã_sửa?)."""
    v = _to_python(value)
    if v in allowed:
        return v, False
    nums = [a for a in allowed if isinstance(a, (int, float)) and not isinstance(a, bool)]
    if isinstance(v, (int, float)) and not isinstance(v, bool) and nums:
        if None in allowed and v > max(nums):           # vd train_years = 5 -> "toàn bộ"
            return None, True
        return min(nums, key=lambda a: abs(a - v)), True
    return None, True   # không hiểu được -> báo để dùng mặc định


# ----------------------------------------------------------------------------
# Kiểm tra cấu hình
# ----------------------------------------------------------------------------
def validate_config(cfg, space=SEARCH_SPACE_V2, base=None):
    """Trả về (cấu hình hợp lệ hoặc None, danh sách issues).
    base: cấu hình dùng để lấp núm còn thiếu (tầng 2 = cấu hình của cha); None -> DEFAULT_CONFIG.
    Núm thiếu được lấp KHÔNG tính là issue (thiết kế 1.6 cho phép LLM chỉ chọn 1-2 núm chính)."""
    issues = []
    if not isinstance(cfg, dict):
        return None, ["config is not an object"]
    raw_model = str(cfg.get("model", "")).strip().lower()
    model = MODEL_ALIASES.get(raw_model)
    if model is None or model not in space:
        return None, [f"unknown model '{cfg.get('model')}'"]
    if raw_model != model:
        issues.append(f"model '{raw_model}' -> '{model}'")

    # LLM có thể để các núm trong "params" (như schema cũ) -> gộp ra ngoài
    knobs = {k: v for k, v in cfg.items() if k not in ("model", "params")}
    knobs.update(cfg.get("params") or {})

    fill = base if (base is not None and base.get("model") == model) else DEFAULT_CONFIG[model]
    out = {"model": model}
    for k, allowed in space[model].items():
        if k not in knobs:
            out[k] = fill[k]
            continue
        v, changed = _snap(knobs[k], allowed)
        if changed and v is None and None not in allowed:
            v = DEFAULT_CONFIG[model][k]
            issues.append(f"{k}={knobs[k]!r} not understood -> default {v}")
        elif changed:
            issues.append(f"{k}={knobs[k]!r} -> nearest allowed {v}")
        out[k] = v
    for k in knobs:
        if k not in space[model]:
            issues.append(f"unknown knob '{k}' dropped")
    return out, issues


# ----------------------------------------------------------------------------
# Kiểm tra cả thought
# ----------------------------------------------------------------------------
def validate_thought(raw, thought_id, depth, parent=None):
    """raw: dict LLM trả về cho MỘT thought. parent: thought cha (tầng 2) hoặc None.
    Trả về thought đã chuẩn hóa; thought['valid'] = False nếu không dùng được."""
    issues = []
    raw = raw if isinstance(raw, dict) else {}
    hyp = raw.get("hypothesis") if isinstance(raw.get("hypothesis"), dict) else {}

    category = str(hyp.get("category", "other")).strip().lower()
    if category not in CATEGORIES:
        issues.append(f"category '{category}' -> 'other'")
        category = "other"
    evidence = [e for e in (hyp.get("evidence") or []) if e in DESCRIPTIONS]
    dropped = [e for e in (hyp.get("evidence") or []) if e not in DESCRIPTIONS]
    if dropped:
        issues.append(f"unknown evidence names dropped: {dropped}")

    raw_cfg = raw.get("config")
    if parent is None:
        config, cfg_issues = validate_config(raw_cfg)
        issues += cfg_issues
    else:
        # Tầng 2: giữ họ mô hình và giả thuyết của cha (thiết kế 1.6);
        # núm con không nhắc tới thì lấy theo cha (con là bản TINH CHỈNH của cha)
        pmodel = parent["config"]["model"]
        if isinstance(raw_cfg, dict):
            asked = MODEL_ALIASES.get(str(raw_cfg.get("model", pmodel)).strip().lower())
            if asked != pmodel:
                issues.append(f"child asked model '{raw_cfg.get('model')}' -> kept parent model '{pmodel}'")
            config, cfg_issues = validate_config({**raw_cfg, "model": pmodel}, base=parent["config"])
            issues += cfg_issues
        else:
            config = None
            issues.append("config is not an object")
        if config == parent["config"]:
            issues.append("child identical to parent")
        category = parent["hypothesis"]["category"]
        if not str(hyp.get("statement", "")).strip():
            hyp = {**hyp, "statement": parent["hypothesis"]["statement"]}
        if not evidence:
            evidence = parent["hypothesis"]["evidence"]

    return {
        "thought_id": thought_id,
        "parent_id": parent["thought_id"] if parent else None,
        "depth": depth,
        "hypothesis": {"category": category,
                       "statement": str(hyp.get("statement", "")).strip(),
                       "evidence": evidence},
        "config": config,
        "change_from_parent": (str(raw.get("change_from_parent", "")).strip() or None) if parent else None,
        "valid": config is not None,
        "issues": issues,
    }


def parse_thoughts(obj, prefix, depth, parent=None, n_expected=3):
    """obj: JSON đã parse từ câu trả lời LLM ({"thoughts": [...]} hoặc một list).
    Trả về đúng n_expected thought (thiếu thì bù thought không hợp lệ để ghi log)."""
    items = obj.get("thoughts", []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
    out = []
    for i in range(n_expected):
        raw = items[i] if i < len(items) else None
        th = validate_thought(raw, f"{prefix}_t{i + 1}", depth, parent)
        if raw is None:
            th["issues"].append("missing thought (LLM returned fewer than expected)")
        out.append(th)
    return out


# ============================================================================
# Bước 6.3: PROMPT tầng 1 (đề xuất) và tầng 2 (tinh chỉnh) + gọi LLM
# ============================================================================
# Ý nghĩa từng núm, in vào prompt để LLM biết núm nào ứng với giả thuyết nào
KNOB_HELP = {
    "snaive": {"lag": "24 = copy same hour yesterday, 168 = copy same hour last week"},
    "mlr": {"train_years": "years of training data: 1, 2 or null (= all history)",
            "poly": "degree of the temperature polynomial (1 = linear, 3 = strongly curved)",
            "month_temp": "temperature effect allowed to differ by month",
            "holiday": "add holiday x hour dummies (holiday load shape differs)",
            "recent_bias": "add yesterday's mean error to today's forecast (quick fix for level shifts)"},
    "mstl_ets": {"window_weeks": "weeks of recent data used to extract daily/weekly seasonality "
                                 "(short = adapts fast, long = stable)"},
    "lgbm": {"num_leaves": "tree complexity (15 simple ... 63 complex)",
             "target": "raw = learn load directly, ratio = learn load / yesterday's mean "
                       "(adapts to level shifts)",
             "train_years": "years of training data: 1, 2 or null (= all history)",
             "n_estimators": "number of trees (200 few ... 800 many)",
             "lags": "short = load lags 24/48/72h only, full = also 168/336h (weekly memory)"},
}
MODEL_HELP = {
    "snaive": "seasonal naive: repeats a past day, no fitting",
    "mlr": "multiple linear regression on calendar x temperature (Tao Hong benchmark), no load lags",
    "mstl_ets": "MSTL decomposition (daily + weekly seasonality) + damped-trend ETS, no temperature",
    "lgbm": "LightGBM gradient boosting on load lags, calendar and temperature features",
}
TEMP_LEVEL_HELP = {
    "L1": "No temperature forecast is available for the target day; temperature-aware models "
          "only see temperatures up to yesterday.",
    "L2": "A temperature FORECAST for the target day is available (typical error about 4 °F).",
    "L3": "The ACTUAL temperature of the target day is available (oracle experiment).",
}


# Cửa sổ chấm điểm (bước 6.5): LLM chọn MỘT cửa sổ cho cả lần quyết định (áp dụng cho cả 9 node)
SCORING_WINDOWS = ["365d", "28d", "7d"]
SCORING_HELP = ('"365d" = last 365 days (robust, but blind to very recent changes), '
                '"28d" = last 28 days, "7d" = last 7 days (reacts to a recent change, but noisy)')


def menu_text(space=SEARCH_SPACE_V2):
    lines = []
    for m, grid in space.items():
        lines.append(f"- model \"{m}\": {MODEL_HELP[m]}")
        for k, vals in grid.items():
            shown = ", ".join("null" if v is None else json.dumps(v) for v in vals)
            lines.append(f"    {k} in [{shown}]  -- {KNOB_HELP[m][k]}")
    return "\n".join(lines)


def system_prompt(metric="MAPE"):
    """Phần CỐ ĐỊNH, đặt đầu mọi lần gọi -> được prompt caching (DeepSeek tính giá rẻ ~50 lần)."""
    return f"""You are an expert in electricity load forecasting. Task: choose a model configuration to forecast
the next 24 hourly loads (00:00-23:00 of the target day) of one load series, made at 00:00 of the target day.
Load is known up to 23:00 of the previous day.

You may ONLY use configurations from this menu (any knob you omit takes a default value):
{menu_text()}

Each configuration is scored by its {metric} (%) over a scoring window of past days, using backtests
(lower is better). Scoring windows: {SCORING_HELP}.

A proposal ("thought") = a hypothesis about the current state of the series + a configuration that tests it.
Hypothesis categories: {", ".join(CATEGORIES)}.
"evidence" must list names of the statistics (given in the user message) that support the hypothesis.

Answer with JSON only, no other text."""


def prompt_propose(ev_text, temp_level, k=3, ask_window=True):
    """Tầng 1: đề xuất k chiến lược KHÁC NHAU từ bằng chứng.
    ask_window=True: LLM chọn thêm cửa sổ chấm điểm cho cả lần quyết định (chế độ "llm");
    ask_window=False: cửa sổ cố định 365d (chế độ "S1", dùng cho ablation)."""
    if ask_window:
        window_rule = (f'\nAlso choose "scoring_window" for this decision, one of {SCORING_WINDOWS}. '
                       "It is used to score ALL proposals of this decision, so pick it from the evidence.")
        window_field = '"scoring_window": "...", '
    else:
        window_rule, window_field = "\nAll proposals will be scored on the 365d window.", ""
    return f"""Statistics of the series at the forecast origin:
{ev_text}

Temperature information: {TEMP_LEVEL_HELP[temp_level]}

Propose {k} DIFFERENT strategies. They must differ in hypothesis category or in model.
For each, choose the model and set the 1-3 knobs that matter most for your hypothesis.{window_rule}
Return exactly:
{{{window_field}"thoughts": [{{"hypothesis": {{"category": "...", "statement": "one sentence", "evidence": ["stat_name", "..."]}},
               "config": {{"model": "...", "knob": value}}}}]}}"""


def prompt_refine(ev_text, temp_level, tier1, parent, metric="MAPE", k=3, window="365d"):
    """Tầng 2: tinh chỉnh `parent`, được xem điểm của cả các thought tầng 1."""
    scored = "\n".join(
        f"- {t['thought_id']}: {t['hypothesis']['category']} | {json.dumps(t['config'])} | "
        f"{metric} = {'invalid' if t.get('score') is None else round(t['score'], 3)}"
        for t in tier1)
    return f"""Statistics of the series at the forecast origin:
{ev_text}

Temperature information: {TEMP_LEVEL_HELP[temp_level]}

First-round proposals and their scores ({metric} % over the "{window}" scoring window, lower is better):
{scored}

Refine proposal {parent['thought_id']}:
  hypothesis: {parent['hypothesis']['statement']}
  config: {json.dumps(parent['config'])}

Propose {k} refined configurations. KEEP the model "{parent['config']['model']}" and the hypothesis;
change 1-2 knobs per refinement, each refinement different from the parent and from each other.
Return exactly:
{{"thoughts": [{{"config": {{"model": "{parent['config']['model']}", "knob": value}},
               "change_from_parent": "what you changed and why"}}]}}"""


def _ask(user_prompt, metric, prefix, depth, parent, k, tag, temperature):
    """Gọi LLM -> parse JSON -> chuẩn hóa thought. JSON hỏng thì gọi lại 1 lần (thiết kế 1.6)."""
    from src.llm import call_llm, extract_json
    log = {"prompt": user_prompt, "responses": []}
    obj = None
    for attempt in (1, 2):
        text = call_llm(user_prompt, system=system_prompt(metric), temperature=temperature,
                        max_tokens=900, tag=f"{tag}|try{attempt}")
        log["responses"].append(text)
        try:
            obj = extract_json(text)
            break
        except ValueError as e:
            log.setdefault("errors", []).append(str(e))
    thoughts = parse_thoughts(obj or {}, prefix, depth, parent, n_expected=k)
    log["parsed"] = obj
    return thoughts, log


def generate_tier1(ev_text, temp_level, prefix, metric="MAPE", k=3, tag="", temperature=0.7,
                   ask_window=True):
    """Trả về (thoughts, log, window). window = cửa sổ chấm LLM chọn (hoặc "365d")."""
    thoughts, log = _ask(prompt_propose(ev_text, temp_level, k, ask_window), metric, f"{prefix}_d1", 1,
                         None, k, f"{prefix}|d1|{tag}", temperature)
    window = "365d"
    if ask_window:
        raw = str((log["parsed"] or {}).get("scoring_window", "")).strip().lower() \
            if isinstance(log["parsed"], dict) else ""
        if raw in SCORING_WINDOWS:
            window = raw
        else:
            log["window_issue"] = f"scoring_window '{raw}' -> '365d'"
    return thoughts, log, window


def generate_tier2(ev_text, temp_level, tier1, parent, prefix, metric="MAPE", k=3, tag="",
                   temperature=0.7, window="365d"):
    pid = parent["thought_id"].split("_")[-1]          # vd "t2"
    return _ask(prompt_refine(ev_text, temp_level, tier1, parent, metric, k, window), metric,
                f"{prefix}_d2_{pid}", 2, parent, k, f"{prefix}|d2|{pid}|{tag}", temperature)


# ----------------------------------------------------------------------------
# Chạy thử:
#   python -m src.thoughts       -> kiểm tra bộ chuẩn hóa với câu trả lời giả
#   python -m src.thoughts llm   -> gọi LLM thật (theo .env) cho zone 10, ngày 10/01/2008
# ----------------------------------------------------------------------------
def _demo_llm():
    import time
    from src.data import load_dataset
    from src.evidence import evidence, format_evidence
    from src.llm import settings, STATS
    print("LLM:", settings()["backend"], settings()["model"])
    ev = format_evidence(evidence(load_dataset(10), "2008-01-10"))
    t0 = time.time()
    tier1, log, window = generate_tier1(ev, "L1", "z10_2008-01-10", metric="WAPE")
    print("scoring_window:", window)
    print(f"--- câu trả lời thô ({time.time() - t0:.0f} s) ---")
    for r in log["responses"]:
        print(r)
    print("--- sau khi chuẩn hóa ---")
    for t in tier1:
        print(json.dumps({k: t[k] for k in ("thought_id", "valid", "hypothesis", "config", "issues")},
                         ensure_ascii=False))
    print(STATS)


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "llm":
    _demo_llm()
elif __name__ == "__main__":
    # Một câu trả lời "bẩn" giống LLM nhỏ hay trả: tên mô hình viết khác, giá trị lệch thực đơn,
    # núm lạ, category lạ, mô hình không có trong thực đơn
    fake = {"thoughts": [
        {"hypothesis": {"category": "level_shift", "statement": "Load tripled 4 days ago.",
                        "evidence": ["level_7d_vs_365d", "made_up_stat"]},
         "config": {"model": "LightGBM", "num_leaves": 40, "train_years": "1", "lags": "short"}},
        {"hypothesis": {"category": "temperature", "statement": "Cold spell."},
         "config": {"model": "mlr", "params": {"train_years": 5, "month_temp": "true", "alpha": 0.1}}},
        {"hypothesis": {"category": "stable_seasonal", "statement": "Same as last week."},
         "config": {"model": "prophet"}},
    ]}
    tier1 = parse_thoughts(fake, "demo_d1", depth=1)
    print("=== Tầng 1 ===")
    for t in tier1:
        print(json.dumps({k: t[k] for k in ("thought_id", "valid", "config", "issues")}, ensure_ascii=False))

    # Con của thought 1: đòi đổi sang mlr (không được phép), chỉ nói num_leaves -> các núm khác lấy theo cha
    tier2 = parse_thoughts({"thoughts": [{"config": {"model": "mlr", "num_leaves": 63},
                                          "change_from_parent": "more leaves for a sharper profile"}]},
                           "demo_d2_p1", depth=2, parent=tier1[0], n_expected=1)
    print("=== Tầng 2 (con của t1) ===")
    for t in tier2:
        print(json.dumps({k: t[k] for k in ("thought_id", "parent_id", "valid", "config", "issues")},
                         ensure_ascii=False))
