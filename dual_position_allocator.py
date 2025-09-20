#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dual_position_allocator.py

双仓位模型（底仓 + 战术仓）· 多模式版（支持输入当前仓位）
- 输入：当前全部现金（--cash），当前已持有市值（--equity）及可选分解（--core_held / --tactical_held）
- 输出：今天建议的底仓投入、战术仓加仓、剩余现金（只加不减），并展示目标与当前的对齐情况

内置模式（--mode）：
  build         建仓：只买底仓（core），战术仓强制 0（不随信号买入）
  balanced      均衡（默认）：60% 底仓 + 40% 战术
  conservative  保守：70% 底仓 + 30% 战术
  aggressive    激进：40% 底仓 + 60% 战术
  trend         顺势：强趋势+低波动 → 战术档位上调一档（减少踏空）
  contrarian    逆向：贪婪&乏回撤降档；恐惧或深回撤升档

信号逻辑（轻量化，和 buy_decider 同源）：
- 情绪优先（FGI）+ 弱化趋势 + 回撤分（相对 MA20）+ 分位加成
- 反人性：VIX ≥ 98% 分位 → 至少 BUY_50；且 (FGI≤20 或 dev20≤-8%) → 直接 ALL_IN
- 美元极强（UUP ≥ 90% 分位）仅降一档（不打回 WAIT）
- 智能波动护栏（方案B）：仅在“高波动(vol20≥0.30) + 主要非下跌驱动(downfrac<0.6) + 未触发反人性”时，
  把 ALL_IN → BUY_50（避免在非下跌驱动的躁市里满仓）

依赖（建议 venv）：
  pip install yfinance pandas numpy requests fear-and-greed pytz python-dateutil

用法示例：
  python dual_position_allocator.py --cash 30000 --mode build --ticker VOO
  python dual_position_allocator.py --cash 30000 --equity 50000 --mode balanced --ticker QQQ
  python dual_position_allocator.py --cash 30000 --equity 50000 --core_held 30000 --tactical_held 20000 --mode contrarian
"""

import sys
import io
import json
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, List

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from dateutil import parser as dateparser
from dateutil.tz import tzlocal

# ------------------------ FGI Helpers ------------------------

CSV_MIRRORS: List[str] = [
    "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/refs/heads/main/datasets/cnn_fear_greed_daily.csv",
    "https://cdn.jsdelivr.net/gh/whit3rabbit/fear-greed-data/datasets/cnn_fear_greed_daily.csv",
    "https://rawcdn.githack.com/whit3rabbit/fear-greed-data/refs/heads/main/datasets/cnn_fear_greed_daily.csv",
    "https://ghproxy.com/https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/refs/heads/main/datasets/cnn_fear_greed_daily.csv",
]

def fgi_label(val: float) -> str:
    return ("extreme fear" if val <= 25 else
            "fear" if val <= 45 else
            "neutral" if val <= 55 else
            "greed" if val <= 75 else
            "extreme greed")

def try_live_fgi(target_date: dt.date) -> Optional[Tuple[float, str, dt.datetime, str]]:
    try:
        today = dt.datetime.now(dt.timezone.utc).date()
        if abs((target_date - today).days) <= 2:
            import fear_and_greed
            res = fear_and_greed.get()
            return float(res.value), str(res.description), res.last_update, "live_cnn"
    except Exception as e:
        sys.stderr.write(f"[WARN] live FGI failed: {e}\n")
    return None

def try_csv_mirrors(target_date: dt.date, retries: int = 2, delay_sec: float = 0.8) -> Optional[Tuple[float, str, dt.datetime, str]]:
    import time
    last_err: Optional[Exception] = None
    for _ in range(retries):
        for url in CSV_MIRRORS:
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                cols = {c.lower().strip(): c for c in df.columns}
                date_col = cols.get("date") or list(df.columns)[0]
                val_col = cols.get("value") or cols.get("fgi") or list(df.columns)[1]
                df[date_col] = pd.to_datetime(df[date_col]).dt.date

                mask = df[date_col] == target_date
                if mask.any():
                    val = float(df.loc[mask, val_col].iloc[0])
                    ts = dt.datetime.combine(target_date, dt.time(0,0,0), tzinfo=dt.timezone.utc)
                    return val, fgi_label(val), ts, f"csv:{url}"
                prev = df[df[date_col] <= target_date].sort_values(date_col)
                if not prev.empty:
                    use_date = prev[date_col].iloc[-1]
                    val = float(prev[val_col].iloc[-1])
                    ts = dt.datetime.combine(use_date, dt.time(0,0,0), tzinfo=dt.timezone.utc)
                    return val, fgi_label(val), ts, f"csv_prev:{url}"
            except Exception as e:
                last_err = e
                continue
        time.sleep(delay_sec)
    if last_err:
        sys.stderr.write(f"[WARN] FGI CSV mirrors failed: {last_err}\n")
    return None

def get_fgi_for_date(target_date: dt.date, manual: Optional[float] = None) -> Tuple[float, str, dt.datetime, str]:
    if manual is not None:
        return float(manual), fgi_label(float(manual)), dt.datetime.combine(target_date, dt.time(0,0,0), tzinfo=dt.timezone.utc), "manual"
    live = try_live_fgi(target_date)
    if live is not None:
        return live
    csv = try_csv_mirrors(target_date)
    if csv is not None:
        return csv
    sys.stderr.write("[WARN] All FGI sources failed; fallback FGI=50. Use --fgi to override.\n")
    val = 50.0
    return val, fgi_label(val), dt.datetime.combine(target_date, dt.time(0,0,0), tzinfo=dt.timezone.utc), "fallback:neutral50"

# ------------------------ Price & Context ------------------------

def get_price_and_ma_asof(target_date: dt.date, ticker: str = "QQQ", lookback_days: int = 320):
    start = target_date - dt.timedelta(days=lookback_days*2)
    end = target_date + dt.timedelta(days=5)
    data = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=False, threads=True)
    if data.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    close = data["Close"].dropna()

    if target_date in set(close.index.date):
        use_idx = close.index[close.index.date == target_date][0]
    else:
        prev_idx = close.index[close.index.date <= target_date]
        if len(prev_idx) == 0:
            raise RuntimeError(f"No trading data on or before {target_date} for {ticker}.")
        use_idx = prev_idx[-1]

    price = close.loc[use_idx]
    price = float(price.item() if hasattr(price, "item") else price)

    ma20  = close.rolling(20).mean().loc[use_idx]
    ma20  = float(ma20.item() if hasattr(ma20, "item") else ma20)

    ma50  = close.rolling(50).mean().loc[use_idx]
    ma50  = float(ma50.item() if hasattr(ma50, "item") else ma50)

    ma200 = close.rolling(200).mean().loc[use_idx]
    ma200 = float(ma200.item() if hasattr(ma200, "item") else ma200)

    return use_idx.date(), price, ma20, ma50, ma200

def rolling_percentile(series: pd.Series, window: int, value: float) -> float:
    hist = series.dropna().tail(window)
    if hist.empty:
        return 0.5
    m = (hist < value).mean()
    return float(m.item() if hasattr(m, "item") else m)

def realized_vol_20(close: pd.Series) -> float:
    ret = np.log(close / close.shift(1)).dropna()
    if len(ret) < 20:
        return 0.2
    vol = ret.tail(20).std() * np.sqrt(252)
    return float(vol.item() if hasattr(vol, "item") else vol)

def downside_vol_20(close: pd.Series) -> float:
    ret = np.log(close / close.shift(1)).dropna()
    if len(ret) < 20:
        return 0.2
    last20 = ret.tail(20)
    neg = last20[last20 < 0]
    if len(neg) < 5:
        return 0.0
    vol = neg.std() * np.sqrt(252)
    return float(vol.item() if hasattr(vol, "item") else vol)

def fetch_close(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=True)
    return df["Close"].dropna()

def build_context_for_date(target_date, ticker="QQQ", lookback_days=500):
    start = (pd.Timestamp(target_date) - pd.Timedelta(days=lookback_days*2)).date().isoformat()
    end   = (pd.Timestamp(target_date) + pd.Timedelta(days=5)).date().isoformat()
    ctx = {}

    px = fetch_close(ticker, start, end)
    use_idx = px.index[px.index.date <= target_date][-1]
    ctx["px_close"] = px
    ctx["use_idx"] = use_idx

    ctx["px_vol20"] = realized_vol_20(px.loc[:use_idx])
    ctx["px_downvol20"] = downside_vol_20(px.loc[:use_idx])

    try:
        vix = fetch_close("^VIX", start, end)
        vix_val = vix.loc[:use_idx].iloc[-1]
        vix_val = float(vix_val.item() if hasattr(vix_val, "item") else vix_val)
        vix_pct = rolling_percentile(vix.loc[:use_idx], 252*2, vix_val)
        ctx["vix_val"], ctx["vix_pct"] = vix_val, vix_pct
    except Exception:
        ctx["vix_val"], ctx["vix_pct"] = None, 0.5

    try:
        uup = fetch_close("UUP", start, end)
        uup_val = uup.loc[:use_idx].iloc[-1]
        uup_val = float(uup_val.item() if hasattr(uup_val, "item") else uup_val)
        uup_pct = rolling_percentile(uup.loc[:use_idx], 252*2, uup_val)
        ctx["uup_val"], ctx["uup_pct"] = uup_val, uup_pct
    except Exception:
        ctx["uup_val"], ctx["uup_pct"] = None, 0.5

    return ctx

# ------------------------ Scoring ------------------------

TIER_ORDER = ["WAIT", "BUY_10", "BUY_20", "BUY_30", "BUY_50", "BUY_75", "ALL_IN"]
TIER_FRAC  = {
    "WAIT":   0.00,
    "BUY_10": 0.10,
    "BUY_20": 0.20,
    "BUY_30": 0.30,
    "BUY_50": 0.50,
    "BUY_75": 0.75,
    "ALL_IN": 1.00,
}

def score_to_tier(score: float):
    """
    7档映射（可按偏好微调阈值）：
      >= 11.0  → ALL_IN      (100%)
      >=  9.0  → BUY_75      (75%)
      >=  7.0  → BUY_50      (50%)
      >=  5.5  → BUY_30      (30%)
      >=  4.5  → BUY_20      (20%)
      >=  3.5  → BUY_10      (10%)
      <   3.5  → WAIT        (0%)
    """
    if score >= 11.0: return ("ALL_IN", TIER_FRAC["ALL_IN"])
    if score >=  9.0: return ("BUY_75", TIER_FRAC["BUY_75"])
    if score >=  7.0: return ("BUY_50", TIER_FRAC["BUY_50"])
    if score >=  5.5: return ("BUY_30", TIER_FRAC["BUY_30"])
    if score >=  4.5: return ("BUY_20", TIER_FRAC["BUY_20"])
    if score >=  3.5: return ("BUY_10", TIER_FRAC["BUY_10"])
    return ("WAIT", TIER_FRAC["WAIT"])

@dataclass
class Inputs:
    fgi: float
    price: float
    ma20: float
    ma50: float
    ma200: float
    ticker: str
    px_close: pd.Series
    use_idx: pd.Timestamp
    px_vol20: float
    px_downvol20: float
    vix_pct: float
    uup_pct: float

def score_signal(inputs: Inputs) -> dict:
    """Return score & tactical tier for allocator."""
    INDEX_WHITELIST = {"QQQ","QQQM","SPY","VOO","IVV","VTI","IWY","SPMO","^NDX","NDX","^GSPC"}
    ticker = getattr(inputs, "ticker", "INDEX")
    is_index = (ticker in INDEX_WHITELIST) or ("^" in ticker) or (len(ticker) <= 5)

    # 1) FGI score（情绪优先）
    fgi = inputs.fgi
    if fgi <= 10: s_fgi = 6
    elif fgi <= 20: s_fgi = 5
    elif fgi <= 35: s_fgi = 3
    elif fgi <= 50: s_fgi = 1
    else: s_fgi = 0

    # 2) 趋势分（恐惧时弱化权重）
    p, ma20, ma50, ma200 = inputs.price, inputs.ma20, inputs.ma50, inputs.ma200
    if p > ma50 > ma200: t_raw = 3
    elif p > ma200 and ma50 >= ma200: t_raw = 2
    elif p > ma200 and ma50 < ma200: t_raw = 1
    else: t_raw = -2

    if fgi <= 10: w_trend = 0.25
    elif fgi <= 20: w_trend = 0.45
    elif fgi <= 50: w_trend = 0.75
    else: w_trend = 1.0
    s_trend = round(t_raw * w_trend, 1)

    # 3) 回撤分（MA20 偏离 + 分位加成）
    dev20 = (p / ma20 - 1.0) if ma20 > 0 else 0.0
    dev_series = (inputs.px_close / inputs.px_close.rolling(20).mean() - 1).dropna()
    dev20_pct = rolling_percentile(dev_series.loc[:inputs.use_idx], 252*2, dev20)

    if -0.25 <= dev20 <= -0.15: base_pull = 5
    elif -0.15 < dev20 <= -0.10: base_pull = 4
    elif -0.10 < dev20 <= -0.05: base_pull = 3
    elif -0.05 < dev20 < 0.03:   base_pull = 1
    elif dev20 >= 0.05:          base_pull = -2
    else:                        base_pull = 0

    s_pull = base_pull + (1 if dev20_pct <= 0.2 and base_pull >= 1 else 0)

    # 4) 初步分数与档位
    S = float(round(s_fgi + s_trend + s_pull, 2))
    action, size = score_to_tier(S)

    # 5) 宏观/逆向护栏
    dev200 = (p / ma200 - 1.0) if ma200 > 0 else 0.0

    contrarian_vix = bool(inputs.vix_pct is not None and inputs.vix_pct >= 0.98)
    if contrarian_vix:
        # 至少提升到 50%
        if action in ("WAIT", "BUY_10", "BUY_20", "BUY_30"):
            action, size = "BUY_50", TIER_FRAC["BUY_50"]
        # 极端条件 → ALL_IN
        if (fgi <= 20 or dev20 <= -0.08):
            action, size = "ALL_IN", TIER_FRAC["ALL_IN"]

    if inputs.uup_pct is not None and inputs.uup_pct >= 0.90:
        # 仅降一档（不打回 WAIT）
        idx = TIER_ORDER.index(action)
        if idx > 0:
            action = TIER_ORDER[idx - 1]
            size = TIER_FRAC[action]

    if p <= ma200 and dev200 <= -0.10:
        if is_index:
            if action == "ALL_IN":
                pass
            elif (fgi <= 15) or (dev20 <= -0.10):
                action, size = "ALL_IN", TIER_FRAC["ALL_IN"]
        else:
            # 个股最多 BUY_50
            if action in ("ALL_IN", "BUY_75"):
                action, size = "BUY_50", TIER_FRAC["BUY_50"]

    # 6) 智能波动护栏（方案B）
    vol20 = getattr(inputs, "px_vol20", None)
    downvol20 = getattr(inputs, "px_downvol20", None)
    downfrac = None
    if (downvol20 is not None) and vol20:
        downfrac = downvol20 / max(1e-6, vol20)

    soft_vol_guard = False
    if (not contrarian_vix) \
       and (vol20 is not None) and (vol20 >= 0.30) \
       and (downfrac is not None) and (downfrac < 0.6) \
       and action == "ALL_IN":
        action, size = "BUY_50", TIER_FRAC["BUY_50"]
        soft_vol_guard = True

    return {
        "score": S,
        "tier": action,
        "tier_fraction_of_tactical": size,
        "risk_switches": {
            "vix_pct": round(inputs.vix_pct, 3) if inputs.vix_pct is not None else None,
            "uup_pct": round(inputs.uup_pct, 3) if inputs.uup_pct is not None else None,
            "dev200": round(dev200, 4),
            "vol20": round(vol20, 4) if vol20 is not None else None,
            "downvol20": round(downvol20, 4) if downvol20 is not None else None,
            "downfrac": round(downfrac, 3) if downfrac is not None else None,
            "contrarian_vix_triggered": contrarian_vix,
            "soft_vol_guard": soft_vol_guard
        },
        "breakdown": {
            "fgi_score": s_fgi,
            "trend_raw": t_raw,
            "trend_weight": w_trend,
            "trend_score": s_trend,
            "pullback_score": s_pull,
            "dev20": round(dev20, 4),
        }
    }

# ------------------------ Mode presets & tier adjust ------------------------

MODE_PRESETS = {
    # 底仓/战术仓比例（对总资金）
    "build":        {"core_ratio": 0.30, "tactical_ratio": 0.70},
    "balanced":     {"core_ratio": 0.60, "tactical_ratio": 0.40},
    "conservative": {"core_ratio": 0.70, "tactical_ratio": 0.30},
    "aggressive":   {"core_ratio": 0.40, "tactical_ratio": 0.60},
    "trend":        {"core_ratio": 0.60, "tactical_ratio": 0.40},
    "contrarian":   {"core_ratio": 0.60, "tactical_ratio": 0.40},
}

def _bump_tier(tier: str, steps: int) -> str:
    i = TIER_ORDER.index(tier)
    i = max(0, min(len(TIER_ORDER)-1, i + steps))
    return TIER_ORDER[i]

def adjust_tier_by_mode(mode: str, sig: dict):
    """
    根据模式微调战术档位（仅影响战术仓的投入比例），返回 (new_tier, new_frac, explain)

    变更点：
    - build 模式：战术仓强制不买（核心：只买底仓）
    - trend / contrarian 保持原有微调逻辑
    """
    tier = sig["tier"]
    frac = sig["tier_fraction_of_tactical"]
    explain = {"base_tier": tier, "base_frac": frac, "adjust": "none"}

    # 需要的上下文字段
    dev20 = None
    trend_raw = None
    vol20 = None
    downfrac = None
    fgi_score = None

    if "breakdown" in sig:
        dev20 = sig["breakdown"].get("dev20", None)
        trend_raw = sig["breakdown"].get("trend_raw", None)
        fgi_score = sig["breakdown"].get("fgi_score", None)
    if "risk_switches" in sig:
        vol20 = sig["risk_switches"].get("vol20", None)
        downfrac = sig["risk_switches"].get("downfrac", None)

    # --- build 模式：战术仓强制 0 ---
    if mode == "build":
        explain["adjust"] = "build: core-only (tactical 0)"
        return "WAIT", 0.0, explain

    # --- trend 模式：强趋势 + 低波动 → 升一档 ---
    if mode == "trend":
        if (trend_raw is not None and trend_raw >= 2) and (vol20 is not None and vol20 <= 0.15):
            new_tier = _bump_tier(tier, +1)
            explain["adjust"] = "trend:+1 (strong trend & low vol)"
            return new_tier, TIER_FRAC[new_tier], explain

    # --- contrarian 模式 ---
    if mode == "contrarian":
        if dev20 is not None and dev20 <= -0.05:
            new_tier = _bump_tier(tier, +1)
            explain["adjust"] = "contrarian:+1 (deep pullback)"
            return new_tier, TIER_FRAC[new_tier], explain
        if (dev20 is not None and dev20 >= 0.03) and (fgi_score == 0):
            new_tier = _bump_tier(tier, -1)
            explain["adjust"] = "contrarian:-1 (greed & extended)"
            return new_tier, TIER_FRAC[new_tier], explain

    # 其它模式：不调整
    return tier, frac, explain

# ------------------------ Main (Allocator) ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cash", type=float, required=True, help="当前可用现金（单位：元）")
    ap.add_argument("--equity", type=float, default=0.0,
                    help="当前已持有的同标的市值（单位：元）。用于按目标比例补齐，不会减仓。")
    ap.add_argument("--core_held", type=float, default=None,
                    help="当前已持有中，视为底仓的部分（元）。若不填则默认优先占满底仓目标。")
    ap.add_argument("--tactical_held", type=float, default=None,
                    help="当前已持有中，视为战术仓的部分（元）。若不填则根据 equity 与 core_held 推断。")

    ap.add_argument("--date", help="YYYY-MM-DD (local date). Default: today.", default=None)
    ap.add_argument("--ticker", help="Ticker (default QQQ). e.g., VOO, IWY, ^NDX", default="QQQ")
    ap.add_argument("--fgi", type=float, help="Manual override of FGI value (0-100).")

    ap.add_argument("--mode", choices=["build","balanced","conservative","aggressive","trend","contrarian"],
                    default="balanced", help="分配模式（默认 balanced）")

    # 可覆盖模式预设（非必填）
    ap.add_argument("--core_ratio", type=float, default=None, help="覆盖目标底仓占比（0~1，对总资金）")
    ap.add_argument("--tactical_ratio", type=float, default=None, help="覆盖目标战术占比（0~1，对总资金）")
    args = ap.parse_args()

    if args.cash < 0: raise SystemExit("--cash 必须 >= 0")
    if args.equity < 0: raise SystemExit("--equity 必须 >= 0")

    target_date = dateparser.parse(args.date).date() if args.date else dt.datetime.now(tzlocal()).date()

    # 1) 获取指标
    fgi_val, fgi_desc, fgi_ts, fgi_source = get_fgi_for_date(target_date, manual=args.fgi)
    use_date, price, ma20, ma50, ma200 = get_price_and_ma_asof(target_date, ticker=args.ticker)
    ctx = build_context_for_date(target_date, ticker=args.ticker)

    # 2) 战术信号
    sig = score_signal(Inputs(
        fgi=fgi_val,
        price=price, ma20=ma20, ma50=ma50, ma200=ma200,
        ticker=args.ticker,
        px_close=ctx["px_close"], use_idx=ctx["use_idx"],
        px_vol20=ctx["px_vol20"], px_downvol20=ctx["px_downvol20"],
        vix_pct=ctx["vix_pct"], uup_pct=ctx["uup_pct"]
    ))
    if not isinstance(sig, dict):
        raise RuntimeError("score_signal returned no result (None). Please check implementation.")

    # 3) 应用模式预设/覆盖
    preset = MODE_PRESETS[args.mode].copy()
    if args.core_ratio is not None:     preset["core_ratio"] = float(args.core_ratio)
    if args.tactical_ratio is not None: preset["tactical_ratio"] = float(args.tactical_ratio)
    if preset["core_ratio"] < 0 or preset["tactical_ratio"] < 0 or preset["core_ratio"] + preset["tactical_ratio"] > 1.0 + 1e-9:
        raise SystemExit("core_ratio + tactical_ratio 必须 ≤ 1 且均为非负。")

    # 4) 依据模式微调战术档位（build 强制 0）
    adj_tier, adj_frac, adj_explain = adjust_tier_by_mode(args.mode, sig)
    sig_out = sig.copy()
    sig_out["tier_after_mode"] = adj_tier
    sig_out["tier_fraction_of_tactical_after_mode"] = adj_frac
    sig_out["mode_adjust_explain"] = adj_explain

    # ------------------------ 5) 分配（只加不减，考虑当前仓位） ------------------------
    cash = float(args.cash)
    equity = float(args.equity)
    total_cap = cash + equity

    # 5.1 今日目标（对总资金）
    core_target_cap = total_cap * preset["core_ratio"]
    tactical_pool_cap = total_cap * preset["tactical_ratio"]
    tactical_target_cap = tactical_pool_cap * adj_frac  # 今日战术目标只用“战术池”的一部分

    # 5.2 推断当前已持有的底/战术归属
    if (args.core_held is not None) and (args.tactical_held is not None):
        core_held = float(args.core_held)
        tactical_held = float(args.tactical_held)
        if core_held < 0 or tactical_held < 0 or abs(core_held + tactical_held - equity) > 1e-6:
            raise SystemExit("--core_held + --tactical_held 必须等于 --equity，且均为非负。")
    else:
        # 默认：优先把已有持仓计入底仓，最多不超过“目标底仓”；剩余计入战术
        core_held = min(equity, core_target_cap)
        tactical_held = max(0.0, equity - core_held)

    # 5.3 计算今天需要补的量（只加不减 & 不超过现金）
    core_need = max(0.0, core_target_cap - core_held)
    core_buy = min(core_need, cash)
    cash_after_core = cash - core_buy

    tactical_need = max(0.0, tactical_target_cap - tactical_held)
    tactical_buy = min(tactical_need, cash_after_core)

    total_invest = core_buy + tactical_buy
    remaining_cash = cash - total_invest

    # 5.4 更新“加仓后”的估计
    new_equity = equity + total_invest
    actual_core_after = core_held + core_buy
    actual_tac_after = tactical_held + tactical_buy
    total_fraction_of_capital = new_equity / max(total_cap, 1e-9)

    out_allocation = {
        "core_invest": round(core_buy, 2),
        "tactical_invest": round(tactical_buy, 2),
        "remaining_cash": round(remaining_cash, 2),
        "total_invest": round(total_invest, 2),
        "total_fraction_of_capital": round(total_fraction_of_capital, 4),
        "targets": {
            "core_target_cap": round(core_target_cap, 2),
            "tactical_target_cap": round(tactical_target_cap, 2),
            "tactical_pool_cap": round(tactical_pool_cap, 2)
        },
        "current_before": {
            "equity": round(equity, 2),
            "core_held_used": round(core_held, 2),
            "tactical_held_used": round(tactical_held, 2)
        },
        "current_after": {
            "equity": round(new_equity, 2),
            "core_estimated": round(actual_core_after, 2),
            "tactical_estimated": round(actual_tac_after, 2)
        }
    }

    # ------------------------ 输出 ------------------------
    out = {
        "request_date": target_date.isoformat(),
        "used_trading_date": use_date.isoformat(),
        "ticker": args.ticker,
        "mode": args.mode,
        "ratios": preset,
        "inputs": {
            "cash": cash,
            "equity": equity,
            "core_held": None if args.core_held is None else float(args.core_held),
            "tactical_held": None if args.tactical_held is None else float(args.tactical_held)
        },
        "fgi": {
            "value": fgi_val, "label": fgi_desc,
            "source_ts": fgi_ts.isoformat(), "source": fgi_source
        },
        "price": price, "ma20": ma20, "ma50": ma50, "ma200": ma200,
        "signal": sig_out,
        "allocation_today": out_allocation,
        "notes": {
            "core": "底仓为长期持有部分，按模式“目标底仓”补齐；只加不减。",
            "tactical": "战术仓在其额度内按 7 档投入：WAIT/10%/20%/30%/50%/75%/100%（可被模式微调）。",
            "build_mode": "建仓模式下战术仓强制 0，仅买底仓。",
        }
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
