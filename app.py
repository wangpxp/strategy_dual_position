import streamlit as st
import subprocess, sys, json, shlex

st.set_page_config(page_title="Dual Position Allocator", layout="wide")
st.title("📈 双仓位模型（底仓 + 战术仓）· 极简模式")

with st.sidebar:
    st.header("参数")
    total_assets = st.number_input("总资产（视作全部现金，元）", value=100000.0, min_value=0.0, step=1000.0)
    ticker = st.text_input("Ticker（如 QQQ / VOO / IWY / ^NDX）", value="QQQ")
    mode = st.selectbox("模式", ["build","balanced","conservative","aggressive","trend","contrarian"], index=1)
    run_btn = st.button("🚀 运行")

st.info(
    "本页面为 **极简模式**：把“总资产”全部视为现金传入后端脚本（不传 equity）。"
    "如你已经持有部分仓位，请改用高级版页面或命令行传入 `--equity / --core_held / --tactical_held`。",
    icon="ℹ️"
)

# 组装命令（只传 3 个参数）
cmd = [
    sys.executable,
    "dual_position_allocator.py",
    "--cash", str(total_assets),
    "--ticker", ticker,
    "--mode", mode
]

st.caption("命令预览：`" + " ".join(shlex.quote(x) for x in cmd) + "`")

def fmt_money(x):
    try:
        return f"¥{float(x):,.2f}"
    except:
        return str(x)

if run_btn:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=120).decode("utf-8")
        data = json.loads(out)
    except subprocess.CalledProcessError as e:
        st.error("脚本运行失败（CalledProcessError）")
        st.code(e.output.decode("utf-8"), language="bash")
    except json.JSONDecodeError:
        st.error("脚本返回的不是有效 JSON，已显示原文：")
        st.code(out, language="json")
    else:
        # 关键信息
        alloc = data.get("allocation_today", {}) or {}
        sig = data.get("signal", {}) or {}
        ratios = data.get("ratios", {}) or {}
        fgi = data.get("fgi", {}) or {}

        core_invest = alloc.get("core_invest", 0.0)
        tactical_invest = alloc.get("tactical_invest", 0.0)
        total_invest = alloc.get("total_invest", 0.0)
        remaining_cash = alloc.get("remaining_cash", 0.0)
        total_frac = alloc.get("total_fraction_of_capital", 0.0)

        tier = sig.get("tier_after_mode") or sig.get("tier", "WAIT")
        tier_frac = sig.get("tier_fraction_of_tactical_after_mode",
                            sig.get("tier_fraction_of_tactical", 0.0))

        # —— 醒目的仓位卡片（简洁视图）——
        st.subheader("📌 今日建议（简洁视图）")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("底仓投入 (Core)", fmt_money(core_invest))
        c2.metric("战术投入 (Tactical)", fmt_money(tactical_invest))
        c3.metric("当前合计", fmt_money(total_invest))
        c4.metric("剩余现金", fmt_money(remaining_cash))

        # “今日投入 / 总资产”的可视化（极简模式下等于 invest / cash）
        st.progress(min(max(total_frac, 0.0), 1.0),
                    text=f"今日投入占总资产：{total_frac*100:.1f}%")

        # 档位徽章 + 关键信号
        badge_color = "#10b981" if str(tier).startswith("BUY") or str(tier) == "ALL_IN" else "#6b7280"
        ticker_used = data.get("ticker", ticker)
        mode_used = data.get("mode", mode)

        dev20_val = sig.get("breakdown", {}).get("dev20", "—")
        fgi_val = fgi.get("value", None)
        fgi_lab = fgi.get("label", "—")
        fgi_txt = f"{fgi_val:.2f}" if isinstance(fgi_val, (int, float)) else "—"

        st.markdown(
            f"""
            <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
              <span style="background:{badge_color};color:white;padding:6px 10px;border-radius:999px;font-weight:600;">
                档位：{tier}（战术比例 {tier_frac*100:.0f}%）
              </span>
              <span style="background:#111827;color:#e5e7eb;padding:6px 10px;border-radius:8px;">
                标的：{ticker_used} · 模式：{mode_used}
              </span>
              <span style="background:#f59e0b;color:#111827;padding:6px 10px;border-radius:8px;">
                FGI：{fgi_txt}（{fgi_lab}）
              </span>
              <span style="background:#374151;color:#e5e7eb;padding:6px 10px;border-radius:8px;">
                MA20 偏离 dev20：{dev20_val}
              </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # 简要表格（可选）
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            st.caption("底/战术目标比")
            st.table({
                "项目": ["Core","Tactical"],
                "比例": [ratios.get("core_ratio","—"), ratios.get("tactical_ratio","—")]
            })
        with bc2:
            st.caption("价格与均线")
            st.table({
                "指标": ["Price","MA20","MA50","MA200"],
                "数值": [data.get("price","—"), data.get("ma20","—"),
                        data.get("ma50","—"), data.get("ma200","—")]
            })
        with bc3:
            st.caption("风控要点（节选）")
            rs = sig.get("risk_switches", {}) or {}
            st.table({
                "项":["VIX分位","UUP分位","dev200","vol20","downfrac"],
                "值":[rs.get("vix_pct","—"), rs.get("uup_pct","—"),
                     rs.get("dev200","—"), rs.get("vol20","—"),
                     rs.get("downfrac","—")]
            })

        # 原始 JSON（可展开）
        with st.expander("🧰 详细/原始 JSON（可展开）", expanded=False):
            st.json(data)
