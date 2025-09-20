import streamlit as st
import subprocess, sys, json, shlex

st.set_page_config(page_title="Dual Position Allocator", layout="wide")
st.title("📈 双仓位模型（底仓 + 战术仓）")

# ---------- 输入区域 ----------
with st.sidebar:
    st.header("参数")
    cash = st.number_input("现金 (元)", value=30000.0, min_value=0.0, step=1000.0)
    ticker = st.text_input("Ticker（如 QQQ / VOO / IWY / ^NDX）", value="VOO")
    mode = st.selectbox("模式", ["build","balanced","conservative","aggressive","trend","contrarian"], index=1)
    date = st.text_input("日期 YYYY-MM-DD（留空=今天）", value="")
    fgi = st.text_input("手动 FGI 覆盖（0-100，可空）", value="")
    equity = st.text_input("当前已持有市值 equity（元，可空）", value="")
    core_ratio = st.text_input("覆盖底仓比例 core_ratio 0-1（可空）", value="")
    tactical_ratio = st.text_input("覆盖战术比例 tactical_ratio 0-1（可空）", value="")
    run_btn = st.button("🚀 运行")

# 组装命令（直接调用你的脚本）
cmd = [sys.executable, "dual_position_allocator.py", "--cash", str(cash), "--ticker", ticker, "--mode", mode]
if date.strip(): cmd += ["--date", date.strip()]
if fgi.strip(): cmd += ["--fgi", fgi.strip()]
if equity.strip(): cmd += ["--equity", equity.strip()]
if core_ratio.strip(): cmd += ["--core_ratio", core_ratio.strip()]
if tactical_ratio.strip(): cmd += ["--tactical_ratio", tactical_ratio.strip()]

st.caption("命令预览：`" + " ".join(shlex.quote(x) for x in cmd) + "`")

# ---------- 执行并展示结果 ----------
def format_money(x):
    try:
        return f"¥{float(x):,.2f}"
    except:  # 防御
        return str(x)

if run_btn:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=120).decode("utf-8")
        data = json.loads(out)
    except subprocess.CalledProcessError as e:
        st.error("脚本运行失败")
        st.code(e.output.decode("utf-8"), language="bash")
    except json.JSONDecodeError:
        st.error("脚本返回的不是有效 JSON，已显示原文：")
        st.code(out, language="json")
    else:
        # 读取关键信息
        alloc = data.get("allocation_today", {})
        sig = data.get("signal", {})
        ratios = data.get("ratios", {})
        fgi = data.get("fgi", {})
        ticker = data.get("ticker", ticker)
        mode_used = data.get("mode", mode)

        core_invest = alloc.get("core_invest", 0.0)
        tactical_invest = alloc.get("tactical_invest", 0.0)
        total_invest = alloc.get("total_invest", 0.0)
        remaining_cash = alloc.get("remaining_cash", 0.0)
        total_frac = alloc.get("total_fraction_of_capital", 0.0)

        tier = sig.get("tier_after_mode") or sig.get("tier")
        tier_frac = sig.get("tier_fraction_of_tactical_after_mode", sig.get("tier_fraction_of_tactical", 0.0))

        # ---------- 醒目的仓位卡片 ----------
        st.subheader("📌 今日建议（简洁视图）")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("底仓投入 (Core)", format_money(core_invest))
        c2.metric("战术投入 (Tactical)", format_money(tactical_invest))
        c3.metric("今日合计", format_money(total_invest))
        c4.metric("剩余现金", format_money(remaining_cash))

        # 进度条：今日投入占现金比例（快速体感）
        st.progress(min(max(total_frac, 0.0), 1.0), text=f"今日投入占现金比：{total_frac*100:.1f}%")

        # 档位徽章 + 关键信号摘要
        badge_color = "#10b981" if str(tier).startswith("BUY") or str(tier)=="ALL_IN" else "#6b7280"
        st.markdown(
            f"""
            <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
              <span style="background:{badge_color};color:white;padding:6px 10px;border-radius:999px;font-weight:600;">
                档位：{tier}（战术比例 {tier_frac*100:.0f}%）
              </span>
              <span style="background:#111827;color:#e5e7eb;padding:6px 10px;border-radius:8px;">
                标的：{ticker} · 模式：{mode_used}
              </span>
              <span style="background:#f59e0b;color:#111827;padding:6px 10px;border-radius:8px;">
                FGI：{fgi.get('value', '—'):.2f}（{fgi.get('label','—')}）
              </span>
              <span style="background:#374151;color:#e5e7eb;padding:6px 10px;border-radius:8px;">
                MA20 偏离 dev20：{sig.get('breakdown',{}).get('dev20','—')}
              </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # ---------- 可选：简要表格 ----------
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            st.caption("底/战术目标比")
            st.table({
                "项目":["Core","Tactical"],
                "比例":[ratios.get("core_ratio", "—"), ratios.get("tactical_ratio", "—")]
            })
        with bc2:
            st.caption("价格与均线")
            st.table({
                "指标":["Price","MA20","MA50","MA200"],
                "数值":[data.get("price","—"), data.get("ma20","—"), data.get("ma50","—"), data.get("ma200","—")]
            })
        with bc3:
            st.caption("风控要点（节选）")
            rs = sig.get("risk_switches", {})
            st.table({
                "项":["VIX分位","UUP分位","dev200","vol20","downfrac"],
                "值":[rs.get("vix_pct","—"), rs.get("uup_pct","—"), rs.get("dev200","—"), rs.get("vol20","—"), rs.get("downfrac","—")]
            })

        # ---------- 折叠：原始 JSON 与调试信息 ----------
        with st.expander("🧰 详细/原始JSON（可展开）", expanded=False):
            st.json(data)
