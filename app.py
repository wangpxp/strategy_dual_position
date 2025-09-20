import streamlit as st
import subprocess, json, sys, os

st.title("Dual Position Allocator (Core + Tactical)")
cash = st.number_input("现金 (元)", value=30000.0, min_value=0.0, step=1000.0)
ticker = st.text_input("Ticker", value="VOO")
mode = st.selectbox("模式", ["build","balanced","conservative","aggressive","trend","contrarian"])
date = st.text_input("日期(YYYY-MM-DD，可空)", value="")

cmd = [sys.executable, "dual_position_allocator.py", "--cash", str(cash), "--ticker", ticker, "--mode", mode]
if date.strip():
    cmd += ["--date", date]

if st.button("计算"):
    out = subprocess.check_output(cmd).decode("utf-8")
    st.code(out, language="json")
    try:
        st.json(json.loads(out))
    except:
        pass
