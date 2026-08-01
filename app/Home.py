import streamlit as st

st.set_page_config(page_title="biosim", layout="wide")
st.title("biosim — バイオリアクター物質収支シミュレーター")
st.markdown(
    "菌体増殖・代謝物生産・基質消費・溶存酸素消費の4つのサブモデルを組み合わせ、"
    "常微分方程式（ODE）による物質収支をバッチ／流加培養／連続培養の3つの運転モードで解く"
    "シミュレーターです。"
)
st.info("左のサイドバーから **Simulation**（シミュレーション）または **Fitting**（実験データフィッティング）のページを選択してください。")
