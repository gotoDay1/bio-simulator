import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from ui_components import render_param_inputs

from biosim import (
    GROWTH_MODELS,
    OXYGEN_MODELS,
    PRODUCT_MODELS,
    SUBSTRATE_MODELS,
    Batch,
    BioreactorSimulation,
    Chemostat,
    FedBatch,
    InitialConditions,
    IntegrationError,
    InvalidParameterError,
    constant_feed,
    exponential_feed,
    step_feed,
)

st.set_page_config(page_title="Bioreactor Simulator", layout="wide")
st.title("バイオリアクターシミュレーター")

with st.sidebar:
    st.header("運転モード")
    mode_choice = st.selectbox("モード", ["batch", "fed_batch", "chemostat"], index=0)

    mode_kwargs: dict = {}
    if mode_choice == "fed_batch":
        st.subheader("流加培養パラメータ")
        profile_choice = st.selectbox(
            "フィードプロファイル", ["constant", "step", "exponential"]
        )
        if profile_choice == "constant":
            rate = st.number_input("フィード流量 F (L/h)", value=0.05, min_value=0.0)
            feed_rate_fn = constant_feed(rate)
        elif profile_choice == "step":
            t_start = st.number_input("開始時刻 t_start (h)", value=2.0, min_value=0.0)
            rate = st.number_input("フィード流量 F (L/h)", value=0.05, min_value=0.0)
            feed_rate_fn = step_feed(t_start=t_start, rate=rate)
        else:
            F0 = st.number_input("初期フィード流量 F0 (L/h)", value=0.02, min_value=0.0)
            mu_set = st.number_input("目標比増殖速度 mu_set (1/h)", value=0.1)
            feed_rate_fn = exponential_feed(F0=F0, mu_set=mu_set)
        S_feed = st.number_input("フィード基質濃度 S_feed (g/L)", value=100.0, min_value=0.0)
        mode_kwargs = {"feed_rate_fn": feed_rate_fn, "S_feed": S_feed}
    elif mode_choice == "chemostat":
        st.subheader("連続培養パラメータ")
        D = st.number_input("希釈率 D (1/h)", value=0.1, min_value=0.0001)
        S_feed = st.number_input("フィード基質濃度 S_feed (g/L)", value=20.0, min_value=0.0)
        mode_kwargs = {"D": D, "S_feed": S_feed}

    st.header("モデル選択")
    growth_choice = st.selectbox("増殖モデル", list(GROWTH_MODELS.keys()))
    st.caption("パラメータ")
    growth_kwargs = render_param_inputs(GROWTH_MODELS[growth_choice], "growth")

    product_choice = st.selectbox("生産モデル", list(PRODUCT_MODELS.keys()))
    st.caption("パラメータ")
    product_kwargs = render_param_inputs(PRODUCT_MODELS[product_choice], "product")

    substrate_choice = st.selectbox("基質消費モデル", list(SUBSTRATE_MODELS.keys()))
    st.caption("パラメータ")
    substrate_kwargs = render_param_inputs(SUBSTRATE_MODELS[substrate_choice], "substrate")

    oxygen_choice = st.selectbox("酸素消費モデル", list(OXYGEN_MODELS.keys()))
    st.caption("パラメータ")
    oxygen_kwargs = render_param_inputs(OXYGEN_MODELS[oxygen_choice], "oxygen")
    oxygen_cls = OXYGEN_MODELS[oxygen_choice]

    st.header("初期条件")
    X0 = st.number_input("X0 (g/L)", value=0.1, min_value=0.0)
    S0 = st.number_input("S0 (g/L)", value=20.0, min_value=0.0)
    P0 = st.number_input("P0 (g/L)", value=0.0, min_value=0.0)
    V0 = st.number_input("V0 (L)", value=1.0, min_value=0.0001)
    C_O2_0 = None
    if oxygen_cls.supports_supply_dynamics:
        C_O2_0 = st.number_input("C_O2_0 (mg/L)", value=5.0, min_value=0.0)

    st.header("シミュレーション時間")
    t_end = st.number_input("t_end (h)", value=24.0, min_value=0.1)
    n_points = st.number_input("出力点数", value=200, min_value=10, max_value=5000, step=10)

    run_clicked = st.button("シミュレーション実行", type="primary")

if run_clicked:
    try:
        growth_model = GROWTH_MODELS[growth_choice](**growth_kwargs)
        product_model = PRODUCT_MODELS[product_choice](**product_kwargs)
        substrate_model = SUBSTRATE_MODELS[substrate_choice](**substrate_kwargs)
        oxygen_model = OXYGEN_MODELS[oxygen_choice](**oxygen_kwargs)

        if mode_choice == "batch":
            operation_mode = Batch()
        elif mode_choice == "fed_batch":
            operation_mode = FedBatch(**mode_kwargs)
        else:
            operation_mode = Chemostat(**mode_kwargs)

        initial_conditions = InitialConditions(X0=X0, S0=S0, P0=P0, V0=V0, C_O2_0=C_O2_0)

        sim = BioreactorSimulation(
            growth_model=growth_model,
            product_model=product_model,
            substrate_model=substrate_model,
            oxygen_model=oxygen_model,
            operation_mode=operation_mode,
            initial_conditions=initial_conditions,
            t_span=(0.0, t_end),
            n_points=int(n_points),
        )
        st.session_state["results"] = sim.run()
        st.session_state["error"] = None
    except (InvalidParameterError, IntegrationError) as e:
        st.session_state["results"] = None
        st.session_state["error"] = str(e)

if st.session_state.get("error"):
    st.error(st.session_state["error"])

results = st.session_state.get("results")
if results is not None:
    st.plotly_chart(results.to_plotly_figure(), width="stretch")

    csv_bytes = results.data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "CSVダウンロード",
        data=csv_bytes,
        file_name="bioreactor_simulation.csv",
        mime="text/csv",
    )
else:
    st.info("左のサイドバーでパラメータを設定し、「シミュレーション実行」を押してください。")
