import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ui_components import render_feed_profile_inputs, render_param_inputs

from biosim import (
    GROWTH_MODELS,
    OXYGEN_MODELS,
    PRODUCT_MODELS,
    SUBSTRATE_MODELS,
    Batch,
    BioreactorSimulation,
    Chemostat,
    ExperimentalDataError,
    FedBatch,
    InitialConditions,
    IntegrationError,
    InvalidParameterError,
    load_experimental_csv,
)

st.set_page_config(page_title="Bioreactor Simulator", layout="wide")
st.title("バイオリアクターシミュレーター")

with st.sidebar:
    st.header("運転モード")
    mode_choice = st.selectbox("モード", ["batch", "fed_batch", "chemostat"], index=0)

    mode_kwargs: dict = {}
    if mode_choice == "fed_batch":
        mode_kwargs = render_feed_profile_inputs("feed")
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

    st.header("実測データとの比較（任意）")
    experimental_data = None
    experimental_file = st.file_uploader(
        "実測データCSV（固定列名: t + X/OD/S/P/OTR のうち存在する列）",
        type="csv",
        key="experimental_csv",
    )
    if experimental_file is not None:
        od_conversion_factor = st.number_input(
            "OD→X 変換係数（OD列がある場合のみ使用, X = 係数 * OD）",
            value=1.0,
            key="od_conversion_factor",
        )
        try:
            experimental_data = load_experimental_csv(
                experimental_file, od_conversion_factor=od_conversion_factor
            )
        except ExperimentalDataError as e:
            st.error(str(e))

if run_clicked:
    try:
        growth_model = GROWTH_MODELS[growth_choice](**growth_kwargs)
        product_model = PRODUCT_MODELS[product_choice](**product_kwargs)
        substrate_model = SUBSTRATE_MODELS[substrate_choice](**substrate_kwargs)
        oxygen_model = OXYGEN_MODELS[oxygen_choice](**oxygen_kwargs)

        if mode_choice == "batch":
            operation_mode = Batch()
        elif mode_choice == "fed_batch":
            if mode_kwargs.get("feed_rate_fn") is None:
                raise InvalidParameterError(
                    "フィードプロファイルCSVをアップロードするか、有効なプロファイルを選択してください。"
                )
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
    fig = results.to_plotly_figure(experimental_data=experimental_data)
    st.plotly_chart(fig, width="stretch")

    csv_bytes = results.data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "CSVダウンロード",
        data=csv_bytes,
        file_name="bioreactor_simulation.csv",
        mime="text/csv",
    )
else:
    st.info("左のサイドバーでパラメータを設定し、「シミュレーション実行」を押してください。")
