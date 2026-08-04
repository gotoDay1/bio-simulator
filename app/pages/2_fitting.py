import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ui_components import render_feed_profile_inputs, render_fit_param_inputs

from biosim import (
    GROWTH_MODELS,
    OXYGEN_MODELS,
    PRODUCT_MODELS,
    SUBSTRATE_MODELS,
    Batch,
    Chemostat,
    ExperimentalDataError,
    FedBatch,
    FittingError,
    InitialConditions,
    IntegrationError,
    InvalidParameterError,
    ModelSpec,
    fit_batch,
    fit_results_to_dataframe,
    load_experimental_csv,
)

st.set_page_config(page_title="実験データフィッティング", layout="wide")
st.title("実験データフィッティング")
st.caption(
    "実測データ（複数バッチ可）に対し、増殖・生産・基質・酸素の各モデルパラメータを、"
    "シミュレーション⇔実測の残差最小化によって推定します。"
    "ODEをシミュレートして実測と比較する汎用的な方式のため、Gompertzのような"
    "力学的ODE形のモデルにも対応しています。"
)

with st.sidebar:
    st.header("運転モード")
    mode_choice = st.selectbox(
        "モード", ["batch", "fed_batch", "chemostat"], index=0, key="fit_mode_choice"
    )

    mode_kwargs: dict = {}
    if mode_choice == "fed_batch":
        mode_kwargs = render_feed_profile_inputs("fit_feed")
    elif mode_choice == "chemostat":
        st.subheader("連続培養パラメータ")
        D = st.number_input(
            "希釈率 D (1/h)", value=0.1, min_value=0.0001, key="fit_chemostat_D"
        )
        S_feed = st.number_input(
            "フィード基質濃度 S_feed (g/L)",
            value=20.0,
            min_value=0.0,
            key="fit_chemostat_S_feed",
        )
        mode_kwargs = {"D": D, "S_feed": S_feed}

    st.header("モデル選択とパラメータ設定")
    st.caption("各パラメータは「固定」または「自由(推定)」を選べます。")

    st.subheader("増殖モデル")
    growth_choice = st.selectbox(
        "増殖モデル", list(GROWTH_MODELS.keys()), key="fit_growth_choice"
    )
    growth_params = render_fit_param_inputs(GROWTH_MODELS[growth_choice], "fit_growth")

    st.subheader("生産モデル")
    product_choice = st.selectbox(
        "生産モデル", list(PRODUCT_MODELS.keys()), key="fit_product_choice"
    )
    product_params = render_fit_param_inputs(PRODUCT_MODELS[product_choice], "fit_product")

    st.subheader("基質消費モデル")
    substrate_choice = st.selectbox(
        "基質消費モデル", list(SUBSTRATE_MODELS.keys()), key="fit_substrate_choice"
    )
    substrate_params = render_fit_param_inputs(
        SUBSTRATE_MODELS[substrate_choice], "fit_substrate"
    )

    st.subheader("酸素消費モデル")
    oxygen_choice = st.selectbox(
        "酸素消費モデル", list(OXYGEN_MODELS.keys()), key="fit_oxygen_choice"
    )
    oxygen_cls = OXYGEN_MODELS[oxygen_choice]
    oxygen_params = render_fit_param_inputs(oxygen_cls, "fit_oxygen")

    st.header("初期条件")
    X0 = st.number_input("X0 (g/L)", value=0.1, min_value=0.0, key="fit_X0")
    S0 = st.number_input("S0 (g/L)", value=20.0, min_value=0.0, key="fit_S0")
    P0 = st.number_input("P0 (g/L)", value=0.0, min_value=0.0, key="fit_P0")
    V0 = st.number_input("V0 (L)", value=1.0, min_value=0.0001, key="fit_V0")
    C_O2_0 = None
    if oxygen_cls.supports_supply_dynamics:
        C_O2_0 = st.number_input("C_O2_0 (mg/L)", value=5.0, min_value=0.0, key="fit_C_O2_0")

    n_points = st.number_input(
        "内部シミュレーション出力点数",
        value=200,
        min_value=10,
        max_value=5000,
        step=10,
        key="fit_n_points",
    )
    n_starts = st.number_input(
        "マルチスタート回数",
        value=8,
        min_value=1,
        max_value=50,
        step=1,
        key="fit_n_starts",
        help=(
            "初期推定値の周辺を複数点サンプリングしてそれぞれ最適化し、"
            "最良の結果を採用します（局所解への収束を避けるため）。"
        ),
    )

    st.header("実験データ（複数バッチ可）")
    od_conversion_factor = st.number_input(
        "OD→X 変換係数（OD列がある場合のみ使用, X = 係数 * OD）",
        value=1.0,
        key="fit_od_conversion_factor",
    )
    uploaded_files = st.file_uploader(
        "実測データCSV（固定列名: t + X/OD/S/P/OTR のうち存在する列）。複数選択可（バッチごとに1ファイル）。",
        type="csv",
        accept_multiple_files=True,
        key="fit_experimental_csvs",
    )

    run_clicked = st.button("フィッティング実行", type="primary")


def _deduplicate_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return result


if run_clicked:
    if not uploaded_files:
        st.session_state["fit_results"] = []
        st.session_state["fit_experimental_by_batch"] = {}
        st.session_state["fit_errors"] = ["実験データCSVを1つ以上アップロードしてください。"]
    else:
        try:
            model_specs = [
                ModelSpec("growth", GROWTH_MODELS[growth_choice], growth_params),
                ModelSpec("product", PRODUCT_MODELS[product_choice], product_params),
                ModelSpec("substrate", SUBSTRATE_MODELS[substrate_choice], substrate_params),
                ModelSpec("oxygen", oxygen_cls, oxygen_params),
            ]
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

            batch_names = _deduplicate_names([Path(f.name).stem for f in uploaded_files])

            fit_results = []
            experimental_by_batch = {}
            errors = []
            with st.spinner("フィッティング実行中..."):
                for batch_name, file in zip(batch_names, uploaded_files, strict=True):
                    try:
                        experimental_data = load_experimental_csv(
                            file, od_conversion_factor=od_conversion_factor
                        )
                        result = fit_batch(
                            batch_name=batch_name,
                            model_specs=model_specs,
                            initial_conditions=initial_conditions,
                            operation_mode=operation_mode,
                            experimental_data=experimental_data,
                            n_points=int(n_points),
                            n_starts=int(n_starts),
                        )
                        fit_results.append(result)
                        experimental_by_batch[batch_name] = experimental_data
                    except (
                        ExperimentalDataError,
                        FittingError,
                        InvalidParameterError,
                        IntegrationError,
                    ) as e:
                        errors.append(f"{batch_name}: {e}")

            st.session_state["fit_results"] = fit_results
            st.session_state["fit_experimental_by_batch"] = experimental_by_batch
            st.session_state["fit_errors"] = errors
        except (InvalidParameterError, FittingError) as e:
            st.session_state["fit_results"] = []
            st.session_state["fit_experimental_by_batch"] = {}
            st.session_state["fit_errors"] = [str(e)]

for error in st.session_state.get("fit_errors", []):
    st.error(error)

fit_results = st.session_state.get("fit_results")
if fit_results:
    result_df = fit_results_to_dataframe(fit_results)
    st.subheader("フィッティング結果")
    st.dataframe(result_df, width="stretch")

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "CSVダウンロード",
        data=csv_bytes,
        file_name="fitting_results.csv",
        mime="text/csv",
    )

    experimental_by_batch = st.session_state.get("fit_experimental_by_batch", {})
    for result in fit_results:
        with st.expander(f"バッチ: {result.batch_name}"):
            if result.simulation_results is not None:
                fig = result.simulation_results.to_plotly_figure(
                    experimental_data=experimental_by_batch.get(result.batch_name)
                )
                st.plotly_chart(fig, width="stretch", key=f"fit_chart_{result.batch_name}")
            else:
                st.warning("ベストフィットパラメータでの最終シミュレーションが失敗しました。")
            st.write(f"収束: {result.success} / コスト(RMSE): {result.cost:.4g}")
            st.caption(result.message)
elif not st.session_state.get("fit_errors"):
    st.info(
        "左のサイドバーでモデル・パラメータを設定し、実験データCSVをアップロードして"
        "「フィッティング実行」を押してください。"
    )
