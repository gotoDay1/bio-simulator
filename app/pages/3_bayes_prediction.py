import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui_components import render_feed_profile_inputs, render_uncovered_param_inputs

from biosim import (
    GROWTH_MODELS,
    OXYGEN_MODELS,
    PRODUCT_MODELS,
    SUBSTRATE_MODELS,
    Batch,
    Chemostat,
    ExperimentDesignError,
    FedBatch,
    InitialConditions,
    IntegrationError,
    InvalidParameterError,
    PredictionError,
    fit_parameter_models,
    fit_results_to_dataframe,
    predict_parameters,
    predictions_to_dataframe,
    resolve_model_classes,
    suggest_next_experiment,
    validate_field_coverage,
)

st.set_page_config(page_title="パラメータ予測とベイズ最適化", layout="wide")
st.title("パラメータ予測とベイズ最適化")
st.caption(
    "過去の実験（フィッティング結果＋温度・pHなどの条件）から、条件→パラメータのガウス過程回帰モデルを学習し、"
    "新しい条件でのパラメータを予測します。さらに、その予測パラメータで実際にシミュレーションした際の"
    "生産物の絶対量（濃度ではなく質量, P×V）を目的関数として、次に試すべき条件をベイズ最適化"
    "（Expected Improvement）で1点提案します。パレートフロントを探す多目的最適化ではなく、単一目的の最適化です。"
)

RESERVED_COLUMNS = {"batch", "cost", "success", "message"}
_CATEGORIES = ("growth", "product", "substrate", "oxygen")
_CATEGORY_LABELS = {
    "growth": "増殖モデル",
    "product": "生産モデル",
    "substrate": "基質消費モデル",
    "oxygen": "酸素消費モデル",
}
_MODEL_LABELS = {
    "growth": GROWTH_MODELS,
    "product": PRODUCT_MODELS,
    "substrate": SUBSTRATE_MODELS,
    "oxygen": OXYGEN_MODELS,
}


def _selectable_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in RESERVED_COLUMNS and not c.endswith("_model")]


# ---------------------------------------------------------------------------
# Section 1: training data + condition -> parameter prediction
# ---------------------------------------------------------------------------
st.header("1. 学習データとパラメータ予測")

col_upload, col_from_fit = st.columns(2)
with col_upload:
    uploaded = st.file_uploader(
        "学習用ワイドCSV（fitting_results.csv形式 ＋ 任意の条件列。例: temperature, pH）",
        type="csv",
        key="bayes_csv_upload",
    )
    if uploaded is not None:
        st.session_state["bayes_training_data"] = pd.read_csv(uploaded)
with col_from_fit:
    if st.session_state.get("fit_results"):
        if st.button("Fittingページの結果を読み込む"):
            st.session_state["bayes_training_data"] = fit_results_to_dataframe(
                st.session_state["fit_results"]
            )
    else:
        st.caption("Fittingページでフィッティングを実行済みの場合、その結果をここで読み込めます。")

training_data = st.session_state.get("bayes_training_data")

if training_data is not None:
    st.caption("学習データ（温度・pHなどの条件列をここで追加・編集できます）")
    training_data = st.data_editor(training_data, num_rows="dynamic", key="bayes_training_editor")
    st.session_state["bayes_training_data"] = training_data

    options = _selectable_columns(training_data)
    condition_columns = st.multiselect(
        "条件列（例: temperature, pH）", options, key="bayes_condition_cols"
    )
    parameter_options = [c for c in options if c not in condition_columns]
    parameter_columns = st.multiselect(
        "予測対象パラメータ列",
        parameter_options,
        default=parameter_options,
        key="bayes_parameter_cols",
    )

    with st.expander("詳細設定（学習）"):
        min_rows = st.number_input(
            "最小データ数 (min_rows)", value=3, min_value=2, key="bayes_min_rows"
        )
        min_correlation_rows = st.number_input(
            "相関推定に必要な最小共通データ数", value=3, min_value=2, key="bayes_min_corr_rows"
        )
        ci_choice = st.selectbox("信頼区間", ["90%", "95%", "99%"], index=1, key="bayes_ci_choice")
    z_score = {"90%": 1.645, "95%": 1.96, "99%": 2.576}[ci_choice]

    if st.button("モデルを学習", type="primary"):
        try:
            model = fit_parameter_models(
                training_data,
                condition_columns,
                parameter_columns,
                min_rows=int(min_rows),
                min_correlation_rows=int(min_correlation_rows),
            )
            st.session_state["bayes_prediction_model"] = model
            st.session_state.pop("bayes_suggestion", None)
            st.session_state.pop("bayes_predictions", None)
        except PredictionError as e:
            st.error(str(e))

prediction_model = st.session_state.get("bayes_prediction_model")

if prediction_model is not None:
    if prediction_model.skipped_parameters:
        st.warning(
            "学習データ不足のためスキップされたパラメータ:\n"
            + "\n".join(f"- {k}: {v}" for k, v in prediction_model.skipped_parameters.items())
        )
    if prediction_model.correlation_fallback_pairs:
        st.warning(
            "相関を推定するデータが不足し、無相関として扱ったパラメータペア:\n"
            + "\n".join(
                f"- {a} / {b}: {reason}"
                for (a, b), reason in prediction_model.correlation_fallback_pairs.items()
            )
        )

    corr = prediction_model.residual_correlation
    if len(corr) >= 2:
        st.subheader("パラメータ間の残差相関")
        corr_values = corr.to_numpy()
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_values,
                x=list(corr.columns),
                y=list(corr.index),
                zmin=-1,
                zmax=1,
                colorscale=[
                    [0.0, "#8a241c"],
                    [0.25, "#e34948"],
                    [0.5, "#f0efec"],
                    [0.75, "#6da7ec"],
                    [1.0, "#0d366b"],
                ],
                colorbar={"title": "相関係数"},
                text=[[f"{v:.2f}" for v in row] for row in corr_values],
                texttemplate="%{text}",
                hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
            )
        )
        fig_corr.update_layout(height=max(300, 60 * len(corr)))
        st.plotly_chart(fig_corr, width="stretch")

    st.subheader("新しい条件での予測")
    query_conditions: dict[str, float] = {}
    input_cols = st.columns(len(prediction_model.condition_names))
    for col, name in zip(input_cols, prediction_model.condition_names, strict=True):
        default = float(training_data[name].mean())
        query_conditions[name] = col.number_input(name, value=default, key=f"bayes_query_{name}")

    if st.button("予測実行"):
        st.session_state["bayes_predictions"] = predict_parameters(
            prediction_model, query_conditions, z_score=z_score
        )

    predictions = st.session_state.get("bayes_predictions")
    if predictions:
        pred_df = predictions_to_dataframe(predictions)
        st.dataframe(pred_df, width="stretch")
        st.download_button(
            "CSVダウンロード",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="parameter_predictions.csv",
            mime="text/csv",
        )

        if len(prediction_model.condition_names) in (1, 2):
            plot_param = st.selectbox(
                "表示するパラメータ", list(prediction_model.parameter_models), key="bayes_plot_param"
            )

        if len(prediction_model.condition_names) == 1:
            condition_name = prediction_model.condition_names[0]
            obs = training_data[condition_name].astype(float)
            lo, hi = float(obs.min()), float(obs.max())
            margin = 0.2 * max(hi - lo, 1e-6)
            grid = np.linspace(lo - margin, hi + margin, 100)
            grid_preds = [
                predict_parameters(prediction_model, {condition_name: float(g)}, z_score=z_score)
                for g in grid
            ]
            means = [next(p.mean for p in gp if p.parameter_name == plot_param) for gp in grid_preds]
            lowers = [next(p.lower for p in gp if p.parameter_name == plot_param) for gp in grid_preds]
            uppers = [next(p.upper for p in gp if p.parameter_name == plot_param) for gp in grid_preds]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([grid, grid[::-1]]),
                    y=np.concatenate([uppers, lowers[::-1]]),
                    fill="toself",
                    fillcolor="rgba(37,106,191,0.18)",
                    line={"width": 0},
                    name=f"{ci_choice}信頼区間",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=grid, y=means, mode="lines", name="GP平均",
                    line={"color": "#256abf", "width": 2},
                )
            )
            if plot_param in training_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=obs, y=training_data[plot_param].astype(float),
                        mode="markers", name="学習データ",
                        marker={"symbol": "circle-open", "size": 8, "color": "#0b0b0b"},
                    )
                )
            fig.update_layout(
                xaxis_title=condition_name, yaxis_title=plot_param, height=400,
                title=f"{plot_param} vs {condition_name}",
            )
            st.plotly_chart(fig, width="stretch")
        elif len(prediction_model.condition_names) == 2:
            name_x, name_y = prediction_model.condition_names
            grid_res = 30

            def _grid_1d(name: str) -> np.ndarray:
                obs = training_data[name].astype(float)
                lo, hi = float(obs.min()), float(obs.max())
                margin = 0.2 * max(hi - lo, 1e-6)
                return np.linspace(lo - margin, hi + margin, grid_res)

            x_vals = _grid_1d(name_x)
            y_vals = _grid_1d(name_y)
            z = np.empty((grid_res, grid_res))
            for i, xv in enumerate(x_vals):
                for j, yv in enumerate(y_vals):
                    preds = predict_parameters(
                        prediction_model, {name_x: float(xv), name_y: float(yv)}, z_score=z_score
                    )
                    z[i, j] = next(p.mean for p in preds if p.parameter_name == plot_param)

            fig = go.Figure()
            fig.add_trace(
                go.Contour(
                    x=x_vals, y=y_vals, z=z.T,
                    colorscale=[
                        [0.0, "#cde2fb"], [0.25, "#6da7ec"], [0.5, "#256abf"],
                        [0.75, "#184f95"], [1.0, "#0d366b"],
                    ],
                    colorbar={"title": f"{plot_param}<br>予測平均"},
                    name="GP平均",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=training_data[name_x].astype(float),
                    y=training_data[name_y].astype(float),
                    mode="markers", name="学習データ",
                    marker={"symbol": "circle-open", "size": 10, "color": "#0b0b0b"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[query_conditions[name_x]], y=[query_conditions[name_y]],
                    mode="markers", name="問い合わせ条件",
                    marker={"symbol": "star", "size": 16, "color": "#eb6834"},
                )
            )
            fig.update_layout(
                xaxis_title=name_x, yaxis_title=name_y, height=550,
                title=f"{plot_param} vs ({name_x}, {name_y})",
            )
            st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Section 2: next-experiment suggestion (simulation-based Bayesian optimization)
# ---------------------------------------------------------------------------
if prediction_model is not None:
    st.header("2. 次の実験点の提案")

    try:
        model_classes = resolve_model_classes(training_data)
    except ExperimentDesignError as e:
        st.error(f"学習データからモデルを特定できません: {e}")
        model_classes = None

    if model_classes is not None:
        predicted_columns = set(prediction_model.parameter_models)

        st.subheader("固定パラメータ（予測対象でないモデルフィールド）")
        fixed_values: dict[str, dict[str, float | None]] = {}
        for category in _CATEGORIES:
            predicted_fields = {
                col[len(category) + 1 :]
                for col in predicted_columns
                if col.startswith(f"{category}_")
            }
            cls = model_classes[category]
            remaining = [f.name for f in dataclasses.fields(cls) if f.name != "feed_rate_fn"]
            if all(f in predicted_fields for f in remaining):
                continue
            st.caption(f"{_CATEGORY_LABELS[category]}（{cls.name}）")
            fixed_values[category] = render_uncovered_param_inputs(
                cls, predicted_fields, key_prefix=f"bayes_fixed_{category}"
            )

        st.subheader("運転モードと初期条件")
        mode_choice = st.selectbox(
            "モード", ["batch", "fed_batch", "chemostat"], index=0, key="bayes_mode_choice"
        )
        mode_kwargs: dict = {}
        if mode_choice == "fed_batch":
            mode_kwargs = render_feed_profile_inputs("bayes_feed")
        elif mode_choice == "chemostat":
            D = st.number_input("希釈率 D (1/h)", value=0.1, min_value=0.0001, key="bayes_D")
            S_feed = st.number_input(
                "フィード基質濃度 S_feed (g/L)", value=20.0, min_value=0.0, key="bayes_S_feed"
            )
            mode_kwargs = {"D": D, "S_feed": S_feed}

        col1, col2, col3, col4 = st.columns(4)
        X0 = col1.number_input("X0 (g/L)", value=0.1, min_value=0.0, key="bayes_X0")
        S0 = col2.number_input("S0 (g/L)", value=20.0, min_value=0.0, key="bayes_S0")
        P0 = col3.number_input("P0 (g/L)", value=0.0, min_value=0.0, key="bayes_P0")
        V0 = col4.number_input("V0 (L)", value=1.0, min_value=0.0001, key="bayes_V0")
        C_O2_0 = None
        if model_classes["oxygen"].supports_supply_dynamics:
            C_O2_0 = st.number_input("C_O2_0 (mg/L)", value=5.0, min_value=0.0, key="bayes_C_O2_0")
        t_end = st.number_input("評価する培養時間 t_end (h)", value=20.0, min_value=0.001, key="bayes_t_end")

        st.subheader("探索範囲")
        bounds: dict[str, tuple[float, float]] = {}
        bound_cols = st.columns(len(prediction_model.condition_names))
        for col, name in zip(bound_cols, prediction_model.condition_names, strict=True):
            obs = training_data[name].astype(float)
            with col:
                lo = st.number_input(
                    f"{name} 下限", value=float(obs.min()), key=f"bayes_bound_lo_{name}"
                )
                hi = st.number_input(
                    f"{name} 上限", value=float(obs.max()), key=f"bayes_bound_hi_{name}"
                )
            bounds[name] = (lo, hi)

        n_dims = len(prediction_model.condition_names)
        with st.expander("詳細設定（次の実験点の提案）"):
            n_mc_samples = st.number_input(
                "モンテカルロサンプル数", value=15, min_value=1, key="bayes_n_mc_samples"
            )
            if n_dims <= 2:
                grid_resolution = st.number_input(
                    "格子解像度（1次元あたり）", value=10, min_value=2, key="bayes_grid_resolution"
                )
                n_candidates = 150
            else:
                n_candidates = st.number_input(
                    "候補点数（ラテン超方格サンプリング）", value=150, min_value=1, key="bayes_n_candidates"
                )
                grid_resolution = 10
            xi = st.number_input(
                "探索パラメータ xi（f_bestに対する相対値）", value=0.01, min_value=0.0, key="bayes_xi"
            )
            min_valid_draws = st.number_input(
                "候補を有効とみなす最小サンプル成功数", value=5, min_value=1, key="bayes_min_valid_draws"
            )
            total_sims = (
                (grid_resolution**n_dims if n_dims <= 2 else n_candidates) * n_mc_samples
            )
            st.caption(
                f"1回の実行あたり約 {total_sims:,} 回のシミュレーションを行います"
                "（このアプリのモデルでは1回あたり概ね5〜12ms）。"
            )

        if st.button("次の実験点を提案", type="primary"):
            try:
                validate_field_coverage(model_classes, predicted_columns, fixed_values)
            except ExperimentDesignError as e:
                st.error(str(e))
            else:
                try:
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

                    with st.spinner("次の実験点を探索中..."):
                        suggestion = suggest_next_experiment(
                            prediction_model,
                            training_data,
                            fixed_values,
                            initial_conditions,
                            operation_mode,
                            float(t_end),
                            bounds,
                            n_mc_samples=int(n_mc_samples),
                            grid_resolution=int(grid_resolution),
                            n_candidates=int(n_candidates),
                            xi=float(xi),
                            min_valid_draws=int(min_valid_draws),
                        )
                    st.session_state["bayes_suggestion"] = suggestion
                except (
                    ExperimentDesignError,
                    InvalidParameterError,
                    IntegrationError,
                    PredictionError,
                ) as e:
                    st.error(str(e))

        suggestion = st.session_state.get("bayes_suggestion")
        if suggestion is not None:
            st.subheader("提案条件")
            best_row = {
                **suggestion.best_condition,
                "予測平均（生産物絶対量, g）": suggestion.best_mean,
                "予測std": suggestion.best_std,
                "EI": suggestion.best_ei,
                "これまでの最良実績 (f_best)": suggestion.f_best,
                "f_bestを達成したバッチ": suggestion.f_best_batch,
            }
            st.dataframe(pd.DataFrame([best_row]), width="stretch")

            st.subheader("過去バッチの実測パラメータによる生産物絶対量")
            hist_df = pd.DataFrame(
                [
                    {"batch": h.batch_name, "生産物絶対量 (g)": h.objective}
                    for h in suggestion.historical_objectives
                ]
            )
            st.dataframe(hist_df, width="stretch")
            if suggestion.skipped_historical_batches:
                st.warning(
                    "シミュレーションできなかったバッチ:\n"
                    + "\n".join(
                        f"- {k}: {v}" for k, v in suggestion.skipped_historical_batches.items()
                    )
                )

            if n_dims in (1, 2):
                condition_to_objective = {
                    h.batch_name: h.objective for h in suggestion.historical_objectives
                }
                hist_points = training_data[training_data["batch"].isin(condition_to_objective)]

            if n_dims == 1:
                name = prediction_model.condition_names[0]
                sorted_candidates = suggestion.evaluated_candidates.sort_values(name)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=sorted_candidates[name], y=sorted_candidates["mean"],
                        mode="lines", name="予測平均（生産物絶対量）",
                        line={"color": "#256abf", "width": 2},
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=hist_points[name],
                        y=[condition_to_objective[b] for b in hist_points["batch"]],
                        mode="markers", name="過去バッチ",
                        marker={"symbol": "circle-open", "size": 8, "color": "#0b0b0b"},
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[suggestion.best_condition[name]], y=[suggestion.best_mean],
                        mode="markers", name="提案条件",
                        marker={"symbol": "star", "size": 14, "color": "#eb6834"},
                    )
                )
                fig.update_layout(
                    xaxis_title=name, yaxis_title="生産物絶対量 (g)", height=450
                )
                st.plotly_chart(fig, width="stretch")
            elif n_dims == 2:
                name_x, name_y = prediction_model.condition_names
                grid_res = int(grid_resolution)
                cand = suggestion.evaluated_candidates
                z = cand["mean"].to_numpy().reshape(grid_res, grid_res)
                x_vals = cand[name_x].to_numpy().reshape(grid_res, grid_res)[:, 0]
                y_vals = cand[name_y].to_numpy().reshape(grid_res, grid_res)[0, :]

                fig = go.Figure()
                fig.add_trace(
                    go.Contour(
                        x=x_vals, y=y_vals, z=z.T,
                        colorscale=[
                            [0.0, "#cde2fb"], [0.25, "#6da7ec"], [0.5, "#256abf"],
                            [0.75, "#184f95"], [1.0, "#0d366b"],
                        ],
                        colorbar={"title": "予測平均<br>(g)"},
                        name="予測平均",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=hist_points[name_x], y=hist_points[name_y],
                        mode="markers", name="過去バッチ",
                        marker={"symbol": "circle-open", "size": 10, "color": "#0b0b0b"},
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[suggestion.best_condition[name_x]],
                        y=[suggestion.best_condition[name_y]],
                        mode="markers", name="提案条件",
                        marker={"symbol": "star", "size": 16, "color": "#eb6834"},
                    )
                )
                fig.update_layout(xaxis_title=name_x, yaxis_title=name_y, height=550)
                st.plotly_chart(fig, width="stretch")
elif training_data is None:
    st.info(
        "上のセクションで学習用CSVをアップロードするか、Fittingページの結果を読み込んでください。"
    )
