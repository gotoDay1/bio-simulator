import streamlit as st

st.set_page_config(page_title="biosim", layout="wide")

st.title("biosim — バイオリアクター物質収支シミュレーター")
st.markdown(
    "菌体増殖・代謝物生産・基質消費・溶存酸素消費の **4つのサブモデル** を組み合わせ、"
    "常微分方程式（ODE）による物質収支を **バッチ／流加培養（fed-batch）／連続培養（chemostat）** "
    "の3つの運転モードで解くシミュレーターです。各サブモデルは差し替え可能に設計されており、"
    "このページではモデルの数式・仮定・選び方、および入力ファイルのフォーマットを詳しく説明します。"
)

# ---------------------------------------------------------------------------
# サンプルCSV（ダウンロード用）
# ---------------------------------------------------------------------------

SAMPLE_EXPERIMENTAL_CSV = """t,X,S,P
0,0.10,20.0,0.00
2,0.25,18.9,0.10
4,0.62,16.5,0.55
6,1.40,11.8,1.60
8,2.55,6.20,3.10
10,3.10,2.10,4.05
12,3.30,0.40,4.40
"""

SAMPLE_FEED_PROFILE_CSV = """time,feed_rate
0.0,0.00
2.0,0.02
6.0,0.05
12.0,0.08
"""

SAMPLE_FITTING_RESULTS_CSV = """batch,temperature,pH,growth_model,growth_mu_max,growth_Ks,product_model,product_alpha,product_beta,substrate_model,substrate_Yxs,substrate_ms,substrate_Yps,oxygen_model,oxygen_Yxo2,oxygen_mo2,cost,success,message
batch_T25,25.0,6.0,monod,0.60,0.20,luedeking_piret,2.0,0.05,yield_maintenance,0.50,0.02,,demand_only,0.90,0.05,0.010,True,ok
batch_T30,30.0,6.0,monod,0.75,0.18,luedeking_piret,2.2,0.06,yield_maintenance,0.52,0.02,,demand_only,0.90,0.05,0.012,True,ok
batch_T35,35.0,6.0,monod,0.65,0.25,luedeking_piret,1.8,0.04,yield_maintenance,0.48,0.03,,demand_only,0.90,0.05,0.015,True,ok
"""

st.info(
    "左のサイドバーから **Simulation**（シミュレーション）、**Fitting**（実験データフィッティング）、"
    "**Bayes Prediction**（条件からのパラメータ予測とベイズ最適化）のページを選択してください。"
    "各機能の詳しい使い方・数式・仮定は下のタブにまとまっています。"
)

tab_overview, tab_models, tab_ops, tab_files, tab_numerics = st.tabs(
    [
        "概要 / はじめに",
        "モデルの数式と選び方",
        "運転モードと流加プロファイル",
        "入力ファイルフォーマット",
        "数値解法・既知の制限",
    ]
)

# ---------------------------------------------------------------------------
# タブ1: 概要 / はじめに
# ---------------------------------------------------------------------------
with tab_overview:
    st.header("biosim でできること")
    st.markdown(
        "反応器を完全混合槽（CSTR）とみなし、菌体（X）・基質（S）・生産物（P）・"
        "（必要に応じて）溶存酸素（C_O2）の物質収支を常微分方程式として解きます。"
        "増殖・生産・基質消費・酸素消費の4カテゴリはそれぞれ複数のモデルから選択でき、"
        "運転モード（バッチ／流加／連続）と自由に組み合わせられます。"
    )

    st.subheader("推奨ワークフロー")
    st.markdown(
        """
1. **Simulation**（`1_simulation.py`）— まずはここでモデルの挙動を確認します。
   運転モード・4カテゴリのモデル・初期条件を選んでシミュレーションを実行し、
   任意で実測データCSVをアップロードして重ね描き比較できます。
2. **Fitting**（`2_fitting.py`）— 実測データ（複数バッチ可）に対してモデルパラメータを推定します。
   パラメータごとに「固定値」か「自由（推定）」かを選び、最適化（least squares）で
   フィットさせ、結果をワイド形式CSVとしてダウンロードできます。
3. **Bayes Prediction**（`3_bayes_prediction.py`）— Fitting の結果に温度・pHなどの
   条件列を加えたものを学習データとして、条件からパラメータを予測（ガウス過程回帰）し、
   さらに「次に試すべき実験条件」をベイズ最適化（Expected Improvement）で提案します。
        """
    )

    st.subheader("各ページの早見表")
    st.markdown(
        """
| ページ | 何を選ぶか | 何を入力するか | 何が得られるか |
|---|---|---|---|
| **Simulation** | 運転モード＋4カテゴリのモデル | 各モデルのパラメータ・初期条件・（任意）実測データCSV | シミュレーション結果のグラフ・CSVダウンロード |
| **Fitting** | 運転モード＋4カテゴリのモデル | パラメータごとの固定/推定設定・実測データCSV（複数可） | 推定パラメータのワイド形式CSV・フィット結果グラフ |
| **Bayes Prediction** | 条件列・予測対象パラメータ列 | 学習用ワイドCSV（Fitting結果＋条件列）・探索範囲 | パラメータ予測（平均・信頼区間）・次の実験条件の提案 |
        """
    )

    st.caption(
        "開発環境のセットアップ（`uv venv` 等）やテストの実行方法など、開発者向けの情報は "
        "リポジトリ直下の `README.md` を参照してください。"
    )

# ---------------------------------------------------------------------------
# タブ2: モデルの数式と選び方
# ---------------------------------------------------------------------------
with tab_models:
    st.header("統一物質収支の枠組み")
    st.markdown("反応器を完全混合槽（CSTR）とみなすと、任意の溶質濃度 `C` に対する一般物質収支は次の通りです。")
    st.latex(r"\frac{d(VC)}{dt} = r(C)\,V + F_{in}C_{feed} - F_{out}C")
    st.markdown(
        "積の微分則 `d(VC)/dt = V·dC/dt + C·dV/dt` と体積収支 `dV/dt = F_in − F_out` を代入すると、"
        "`F_out` の項が相殺され、次のシンプルな形に帰着します（本シミュレーターの核となる関係式）。"
    )
    st.latex(r"\frac{dC}{dt} = r(C) + \frac{F_{in}}{V}\left(C_{feed}-C\right), \qquad \frac{dV}{dt}=F_{in}-F_{out}")
    st.markdown(
        "状態ベクトルは `[X, S, P, (C_O2), V]`。`C_O2` は酸素モデルが供給ダイナミクス（kLa）を"
        "サポートする場合のみ状態に追加されます。各カテゴリのモデルは"
        "**増殖 → 生産 → 基質 → 酸素** の順に呼び出されます（生産・基質・酸素の速度式は"
        "増殖モデルが計算した `dX/dt` に依存するため）。"
    )

    st.divider()
    st.header("① 菌体増殖モデル")
    st.caption("`src/biosim/models/growth.py` — 比増殖速度 μ (1/h) を返し、`dX/dt = μ·X` として使われます。")

    with st.expander("Monod式（基質律速）", expanded=True):
        st.latex(r"\mu(S) = \mu_{max}\,\frac{S}{K_s+S}")
        st.markdown(
            """
| パラメータ | 意味 | 単位 | 典型範囲 | デフォルト |
|---|---|---|---|---|
| `mu_max` | 最大比増殖速度 | 1/h | 0.1–1.0 | 0.6 |
| `Ks` | 半飽和定数（μ=μ_max/2となる基質濃度） | g/L | 0.01–1.0 | 0.2 |

古典的な Michaelis-Menten 型の飽和曲線。`S >> Ks` で `μ→μ_max`、`S→0` で `μ→0`。
            """
        )

    with st.expander("Logistic式（環境容量律速）"):
        st.latex(r"\mu(X) = \mu_{max}\left(1-\frac{X}{X_{max}}\right)")
        st.markdown(
            """
| パラメータ | 意味 | 単位 | 典型範囲 | デフォルト |
|---|---|---|---|---|
| `mu_max` | 最大比増殖速度 | 1/h | 0.1–1.0 | 0.6 |
| `Xmax` | 環境収容力（最大菌体濃度） | g/L | 5–50 | 10.0 |

基質濃度 `S` には依存せず、菌体濃度自体が収容力に近づくにつれ増殖速度が線形に減衰します。
`X = Xmax` で `μ=0`、`X > Xmax` では `μ<0`（減衰）となり、これはクリップせずそのまま許容されます。
            """
        )

    with st.expander("Gompertz式（力学的ODE形）"):
        st.latex(r"\mu(X) = \mu_{max}\,\ln\frac{X_{max}}{X}")
        st.markdown(
            """
| パラメータ | 意味 | 単位 | 典型範囲 | デフォルト |
|---|---|---|---|---|
| `mu_max` | 最大比増殖速度 | 1/h | 0.1–1.0 | 0.6 |
| `Xmax` | 漸近的な最大菌体濃度 | g/L | 5–50 | 10.0 |

⚠️ **注意**: これはODE（物質収支）に組み込むための **力学的Gompertz式** であり、実験データの
終点フィッティングに使われる閉形式回帰曲線 `X(t) = Xmax·exp(−exp(...))` とは異なります。
上式を `dX/dt = μ·X` に代入すると `dX/dt = μ_max·X·ln(Xmax/X)` となり、これが非対称な
S字増殖曲線を与える標準的な力学モデルです。境界値の扱い: `X ≤ 0` または `X ≥ Xmax` のとき
`μ=0` にクリップされます。
            """
        )

    st.markdown(
        "**どれを選ぶべきか**: 基質濃度と増殖速度の関係が分かっている・基質枯渇による増殖鈍化を"
        "再現したい → **Monod**。基質を追跡しない、または菌体密度自体が増殖を頭打ちにする現象を"
        "表したい → **Logistic**。緩やかな立ち上がり→急増→緩やかな頭打ち、という非対称なS字カーブを"
        "力学的ODEとして再現したい → **Gompertz**。"
    )

    st.divider()
    st.header("② 代謝物生産モデル")
    st.caption("`src/biosim/models/product.py` — 生産速度 dP/dt (g/L/h) を返します。")

    with st.expander("Luedeking-Piret式（増殖連動＋非連動）", expanded=True):
        st.latex(r"\frac{dP}{dt} = \alpha\,\frac{dX}{dt} + \beta X")
        st.markdown(
            """
| パラメータ | 意味 | 単位 | 典型範囲 | デフォルト |
|---|---|---|---|---|
| `alpha` | 増殖連動生産係数 | g生産物/g菌体 | 0–10 | 2.0 |
| `beta` | 非増殖連動生産係数 | 1/h | 0–0.5 | 0.05 |

`alpha=0` で純粋な非増殖連動型、`beta=0` で純粋な増殖連動型になります。
            """
        )

    with st.expander("NoProduct"):
        st.latex(r"\frac{dP}{dt} = 0")
        st.markdown("生産物を追跡しない場合の恒等モデル。パラメータはありません。")

    st.markdown(
        "**どれを選ぶべきか**: 一次代謝物（増殖と共役して生産される）→ `alpha` を優勢にした "
        "**Luedeking-Piret**。二次代謝物・定常期の生産 → `beta` を優勢にした **Luedeking-Piret**。"
        "生産物を扱わない解析 → **NoProduct**。"
    )

    st.divider()
    st.header("③ 基質消費モデル")
    st.caption("`src/biosim/models/substrate.py` — 現時点で唯一の選択肢です。")

    with st.expander("収率係数＋維持代謝モデル（Pirtの式）", expanded=True):
        st.latex(r"\frac{dS}{dt} = -\frac{1}{Y_{xs}}\frac{dX}{dt} - m_sX \;\left(-\frac{1}{Y_{ps}}\frac{dP}{dt}\right)")
        st.markdown(
            """
| パラメータ | 意味 | 単位 | 典型範囲 | デフォルト |
|---|---|---|---|---|
| `Yxs` | 基質に対する菌体収率 | g菌体/g基質 | 0.1–0.6 | 0.5 |
| `ms` | 維持代謝係数 | g基質/g菌体/h | 0–0.1 | 0.02 |
| `Yps`（任意） | 基質に対する生産物収率。設定した場合のみ生産物消費項を追加 | g生産物/g基質 | — | None（無効） |

古典的な Pirt の式（維持代謝を考慮した収率モデル）です。`Yps` は生産物モデルを併用する場合に
設定します（未設定＝この項は無効）。なお `ms` の項は基質濃度に依存せず菌体濃度にのみ比例するため、
基質が枯渇した後もこの項は消費を続けようとします。これは単純化されたモデルの既知の挙動で、
枯渇後は `S` が数値的に負になり得ます（シミュレーターはこれを検知して警告を出しますが、
クリップは行いません）。
            """
        )

    st.divider()
    st.header("④ 酸素消費モデル")
    st.caption("`src/biosim/models/oxygen.py`")

    with st.expander("需要のみモデル（OxygenDemandOnly）", expanded=True):
        st.latex(r"OUR = -\frac{1}{Y_{xo2}}\frac{dX}{dt} - m_{o2}X")
        st.markdown(
            """
| パラメータ | 意味 | 単位 | 典型範囲 | デフォルト |
|---|---|---|---|---|
| `Yxo2` | 酸素に対する菌体収率 | g菌体/g O2 | 0.5–1.5 | 0.9 |
| `mo2` | 維持代謝酸素係数 | g O2/g菌体/h | 0–0.2 | 0.05 |

酸素が律速にならないと仮定する場合に使用します。溶存酸素濃度 `C_O2` は状態変数として持たず、
累積OUR（酸素摂取速度）のみを診断値として報告します。
            """
        )

    with st.expander("kLa供給付きモデル（OxygenWithKLa）"):
        st.markdown("需要式（OUR）は同じ。これに加えて供給側（OTR: Oxygen Transfer Rate）をモデル化します。")
        st.latex(r"OTR = k_La\left(C_s^{*} - C_{O2}\right)")
        st.latex(r"\frac{dC_{O2}}{dt} = OUR + OTR + \frac{F_{in}}{V}\left(C_{O2,feed}-C_{O2}\right)")
        st.markdown(
            """
| パラメータ | 意味 | 単位 | 典型範囲 | デフォルト |
|---|---|---|---|---|
| `Yxo2` | 酸素に対する菌体収率 | g菌体/g O2 | 0.5–1.5 | 0.9 |
| `mo2` | 維持代謝酸素係数 | g O2/g菌体/h | 0–0.2 | 0.05 |
| `kLa` | 総括容量物質移動係数（通気・撹拌条件を反映） | 1/h | 10–400 | 100.0 |
| `Cs_star` | 飽和溶存酸素濃度 | mg/L | 6–8 | 7.5 |

溶存酸素濃度 `C_O2` を状態変数として追跡します。このモデルを選ぶと初期条件 `C_O2_0` の
設定が必須になります。`C_O2 < Cs*` のとき酸素供給（`OTR>0`）、過飽和のときは負（脱気側）になります。
            """
        )

    st.markdown(
        "**どれを選ぶべきか**: 酸素が律速にならないと仮定してよい・OURだけ知りたい → "
        "**需要のみモデル**（状態変数が少なく計算も軽い）。通気・撹拌条件（kLa）が生産にどう影響するか、"
        "溶存酸素が需要に追いつくかを見たい → **kLa供給付きモデル**。いずれのモデルも増殖速度 `μ` "
        "自体を溶存酸素でゲーティングしない（酸素律速による増殖阻害そのものは表現されない）点に注意してください。"
    )

# ---------------------------------------------------------------------------
# タブ3: 運転モードと流加プロファイル
# ---------------------------------------------------------------------------
with tab_ops:
    st.header("運転モード")
    st.markdown(
        "統一物質収支 `dC/dt = r(C) + (F_in/V)(C_feed − C)` に対して `F_in(t)`, `F_out(t)`, "
        "`C_feed` の与え方を変えるだけで、3つの運転モードすべてを表現できます。"
    )
    st.markdown(
        """
| 運転モード | F_in | F_out | 帰結 |
|---|---|---|---|
| **バッチ**（`Batch`） | 0 | 0 | `dC/dt = r(C)`（反応項のみ）、`dV/dt = 0`（体積一定） |
| **流加培養**（`FedBatch`） | `F(t)`（ユーザー指定プロファイル） | 0 | `dC/dt = r(C) + (F(t)/V)(C_feed−C)`、体積は単調増加 |
| **連続培養**（`Chemostat`） | `D·V` | `D·V` | `dC/dt = r(C) + D(C_feed−C)`（教科書通りのケモスタット式）、`V` は一定 |
        """
    )
    st.warning(
        "連続培養（Chemostat）では希釈率 `D`（1/h、典型 0.05–0.5）が増殖モデルの `μ_max` 未満で"
        "ないと、菌体が反応器から流出し続けて洗い出し（washout）が起こります。"
    )

    st.divider()
    st.header("流加培養（fed-batch）のフィードプロファイル")
    st.markdown(
        "流加培養を選ぶと、フィード流量 `F(t)`（L/h）のプロファイルを4種類から選択できます "
        "（`app/ui_components.py` の「フィードプロファイル」セレクトボックスに対応）。"
    )

    with st.expander("constant（一定）", expanded=True):
        st.markdown("フィード流量 `F` を全期間一定に保ちます。入力: `F`（L/h）。")

    with st.expander("step（単一ステップ）"):
        st.markdown("開始時刻 `t_start` までは0、それ以降は一定値 `F` を流します。入力: `t_start`（h）, `F`（L/h）。")

    with st.expander("exponential（指数関数）"):
        st.latex(r"F(t) = F_0\,e^{\mu_{set}\,t}")
        st.markdown(
            "目標比増殖速度 `mu_set` を一定に保つことを狙った feed-forward プロファイルです。"
            "入力: `F0`（初期フィード流量, L/h）, `mu_set`（目標比増殖速度, 1/h）。"
        )

    with st.expander("csv（多段階ステップホールド）"):
        st.markdown(
            "任意個数のブレークポイントを持つCSVをアップロードし、ブレークポイント間を"
            "階段状（ステップホールド）で保持します。最初のブレークポイントより前はCSV1行目の値、"
            "最後のブレークポイントより後は最終行の値をそのまま保持します。列フォーマットの詳細は"
            "「入力ファイルフォーマット」タブを参照してください。"
        )

    st.caption(
        "いずれのプロファイルでも、フィード基質濃度 `S_feed`（g/L）は別途の数値入力として指定します。"
    )

# ---------------------------------------------------------------------------
# タブ4: 入力ファイルフォーマット
# ---------------------------------------------------------------------------
with tab_files:
    st.header("① 実測データCSV")
    st.caption("Simulation・Fitting ページで使用（`load_experimental_csv`）。列名は固定スキーマでパースされます。")
    st.markdown(
        """
| 列名 | 必須/任意 | 内容 |
|---|---|---|
| `t` | **必須** | 時間 (h) |
| `X` | 任意 | 菌体濃度・DCW (g/L) |
| `OD` | 任意 | 濁度。`X` 列が無い場合のみ `X = od_conversion_factor × OD` として換算 |
| `S` | 任意 | 基質濃度 (g/L) |
| `P` | 任意 | 生産物濃度 (g/L) |
| `OTR` | 任意 | 酸素移動速度。`kla_supply`（kLaモデル）使用時のみ意味を持つ |
        """
    )
    st.markdown(
        "`X`/`S`/`P`/`OTR`/`OD` はすべて任意で、1つのCSVに好きな組み合わせで含められます。"
        "存在しない列は自動的にプロットからスキップされます。"
    )
    st.markdown(
        "**バリデーション**: `t` 列が無い場合は "
        "`Experimental data CSV is missing the required 't' column.` というエラーになります。"
        "`X`/`OD`/`S`/`P`/`OTR` のいずれも存在しない場合も同様にエラーになります。"
    )
    with st.expander("サンプルCSVの中身を見る"):
        st.code(SAMPLE_EXPERIMENTAL_CSV, language="csv")
    st.download_button(
        "実測データCSVのサンプルをダウンロード",
        data=SAMPLE_EXPERIMENTAL_CSV,
        file_name="experimental_data_sample.csv",
        mime="text/csv",
    )

    st.divider()
    st.header("② 流加フィードプロファイルCSV")
    st.caption("運転モードが流加培養（fed-batch）のとき、フィードプロファイルで「csv」を選ぶと使用します（`load_feed_profile_csv`）。")
    st.markdown(
        """
| 列名 | 必須/任意 | 内容 |
|---|---|---|
| `time` | **必須** | 時間 (h) |
| `feed_rate` | **必須** | フィード流量 (L/h) |
        """
    )
    st.markdown(
        "行の順序は問いません（内部で `time` 昇順にソートされます）。ブレークポイント間は"
        "階段状（ステップホールド）で保持されます。"
    )
    st.markdown(
        "**バリデーション**: `time`/`feed_rate` 列が無い場合や有効な行が無い場合はエラーになります。"
        "また、`time` に重複や減少（非単調増加）がある場合、`feed_rate` に負の値がある場合も"
        "それぞれエラーになります。"
    )
    with st.expander("サンプルCSVの中身を見る"):
        st.code(SAMPLE_FEED_PROFILE_CSV, language="csv")
    st.download_button(
        "流加フィードプロファイルCSVのサンプルをダウンロード",
        data=SAMPLE_FEED_PROFILE_CSV,
        file_name="feed_profile_sample.csv",
        mime="text/csv",
    )

    st.divider()
    st.header("③ パラメータ予測用ワイドCSV")
    st.caption("Bayes Prediction ページの学習データとして使用します。Fitting ページの出力形式に条件列を加えたものです。")
    st.markdown(
        """
| 列名 | 必須/任意 | 内容 |
|---|---|---|
| `batch` | 任意（推奨） | バッチ識別名 |
| 任意の数値条件列（例: `temperature`, `pH`） | 任意（1列以上推奨） | 実験条件。予測の入力になります |
| `{category}_model`（例: `growth_model`） | **必須**（4カテゴリ分） | そのカテゴリで使用したモデル名（レジストリのキー、例: `monod`） |
| `{category}_{param}`（例: `growth_mu_max`） | 必須（モデルのパラメータ数だけ） | 固定値またはフィット済みの値 |
| `cost`, `success`, `message` | 任意（予約列） | フィッティング結果の付随情報。予測には使われません |

`category` は `growth` / `product` / `substrate` / `oxygen` の4つです。この形式は "
        """
    )
    st.markdown(
        "Fittingページの「CSVダウンロード」で得られる `fitting_results.csv` に、"
        "`temperature` や `pH` などの条件列を手動で追加すればそのまま学習データとして使えます。"
        "`batch` / `cost` / `success` / `message` は予約列として扱われ、条件列・予測対象パラメータ列"
        "としては選択できません。"
    )
    with st.expander("サンプルCSVの中身を見る"):
        st.code(SAMPLE_FITTING_RESULTS_CSV, language="csv")
    st.download_button(
        "パラメータ予測用ワイドCSVのサンプルをダウンロード",
        data=SAMPLE_FITTING_RESULTS_CSV,
        file_name="fitting_results_sample.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# タブ5: 数値解法・既知の制限
# ---------------------------------------------------------------------------
with tab_numerics:
    st.header("数値解法")
    st.markdown(
        "`scipy.integrate.solve_ivp` を `method=\"LSODA\"`（デフォルト, `rtol=1e-6, atol=1e-9`）"
        "で使用しています。求解に失敗した場合は `IntegrationError` を送出します。"
    )

    st.header("既知の制限（v1スコープ）")
    st.markdown(
        """
- 基質・酸素の維持代謝項は枯渇後もそのまま作用し続けるため、長時間シミュレーションで濃度が
  数値的に負になり得ます（警告は出ますが、クリップやイベント停止は行いません）。
- 実験データフィッティングは、同一バッチグループ内の全バッチで初期条件・運転モードを
  共有する前提です（バッチごとに異なる初期条件を自動設定する機能はありません）。
- 複数シナリオの重ね合わせ比較・DB永続化・認証などはGUIのv1スコープ外です。
- 流加プロファイルCSV（`csv`プロファイル）でブレークポイントが密な場合、ステップ状の
  不連続性によりLSODAの積分が遅くなることがあります。
- パラメータ予測・次の実験点提案も、学習データ内の全バッチで同一モデル・同一初期条件/運転モードを
  共有する前提です。次の実験点の提案は候補数×モンテカルロサンプル数だけシミュレーションを
  実行するため、条件変数が多い・格子解像度やサンプル数を大きくすると計算時間が伸びます。
        """
    )
    st.caption("より詳しい理論背景・実装の詳細は開発者向けドキュメント `README.md` を参照してください。")
