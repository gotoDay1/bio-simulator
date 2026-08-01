# biosim — バイオリアクター物質収支シミュレーター

菌体増殖・代謝物生産・基質消費・溶存酸素消費の4つのサブモデルを組み合わせ、常微分方程式（ODE）による物質収支をバッチ／流加培養（fed-batch）／連続培養（chemostat）の3つの運転モードで解くシミュレーターです。各サブモデルは戦略パターン（抽象基底クラス＋レジストリ）で差し替え可能に設計されています。コアはUI非依存の純粋なPythonライブラリで、Streamlit製GUIはその薄いラッパーです。

- コアライブラリ: `src/biosim/`
- GUI（起動ページ）: `app/Home.py`
- GUI（シミュレーション）: `app/pages/1_simulation.py`
- GUI（実験データフィッティング）: `app/pages/2_fitting.py`
- テスト: `tests/`（pytest, 77件）
- 動作デモ: `examples/batch_monod_example.py`

## セットアップ

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## 実行

```bash
# ライブラリ単体のデモ
.venv/bin/python examples/batch_monod_example.py

# GUI
.venv/bin/streamlit run app/Home.py

# テスト
.venv/bin/python -m pytest
```

---

## 実測データとの比較

GUIのサイドバー下部「実測データとの比較」から、1つのCSVファイルをアップロードしてシミュレーション結果と重ね描きできます（`src/biosim/experimental_data.py`）。列名は固定スキーマでパースされ、必須の `t` 列がない場合は `ExperimentalDataError` を出します。

| 列名 | 必須/任意 | 内容 |
|---|---|---|
| `t` | 必須 | 時間 (h) |
| `X` | 任意 | 菌体濃度・DCW (g/L) |
| `OD` | 任意 | 濁度。`X` 列が無い場合のみ `X = OD変換係数 * OD` として `X` に換算 |
| `S` | 任意 | 基質濃度 (g/L) |
| `P` | 任意 | 生産物濃度 (g/L) |
| `OTR` | 任意 | 酸素移動速度。`OxygenWithKLa`（kLaモデル）使用時のみシミュレーション側に `OTR` 列が存在 |

`X`/`S`/`P`/`OTR`/`OD` はすべて任意で、1つのCSVに好きな組み合わせで含められます。存在しない列は自動的にプロットからスキップされます。ライブラリを直接使う場合は `load_experimental_csv(path, od_conversion_factor=1.0)` を呼び、返された DataFrame を `SimulationResults.to_plotly_figure(experimental_data=...)` に渡します。

---

## 実験データフィッティング

GUIの「Fitting」ページ（`app/pages/2_fitting.py`）から、実測データに対して増殖・生産・基質・酸素の各モデルパラメータを推定できます（`src/biosim/fitting.py`）。

- **複数バッチ対応**: 実測CSVを複数アップロードすると、同一のモデル・パラメータ設定を各バッチに独立にフィットし、バッチごとの推定結果を比較できます（ファイル名からバッチ名を自動生成、重複時は自動で連番付与）。
- **パラメータごとの固定/フィット選択**: 4カテゴリ（増殖・生産・基質・酸素）それぞれについて使用するモデルを選び、各パラメータを「固定値」か「自由(推定)」（初期推定値＋上下限）のどちらにするか個別に指定できます。`Yps` のような任意パラメータは「未使用 (None)」も選べます。
- **フィッティング方式**: 閉形式回帰ではなく、候補パラメータで実際にODEをシミュレーションし、実測時刻に補間した値との残差を `scipy.optimize.least_squares` で最小化する汎用的な simulate-and-compare 方式です。そのため `GompertzGrowth` のような力学的ODE形のモデルにもそのままフィッティングできます。
- **CSVエクスポート**: フィッティング結果はワイド形式でエクスポートできます。列は `batch`（バッチ名）、`{カテゴリ}_model`（使用モデル名。例: `growth_model` = `gompertz`）、`{カテゴリ}_{パラメータ名}`（固定値またはフィット値の最終値。例: `growth_mu_max`）、`cost`（正規化残差のRMSE）、`success`（収束フラグ）です。

ライブラリを直接使う場合は `ModelSpec`/`ParameterSpec` でモデルとパラメータ設定を組み立て、`fit_batch(...)` に実測データ（`load_experimental_csv` の戻り値と同じスキーマの DataFrame）を渡します。複数バッチの `FitResult` は `fit_results_to_dataframe(...)` でまとめてCSV化できます。

---

## 理論背景

### 1. 統一物質収支の枠組み

反応器を完全混合槽（CSTR: Continuous Stirred-Tank Reactor）とみなすと、任意の溶質濃度 `C` に対する一般物質収支は次の通りです。

```
d(V·C)/dt = r(C)·V + F_in·C_feed − F_out·C
```

- `V`: 作動体積 (L)
- `r(C)`: 反応速度（濃度の時間変化率, g/L/h）
- `F_in`, `F_out`: 流入・流出流量 (L/h)
- `C_feed`: 流入液中の濃度

積の微分則 `d(VC)/dt = V·dC/dt + C·dV/dt` と全体の体積収支 `dV/dt = F_in − F_out` を代入すると、`F_out` の項が完全に相殺され、次の非常にシンプルな形に帰着します（本シミュレーターの核となる関係式）。

```
dC/dt = r(C) + (F_in/V)·(C_feed − C)
dV/dt = F_in − F_out
```

この1本の式に対して `F_in(t)`, `F_out(t)`, `C_feed` の与え方を変えるだけで、3つの運転モードすべてを表現できます（実装: `src/biosim/simulation.py` の `_rhs`、モード別の流量供給は `src/biosim/operation_modes.py`）。

| 運転モード | F_in | F_out | 帰結 |
|---|---|---|---|
| バッチ (`Batch`) | 0 | 0 | `dC/dt = r(C)`（反応項のみ）、`dV/dt = 0` |
| 流加培養 (`FedBatch`) | `F(t)`（ユーザー指定プロファイル） | 0 | `dC/dt = r(C) + (F(t)/V)(C_feed−C)`、体積は単調増加 |
| 連続培養 (`Chemostat`) | `D·V` | `D·V` | `dC/dt = r(C) + D(C_feed−C)`（教科書通りのケモスタット式）、`V` は一定 |

状態ベクトルは `[X, S, P, (C_O2), V]`（`X`: 菌体, `S`: 基質, `P`: 生産物, `C_O2`: 溶存酸素 [任意], `V`: 体積）。`C_O2` は酸素モデルが供給ダイナミクス（kLa）をサポートする場合のみ状態に追加され、そうでない場合は次元を持たず、累積OURのみ診断値として報告されます（`src/biosim/state.py` の `StateLayout` が可変長の状態ベクトルを動的に管理）。

流加培養のフィードプロファイル `F(t)` はGUI上で `constant`（一定）・`step`（単一ステップ）・`exponential`（指数関数、`mu_set` 一定を狙うfeed-forward）・`csv`（多段階ステップ、任意個数のブレークポイント）から選べます。`csv` は `time`（h）・`feed_rate`（L/h）の2列を持つCSVをアップロードし、ブレークポイント間を階段状（ステップホールド）で保持します。最初のブレークポイントより前はCSV1行目の値、最後のブレークポイントより後は最終行の値をそのまま保持します（`stepwise_feed`/`load_feed_profile_csv`、`src/biosim/operation_modes.py`・`src/biosim/feed_profile.py`）。

各カテゴリのモデルは以下の順序で呼び出され、依存関係が解決されます: **増殖 → 生産 → 基質 → 酸素**（生産・基質・酸素の速度式は増殖モデルが計算した `dX/dt` に依存するため）。

---

### 2. 菌体増殖モデル (`src/biosim/models/growth.py`)

全ての増殖モデルは比増殖速度 `μ (1/h)` を返すインターフェースに統一されており、`dX/dt = μ·X` はオーケストレーター側（`simulation.py`）で計算されます。

#### Monod式（基質律速）

```
μ(S) = μ_max · S / (Ks + S)
```

- `μ_max`: 最大比増殖速度 (1/h)
- `Ks`: 半飽和定数 (g/L) — `μ = μ_max/2` となる基質濃度

古典的なMichaelis-Menten型の飽和曲線で、`S >> Ks` で `μ→μ_max`、`S→0` で `μ→0`。

#### Logistic式（環境容量律速）

```
μ(X) = μ_max · (1 − X/Xmax)
```

- `Xmax`: 環境収容力（最大菌体濃度）(g/L)

基質濃度に依存せず、菌体濃度自体が収容力に近づくにつれ増殖速度が線形に減衰するモデル。`X = Xmax` で `μ=0`、`X > Xmax` では `μ<0`（減衰）となり、これはクリップせずそのまま許容しています。

#### Gompertz式（力学的ODE形）

```
μ(X) = μ_max · ln(Xmax / X)
```

⚠️ **注意**: これはODE（物質収支）に組み込むための**力学的Gompertz式**であり、実験データの終点フィッティングに使われる閉形式回帰曲線 `X(t) = Xmax·exp(−exp(...))` とは異なります。上式を `dX/dt = μ·X` に代入すると `dX/dt = μ_max·X·ln(Xmax/X)` となり、これがGompertz型の非対称S字増殖曲線を与える標準的な力学モデルです。simulate-and-compare方式の[実験データフィッティング](#実験データフィッティング)機能は、閉形式解を必要としないため、この力学的形のまま `μ_max`/`Xmax` を実測データから推定できます。

境界値の扱い: `X ≤ 0` または `X ≥ Xmax` のとき `ln` が未定義／負になるため、実装では `μ=0` にクリップしています。

---

### 3. 代謝物生産モデル (`src/biosim/models/product.py`)

#### Luedeking-Piret式（増殖連動＋非連動）

```
dP/dt = α·(dX/dt) + β·X
```

- `α`: 増殖連動生産係数 (g生産物 / g菌体) — 増殖速度に比例する生産（一次代謝物によくみられる）
- `β`: 非増殖連動生産係数 (1/h) — 菌体濃度に比例する生産（二次代謝物・定常期の生産をモデル化）

`α=0` で純粋な非増殖連動型、`β=0` で純粋な増殖連動型になります。

#### NoProduct

生産物を追跡しない場合の恒等モデル（`dP/dt = 0`）。

---

### 4. 基質消費モデル (`src/biosim/models/substrate.py`)

#### 収率係数＋維持代謝モデル

```
dS/dt = −(1/Yxs)·(dX/dt) − ms·X   [ − (1/Yps)·(dP/dt) ]
```

- `Yxs`: 基質に対する菌体収率 (g菌体 / g基質)
- `ms`: 維持代謝係数 (g基質 / g菌体 / h) — 増殖に使われず細胞の維持のみに消費される基質
- `Yps`（任意）: 基質に対する生産物収率 (g生産物 / g基質) — 設定した場合のみ生産物消費項を追加（デフォルトは無効）

これは古典的な Pirt の式（維持代謝を考慮した収率モデル）です。なお、`ms` の項は基質濃度に依存せず菌体濃度にのみ比例するため、基質が枯渇した後もこの項は消費を続けようとします。これは単純化されたモデルの既知の挙動で、枯渇後は `S` が数値的に負になり得ます（`BioreactorSimulation` はこれを検知して警告を出します）。

---

### 5. 酸素消費モデル (`src/biosim/models/oxygen.py`)

#### 需要のみモデル (`OxygenDemandOnly`)

```
OUR = −(1/Yxo2)·(dX/dt) − mo2·X
```

- `Yxo2`: 酸素に対する菌体収率 (g菌体 / g O2)
- `mo2`: 維持代謝酸素係数 (g O2 / g菌体 / h)

酸素が律速にならないと仮定する場合に使用し、溶存酸素濃度 `C_O2` は状態変数として持たず、累積OUR（酸素摂取速度）のみを診断値として報告します。

#### kLa供給付きモデル (`OxygenWithKLa`)

需要式は同じ（OUR）に加えて、供給側（OTR: Oxygen Transfer Rate）を次式でモデル化し、溶存酸素濃度 `C_O2` を状態変数として追跡します。

```
OTR = kLa · (Cs* − C_O2)
dC_O2/dt = OTR + OUR + (F_in/V)·(C_O2_feed − C_O2)
```

- `kLa`: 総括容量物質移動係数 (1/h) — 通気・撹拌条件を反映するパラメータ
- `Cs*`: 飽和溶存酸素濃度 (mg/L)

`C_O2 < Cs*` のとき酸素供給（`OTR>0`）、過飽和（`C_O2 > Cs*`）のときは負（脱気側）になります。

---

## パラメータ選択と拡張性

各具象モデルは `@dataclass` で実装され、デフォルト値と物理的に妥当な範囲がdocstringに明記されています。パラメータの妥当性（正値・非負制約）は `__post_init__` で検証され、不正な値は `InvalidParameterError` を送出します。

モデルの選択は `src/biosim/models/registry.py` の `dict[str, type]` レジストリを介して行われ、新しい増殖モデル（例: Contois式）を追加する場合は、

1. `GrowthModel` を継承した `@dataclass` を実装
2. `GROWTH_MODELS` 辞書に1行追加

するだけでよく、シミュレーションオーケストレーターやGUIには一切変更が不要です（GUIのセレクトボックスはレジストリのキーから自動生成されます）。

## 数値解法

`scipy.integrate.solve_ivp` を `method="LSODA"`（デフォルト, `rtol=1e-6, atol=1e-9`）で使用しています。求解に失敗した場合（`sol.success=False`）は `IntegrationError` を送出します。

## 既知の制限（v1スコープ）

- 基質・酸素の維持代謝項は枯渇後もそのまま作用し続けるため、長時間シミュレーションで濃度が数値的に負になり得ます（警告を出しますが、クリップやイベント停止は行いません）。
- 実験データフィッティングは、同一バッチグループ内の全バッチで初期条件・運転モードを共有する前提です（バッチごとに異なる初期条件を自動設定する機能は含みません）。
- 複数シナリオの重ね合わせ比較・DB永続化・認証などはGUIのv1スコープ外です。
- 流加プロファイルCSV（`csv` プロファイル）でブレークポイントが密（時間間隔が非常に短い箇所が多数）な場合、ステップ状の不連続性によりLSODAの積分が遅くなることがあります。`max_step` 等のソルバーチューニングは現状GUI/APIから公開していません。
