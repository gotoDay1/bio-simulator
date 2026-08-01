# biosim — バイオリアクター物質収支シミュレーター

菌体増殖・代謝物生産・基質消費・溶存酸素消費の4つのサブモデルを組み合わせ、常微分方程式（ODE）による物質収支をバッチ／流加培養（fed-batch）／連続培養（chemostat）の3つの運転モードで解くシミュレーターです。各サブモデルは戦略パターン（抽象基底クラス＋レジストリ）で差し替え可能に設計されています。コアはUI非依存の純粋なPythonライブラリで、Streamlit製GUIはその薄いラッパーです。

- コアライブラリ: `src/biosim/`
- GUI: `app/streamlit_app.py`
- テスト: `tests/`（pytest, 47件）
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
.venv/bin/streamlit run app/streamlit_app.py

# テスト
.venv/bin/python -m pytest
```

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

⚠️ **注意**: これはODE（物質収支）に組み込むための**力学的Gompertz式**であり、実験データの終点フィッティングに使われる閉形式回帰曲線 `X(t) = Xmax·exp(−exp(...))` とは異なります。上式を `dX/dt = μ·X` に代入すると `dX/dt = μ_max·X·ln(Xmax/X)` となり、これがGompertz型の非対称S字増殖曲線を与える標準的な力学モデルです。

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
- Gompertzモデルは力学的ODE形であり、実験データへの回帰フィッティング機能は含みません。
- 複数シナリオの重ね合わせ比較・DB永続化・認証などはGUIのv1スコープ外です。
