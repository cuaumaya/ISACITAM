import os
import warnings
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import HuberRegressor
from catboost import CatBoostRegressor, Pool

# --- Configuración ---
PANEL_CSV       = "panel_player_team_join.csv"
MODEL_DIR       = "models"
SAVE_PRED_TABLE = True
RANDOM_SEED     = 42
SEEDS_CAT       = [13, 42, 777]
N_SPLITS_INNER  = 5
TARGET = "dest_team_obv_t1"

# Características (AÑADIMOS 'year' a PLAYER_FEATS)
PLAYER_FEATS = ["player_season_obv_90", "player_season_xgchain_90", "player_season_xa_90", 
                "player_season_f3_lbp_ratio", "player_season_carry_length", "player_season_left_foot_ratio",
                "player_season_pressured_pass_length_ratio", "player_season_pressure_regains_90",
                "player_season_defensive_action_regains_90", "player_season_obv_shot_90", 
                "player_season_shot_on_target_ratio", 
                "Age"] # <--- NUEVA VARIABLE AÑADIDA
TEAM_CONTROLS_12 = ["team_season_xgd_pg", "team_season_possession", "team_season_obv_pg",
                    "team_season_gd_pg", "team_season_successful_passes_pg", "team_season_ppda",
                    "team_season_yellow_cards_pg", "team_season_deep_progressions_pg",
                    "team_season_obv_conceded_pg", "team_season_successful_crosses_into_box_pg",
                    "team_season_goals_from_corners_conceded_pg", "team_season_passing_ratio"]
TEAM_BASELINE = ["obv_for_to_team_t0"]
CATS_OPTIONAL = ["primary_position", "competition_name"]

# Parámetros CatBoost ESTABILIZADOS
cat_params_base = dict(
    iterations=3000, 
    learning_rate=0.03, 
    depth=6,
    l2_leaf_reg=20, 
    loss_function="RMSE",
    eval_metric="R2",
    early_stopping_rounds=300,
    verbose=False,
    random_strength=1.0,
)

# --- Utilidades ---
def safe_impute_numeric(df: pd.DataFrame, cols):
    df = df.copy()
    if len(cols) == 0: return df
    for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    med = df[cols].median()
    df[cols] = df[cols].fillna(med)
    return df

def add_interaction_features(df):
    """Crea las features de interacción."""
    if 'team_season_possession' in df.columns and 'team_season_xgd_pg' in df.columns:
        df['possession_xgd_interaction'] = df['team_season_possession'] * df['team_season_xgd_pg']
    if 'team_season_directness' in df.columns and 'player_season_f3_lbp_ratio' in df.columns:
        df['style_fit_direct'] = df['team_season_directness'] * df['player_season_f3_lbp_ratio']
    if 'player_season_pressure_regains_90' in df.columns and 'team_season_ppda' in df.columns:
        df['intensity_diff'] = df['player_season_pressure_regains_90'] - df['team_season_ppda']
    if 'player_season_obv_pass_90' in df.columns and 'team_season_possession' in df.columns:
        df['pass_quality_context'] = df['player_season_obv_pass_90'] * df['team_season_possession']
    return df

def print_report(title, y_true, y_pred):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n===== {title} =====")
    print(f"R²  : {r2:.3f}")
    print(f"MAE : {mae:.3f}")
    print("="*70)
    return r2, mae

# --- Carga y Prep ---
Path("models").mkdir(parents=True, exist_ok=True)
if not os.path.exists(PANEL_CSV):
    raise FileNotFoundError(f"No encuentro {PANEL_CSV}.")

df = pd.read_csv(PANEL_CSV)
df = add_interaction_features(df) # Ingeniería de Features

INTERACTION_FEATS = ['possession_xgd_interaction', 'style_fit_direct', 'intensity_diff', 'pass_quality_context']
feature_cols = PLAYER_FEATS + TEAM_CONTROLS_12 + TEAM_BASELINE + CATS_OPTIONAL + INTERACTION_FEATS
feature_cols = [c for c in feature_cols if c in df.columns]

# Limpieza Numérica (incluye 'year')
num_cols = [c for c in feature_cols if c not in CATS_OPTIONAL]
df = safe_impute_numeric(df, num_cols)

# Limpieza Categórica (solución a 'nan')
for c in CATS_OPTIONAL:
    df[c] = df[c].fillna("NA").astype("string")

# Filas válidas (target)
df = df[df[TARGET].notna()].copy()

# Holdout
holdout_season = df["season_id"].max()
df_train = df[df["season_id"] < holdout_season].copy()
df_test  = df[df["season_id"] == holdout_season].copy()

X_train = df_train[feature_cols].copy()
y_train = df_train[TARGET].astype(float).copy()
X_test  = df_test[feature_cols].copy()
y_test  = df_test[TARGET].astype(float).copy()
cat_idx = [X_train.columns.get_loc(c) for c in CATS_OPTIONAL if c in X_train.columns]

# --- CatBoost Ensemble (Monolítico) ---
gkf = GroupKFold(n_splits=min(N_SPLITS_INNER, df_train["season_id"].nunique()))
cat_oof = np.zeros(len(X_train), dtype=float)
cat_pred_test_seeds = []

print("\nEntrenando CatBoost Monolítico (INCLUYE EDAD) con ensemble de seeds...")

for seed in SEEDS_CAT:
    params = {**cat_params_base, "random_seed": seed}
    oof_seed = np.zeros(len(X_train), dtype=float)
    
    # OOF
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_train, y_train, groups=df_train["season_id"]), start=1):
        tr_pool = Pool(X_train.iloc[tr_idx], y_train.iloc[tr_idx], cat_features=cat_idx if cat_idx else None)
        te_pool = Pool(X_train.iloc[te_idx], y_train.iloc[te_idx], cat_features=cat_idx if cat_idx else None)
        model = CatBoostRegressor(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(tr_pool, eval_set=te_pool, use_best_model=True)
        oof_seed[te_idx] = model.predict(te_pool)
        
    # Full fit para holdout
    pool_tr = Pool(X_train, y_train, cat_features=cat_idx if cat_idx else None)
    pool_te = Pool(X_test,  y_test,  cat_features=cat_idx if cat_idx else None)
    final_cat = CatBoostRegressor(**params)
    final_cat.fit(pool_tr, eval_set=pool_te, use_best_model=True)
    
    cat_oof += oof_seed / len(SEEDS_CAT)
    cat_pred_test_seeds.append(final_cat.predict(pool_te))

cat_pred_test = np.mean(np.column_stack(cat_pred_test_seeds), axis=1)

# --- Meta-Modelo (Stacking Huber) ---
meta_cols = ["cat"]
Z_train = cat_oof.reshape(-1, 1)
Z_test  = cat_pred_test.reshape(-1, 1)

meta = HuberRegressor()
used_meta = "HuberRegressor"

meta.fit(Z_train, y_train)
y_pred_stack = meta.predict(Z_test)

# --- Reporte Final ---
print("\n" + "="*70)
print(f"RESULTADO FINAL: MODELO MONOLÍTICO (CON EDAD/AÑO)")
print("="*70)

R2_final, MAE_final = print_report(f"STACKING MONOLÍTICO con {used_meta}", y_test, y_pred_stack)

# --- Importancias (Top-20) ---
imp_cat = CatBoostRegressor(**{**cat_params_base, "random_seed": RANDOM_SEED})
imp_pool_tr = Pool(X_train, y_train, cat_features=cat_idx if cat_idx else None)
imp_pool_te = Pool(X_test,  y_test,  cat_features=cat_idx if cat_idx else None)
imp_cat.fit(imp_pool_tr, eval_set=imp_pool_te, use_best_model=True)

imp_vals = imp_cat.get_feature_importance(imp_pool_tr)
fi = pd.DataFrame({"feature": X_train.columns, "importance": imp_vals}).sort_values("importance", ascending=False)  

print("\nTop-20 features (CatBoost en train completo):")
print(fi.head(20).to_string(index=False))

# --- Guardado ---
if SAVE_PRED_TABLE:
    out_pred = df_test.copy()
    out_pred["y_pred_stack"] = y_pred_stack
    out_path = os.path.join("models", "predicciones_holdout_monolitico_final_con_year.csv")
    out_pred.to_csv(out_path, index=False)
    print(f"\nPredicciones finales guardadas en: {out_path}")

print("\n✅ EJECUCIÓN FINALIZADA. Revisa el R² y el lugar de 'Age' en la importancia.")

import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import os

def draw_hierarchy_diagram_matplotlib(outfile="figs/fig6_hierarchy.png"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, label, lw=1.6, bold=False):
        rect = patches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=lw, edgecolor="black", facecolor="white")
        ax.add_patch(rect)
        fs = 10 if not bold else 11.5
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        arr = FancyArrowPatch((x0, y0), (x1, y1),
                              arrowstyle="->", mutation_scale=12, linewidth=1.5)
        ax.add_patch(arr)

    # Cajas
    box(0.03, 0.62, 0.22, 0.28, "MICRO (Player)\nPLAYER_FEATS")
    box(0.03, 0.10, 0.22, 0.28, "MESO (Team)\nTEAM_CONTROLS_12")
    box(0.30, 0.36, 0.22, 0.28, "META (Fit)\nINTERACTION_FEATS")
    box(0.58, 0.36, 0.18, 0.28, "CatBoost\nEnsemble", bold=True)
    box(0.80, 0.36, 0.16, 0.28, "Huber\nRegressor", bold=True)
    ax.text(0.88, 0.74, "Predicted Team OBV (t+1)\n(dest_team_obv_t1)",
            ha="center", va="center", fontsize=11)

    # Flechas
    arrow(0.25, 0.76, 0.58, 0.50)
    arrow(0.25, 0.24, 0.58, 0.50)
    arrow(0.52, 0.50, 0.58, 0.50)
    arrow(0.76, 0.50, 0.80, 0.50)
    arrow(0.88, 0.64, 0.88, 0.50)
    arrow(0.88, 0.50, 0.88, 0.64)

    fig.tight_layout()
    os.makedirs("figs", exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

# Llamada:
draw_hierarchy_diagram_matplotlib("figs/fig6_hierarchy.png")
print("✅ Figura 6 guardada en figs/fig6_hierarchy.png")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Datos de importancia (usando tus resultados del modelo) ===
data = {
    "feature": [
        "team_season_possession", "obv_for_to_team_t0", "team_season_obv_pg",
        "team_season_passing_ratio", "possession_xgd_interaction",
        "team_season_successful_passes_pg", "team_season_gd_pg",
        "team_season_xgd_pg", "team_season_ppda", "player_season_xgchain_90",
        "team_season_obv_conceded_pg", "team_season_successful_crosses_into_box_pg",
        "player_season_obv_shot_90", "team_season_yellow_cards_pg",
        "intensity_diff", "primary_position", "player_season_carry_length",
        "player_season_defensive_action_regains_90", "player_season_left_foot_ratio",
        "style_fit_direct"
    ],
    "importance": [
        20.475, 5.939, 4.948, 4.902, 4.602, 4.209, 3.540, 3.452, 3.370,
        3.357, 3.221, 3.033, 2.993, 2.811, 2.327, 2.249, 2.242, 2.240, 2.188, 2.099
    ]
}

df_fi = pd.DataFrame(data)

# === Clasificación jerárquica manual (según nivel de variable) ===
def classify_level(feat):
    if feat.startswith("player_"):
        return "Micro (Jugador)"
    elif feat.startswith("team_"):
        return "Meso (Equipo)"
    elif feat in ["possession_xgd_interaction", "intensity_diff", "style_fit_direct", "pass_quality_context"]:
        return "Meta (Interacción)"
    else:
        return "Meta (Interacción)"

df_fi["Nivel"] = df_fi["feature"].apply(classify_level)

# === Ordenar por importancia descendente ===
df_fi = df_fi.sort_values("importance", ascending=True)

# === Configuración visual ===
sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(9, 10))

palette = {
    "Micro (Jugador)": "#4E79A7",      # azul Harvard-style
    "Meso (Equipo)": "#E15759",        # rojo
    "Meta (Interacción)": "#76B7B2"    # verde agua
}

barplot = sns.barplot(
    data=df_fi,
    y="feature",
    x="importance",
    hue="Nivel",
    palette=palette,
    dodge=False
)

# === Personalización ===
plt.title("Figura 7. Distribución jerárquica de importancia de variables (CatBoost)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Importancia relativa (gain-based importance)", fontsize=12)
plt.ylabel("")
plt.legend(title="Nivel jerárquico", loc="lower right", frameon=True)

# === Etiquetas directas de valores ===
for i, v in enumerate(df_fi["importance"]):
    plt.text(v + 0.2, i, f"{v:.2f}", va="center", fontsize=9)

plt.tight_layout()

# === Guardado ===
Path("figs").mkdir(exist_ok=True)
plt.savefig("figs/fig7_feature_importance.png", dpi=400, bbox_inches="tight")
plt.show()

# === FIGURA 8: SHAP GLOBAL (CatBoost nativo) ===
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # <- evita errores GUI en Windows/servers
import matplotlib.pyplot as plt

# Asume que ya tienes:
# - imp_cat: CatBoostRegressor entrenado en train completo (como en tu script)
# - X_train, X_test, y_test, cat_idx, imp_pool_tr, imp_pool_te definidos
# Si no, crea el Pool según tengas: Pool(X_train, y_train, cat_features=cat_idx or None)

# 1) Elegimos un conjunto para explicar: test si existe, si no una muestra del train
use_X = X_test.copy() if 'X_test' in globals() and len(X_test) > 0 else X_train.sample(min(2000, len(X_train)), random_state=42)

# 2) Obtenemos SHAP values nativos de CatBoost (incluye columna base value al final)
from catboost import Pool
pool_use = Pool(use_X, cat_features=cat_idx if 'cat_idx' in globals() else None)
shap_values = imp_cat.get_feature_importance(pool_use, type='ShapValues')  # shape: [n_rows, n_features+1]

# 3) Separamos matriz SHAP (sin base value)
shap_matrix = shap_values[:, :-1]  # columnas 0..n_features-1 son SHAP por feature
feature_names = list(use_X.columns)

# 4) |SHAP| medio por feature
mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
df_shap = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
})

# 5) Clasificación jerárquica para colorear
def classify_level(feat: str) -> str:
    if feat.startswith("player_"):
        return "Micro (Jugador)"
    elif feat.startswith("team_"):
        return "Meso (Equipo)"
    elif feat in {"possession_xgd_interaction","style_fit_direct","intensity_diff","pass_quality_context","pressure_vs_da_ratio","age_squared"}:
        return "Meta (Interacción)"
    elif feat in {"primary_position","competition_name"}:
        # categóricas: dejan que CatBoost haga ordered encoding; conceptualmente son micro/meta
        return "Micro (Jugador)" if feat == "primary_position" else "Meta (Interacción)"
    else:
        return "Meta (Interacción)"

df_shap["Nivel"] = df_shap["feature"].apply(classify_level)

# 6) Orden descendente y opcional: quedarnos con Top-20
df_shap = df_shap.sort_values("mean_abs_shap", ascending=True)
topk = 20
df_plot = df_shap.tail(topk)

# 7) Plot horizontal
palette = {
    "Micro (Jugador)": "#4E79A7",      # azul
    "Meso (Equipo)": "#E15759",        # rojo
    "Meta (Interacción)": "#76B7B2"    # verde agua
}

fig, ax = plt.subplots(figsize=(9, 10))
for i, (feat, val, lvl) in enumerate(zip(df_plot["feature"], df_plot["mean_abs_shap"], df_plot["Nivel"])):
    ax.barh(feat, val, color=palette.get(lvl, "#999999"), edgecolor="none")
    ax.text(val + df_plot["mean_abs_shap"].max()*0.01, i, f"{val:.3f}", va="center", fontsize=9)

ax.set_title("Figura 8. Impacto global (|SHAP| medio) por nivel jerárquico", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("|SHAP| medio (magnitud de impacto en la predicción)")
ax.set_ylabel("")
# leyenda manual
handles = [plt.Line2D([0],[0], color=c, lw=10) for c in [palette["Micro (Jugador)"], palette["Meso (Equipo)"], palette["Meta (Interacción)"]]]
ax.legend(handles, ["Micro (Jugador)", "Meso (Equipo)", "Meta (Interacción)"], loc="lower right", frameon=True)

plt.tight_layout()
Path("figs").mkdir(exist_ok=True)
out_path = Path("figs/fig8_shap_global.png")
plt.savefig(out_path, dpi=400, bbox_inches="tight")
print(f"✅ Figura 8 guardada en: {out_path}")

# === FIGURA 9: SHAP LOCAL (waterfall simple) ===
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import Pool

# 1) Elegimos un ejemplo del conjunto de explicación
row_idx = 0  # cambia a la fila que quieras inspeccionar
x_row = use_X.iloc[[row_idx]]
pool_row = Pool(x_row, cat_features=cat_idx if 'cat_idx' in globals() else None)

# 2) SHAP para esa fila (incluye base value al final)
sv_row = imp_cat.get_feature_importance(pool_row, type='ShapValues')  # shape: (1, n_features+1)
base_value = sv_row[0, -1]
shaps = sv_row[0, :-1]

# 3) Ordenar por contribución absoluta y quedarse con top 12 para claridad
feature_names = use_X.columns.tolist()
order = np.argsort(np.abs(shaps))[::-1]
k = min(12, len(order))
idxs = order[:k]
feat_top = [feature_names[i] for i in idxs]
shap_top = shaps[idxs]

# 4) Predicción total
pred = base_value + shaps.sum()

# 5) Waterfall
colors = ["#2E7D32" if v >= 0 else "#C62828" for v in shap_top]
fig, ax = plt.subplots(figsize=(9, 6))
running = base_value
ax.axvline(base_value, color="#555555", linestyle="--", linewidth=1)
ypos = np.arange(k)

for i, (f, v) in enumerate(zip(feat_top, shap_top)):
    ax.barh(i, v, left=running if v>=0 else running+v, color=colors[i])
    running += v
    ax.text(running + (0.01*np.sign(v)), i, f"{v:+.3f}", va="center", fontsize=9)

ax.scatter([base_value], [-1], color="#555", s=30)  # base marker (opcional)
ax.set_yticks(ypos)
ax.set_yticklabels(feat_top)
ax.set_xlabel("Contribución SHAP")
ax.set_title("Figura 9. Explicación local (waterfall) — contribuciones a la predicción", fontsize=13, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
out_path_local = Path("figs/fig9_shap_local_waterfall.png")
plt.savefig(out_path_local, dpi=400, bbox_inches="tight")
print(f"✅ Figura 9 guardada en: {out_path_local}")
print(f"Predicción total (modelo): {pred:.3f} | Base value: {base_value:.3f}")

import numpy as np
import matplotlib.pyplot as plt
from catboost import Pool

# Suponiendo que imp_cat y X_test existen
# Elegimos un jugador de referencia
row = X_test.iloc[[0]].copy()

# Rango paramétrico
pos_range = np.linspace(X_test['team_season_possession'].mean()-2*X_test['team_season_possession'].std(),
                        X_test['team_season_possession'].mean()+2*X_test['team_season_possession'].std(), 25)
ppda_range = np.linspace(X_test['team_season_ppda'].mean()-2*X_test['team_season_ppda'].std(),
                         X_test['team_season_ppda'].mean()+2*X_test['team_season_ppda'].std(), 25)

Z = np.zeros((len(pos_range), len(ppda_range)))

for i, pos in enumerate(pos_range):
    for j, ppda in enumerate(ppda_range):
        tmp = row.copy()
        tmp['team_season_possession'] = pos
        tmp['team_season_ppda'] = ppda
        pool_tmp = Pool(tmp, cat_features=cat_idx if 'cat_idx' in globals() else None)
        Z[i, j] = imp_cat.predict(pool_tmp)

# Plot heatmap
plt.figure(figsize=(8,6))
plt.imshow(Z, origin='lower', aspect='auto',
           extent=[ppda_range.min(), ppda_range.max(), pos_range.min(), pos_range.max()],
           cmap='coolwarm')
plt.colorbar(label='Predicción OBV proyectado')
plt.xlabel("PPDA (presión colectiva)")
plt.ylabel("Posesión del equipo (%)")
plt.title("Figura 10. Mapa contrafactual de sensibilidad del OBV proyectado")
plt.tight_layout()
plt.savefig("figs/fig10_contrafactual_sensitivity.png", dpi=400)
plt.show()

# ===========================
# FIGURA 10: Sensibilidad contrafactual de OBV
# ===========================
import numpy as np
import pandas as pd
from pathlib import Path

# Evita problemas de GUI en Windows / servidores headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import Pool

# ------- Config -------
Path("figs").mkdir(exist_ok=True)
OUT_STATIC = "figs/fig10_contrafactual_sensitivity.png"
OUT_INTERACTIVE = "figs/fig10b_contrafactual_plotly.html"

# Variables meso a perturbar (ejes del mapa)
VAR_X = "team_season_ppda"          # eje X
VAR_Y = "team_season_possession"    # eje Y
# Opcional: mantener un tercer contexto fijo o barrerlo
VAR_FIXED = "team_season_passing_ratio"   # contexto fijo
FIXED_STRATEGY = "median"  # "median" | "row" | float valor directo

# ------- Helpers -------
def get_fixed_value(df, col, row_series, strategy="median"):
    if col not in df.columns:
        return None
    if strategy == "row":
        return float(row_series[col])
    if strategy == "median":
        return float(df[col].median())
    # Si pasan un número
    if isinstance(strategy, (int, float)):
        return float(strategy)
    return float(df[col].median())

def make_ranges(df, col, n=31, n_std=2.0):
    """Rango robusto ± n_std*std alrededor de la media, acotado a [p1, p99] para evitar extrapolación loca."""
    if col not in df.columns:
        raise ValueError(f"Columna no disponible: {col}")
    m = df[col].mean()
    s = df[col].std(ddof=0)
    a = m - n_std * s
    b = m + n_std * s
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    a = max(a, lo)
    b = min(b, hi)
    return np.linspace(a, b, n)

def simulate_grid(model, base_row: pd.Series, x_vals, y_vals, var_x: str, var_y: str,
                  var_fixed=None, fixed_val=None, cat_idx=None):
    Z = np.zeros((len(y_vals), len(x_vals)))
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            tmp = base_row.to_frame().T.copy()
            tmp[var_x] = x
            tmp[var_y] = y
            if var_fixed is not None and fixed_val is not None and var_fixed in tmp.columns:
                tmp[var_fixed] = fixed_val
            pool = Pool(tmp, cat_features=cat_idx if cat_idx else None)
            Z[i, j] = float(model.predict(pool))
    return Z

def plot_heatmap(x_vals, y_vals, Z, xlabel, ylabel, title, outfile):
    plt.figure(figsize=(8.6, 6.8))
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        cmap="coolwarm"
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Predicción OBV proyectado", rotation=90)
    CS = plt.contour(x_vals, y_vals, Z, colors="k", linewidths=0.4, alpha=0.5)
    plt.clabel(CS, inline=True, fontsize=7, fmt="%.2f")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outfile, dpi=400, bbox_inches="tight")
    print(f"✅ Figura 10 (estática) guardada en: {outfile}")

# ------- Selección del jugador de referencia -------
# Usa el primer registro de holdout; cambia a otro índice si quieres
row_idx = 0
row = X_test.iloc[row_idx].copy()

# ------- Construcción de rangos -------
x_vals = make_ranges(X_test, VAR_X, n=41, n_std=2.0)
y_vals = make_ranges(X_test, VAR_Y, n=41, n_std=2.0)

# Fija el contexto VAR_FIXED
fixed_val = get_fixed_value(X_test, VAR_FIXED, row, strategy=FIXED_STRATEGY)

# ------- Simulación -------
Z = simulate_grid(
    model=imp_cat,
    base_row=row,
    x_vals=x_vals,
    y_vals=y_vals,
    var_x=VAR_X,
    var_y=VAR_Y,
    var_fixed=VAR_FIXED,
    fixed_val=fixed_val,
    cat_idx=cat_idx if 'cat_idx' in globals() else None
)

# ------- Visual (paper-ready) -------
x_lab = "PPDA (intensidad de presión colectiva; menor = más presión)"
y_lab = "Posesión del equipo destino (%)"
ttl = "Figura 10. Mapa contrafactual de sensibilidad del OBV proyectado\n(jugador fijo; variación conjunta de PPDA y Posesión)"
plot_heatmap(x_vals, y_vals, Z, x_lab, y_lab, ttl, OUT_STATIC)

# ------- (Opcional) Versión interactiva con Plotly -------
try:
    import plotly.graph_objects as go
    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=x_vals,
            y=y_vals,
            coloraxis="coloraxis"
        )
    )
    fig.update_layout(
        title="Figura 10b. Heatmap contrafactual interactivo (OBV proyectado)",
        xaxis_title=x_lab,
        yaxis_title=y_lab,
        coloraxis=dict(colorscale="RdBu", reversescale=True, colorbar_title="OBV"),
        template="plotly_white",
        width=900, height=650
    )
    fig.write_html(OUT_INTERACTIVE, include_plotlyjs="cdn")
    print(f"✅ Figura 10b (interactiva) guardada en: {OUT_INTERACTIVE}")
except Exception as e:
    print("Plotly no disponible o falló la exportación interactiva:", e)

import numpy as np
from catboost import Pool

def local_elasticities(model, xrow, cat_idx=None, h_pos=0.5, h_ppda=0.5,
                       c_pos='team_season_possession', c_ppda='team_season_ppda'):
    base = xrow.copy()
    def pred(z):
        return float(model.predict(Pool(z.to_frame().T, cat_features=cat_idx if cat_idx else None)))
    y0 = pred(base)

    z_pos_up, z_pos_dn = base.copy(), base.copy()
    z_pos_up[c_pos] += h_pos; z_pos_dn[c_pos] -= h_pos
    dy_dpos = (pred(z_pos_up) - pred(z_pos_dn)) / (2*h_pos)

    z_ppda_up, z_ppda_dn = base.copy(), base.copy()
    z_ppda_up[c_ppda] += h_ppda; z_ppda_dn[c_ppda] -= h_ppda
    dy_dppda = (pred(z_ppda_up) - pred(z_ppda_dn)) / (2*h_ppda)

    return dict(y0=y0, d_OBV_d_pos=dy_dpos, d_OBV_d_ppda=dy_dppda)

row_idx = 0  # jugador de referencia del holdout
elas = local_elasticities(imp_cat, X_test.iloc[row_idx], cat_idx=cat_idx)
print("Elasticidades locales:", elas)

import pandas as pd
from catboost import Pool

def regime_effect(model, xrow, col, lo, hi, cat_idx=None, keep_others=None):
    base_lo = xrow.copy(); base_lo[col] = lo
    base_hi = xrow.copy(); base_hi[col] = hi
    if keep_others:
        for k,v in keep_others.items():
            base_lo[k] = v; base_hi[k] = v
    def pred(z):
        return float(model.predict(Pool(z.to_frame().T, cat_features=cat_idx if cat_idx else None)))
    return pred(base_hi) - pred(base_lo)

row = X_test.iloc[row_idx].copy()
p = {
  "pos_25": X_test['team_season_possession'].quantile(0.25),
  "pos_75": X_test['team_season_possession'].quantile(0.75),
  "ppda_25": X_test['team_season_ppda'].quantile(0.25),
  "ppda_75": X_test['team_season_ppda'].quantile(0.75),
  "pass_med": X_test['team_season_passing_ratio'].median(),
}
effects = []
effects.append(dict(var="Posesión",  delta=regime_effect(imp_cat, row, 'team_season_possession',
                    p['pos_25'], p['pos_75'], cat_idx=cat_idx,
                    keep_others={'team_season_passing_ratio':p['pass_med']})))
effects.append(dict(var="PPDA",      delta=regime_effect(imp_cat, row, 'team_season_ppda',
                    p['ppda_75'], p['ppda_25'], cat_idx=cat_idx,
                    keep_others={'team_season_passing_ratio':p['pass_med']})))  # menor PPDA = más presión

df_eff = pd.DataFrame(effects)
print(df_eff)
df_eff.to_csv("figs/tabla_effects_p25_p75.csv", index=False)