import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ====== CONFIG ======
st.set_page_config(
    page_title="OBV Compatibility Studio — Landing + Predictor",
    page_icon="⚽",
    layout="wide",
)

# ====== RUTAS/MODELOS ======
MODEL_DIR = "models"
CATBOOST_PATH = os.path.join(MODEL_DIR, "catboost_base.cbm")
META_PATH = os.path.join(MODEL_DIR, "meta_huber.pkl")

# ====== ESTILO GLOBAL (CSS) ======
st.markdown("""
<style>
:root {
  --accent:#10b981;   /* verde esmeralda */
  --blue:#2563eb;     /* azul intenso */
  --amber:#facc15;    /* dorado suave */
  --slate:#050505;    /* fondo casi negro */
}
.block-container { padding-top: 0.4rem; background-color: #050505; color: #e5e5e5; }

.hero {
  min-height: 86vh;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(1200px 600px at 10% -10%, rgba(16,185,129,0.05), transparent),
              radial-gradient(1200px 600px at 110% 110%, rgba(59,130,246,0.08), transparent),
              linear-gradient(135deg, #0a0a0a 0%, #050505 50%, #0a0a0a 100%);
  color: #f9fafb;
  text-align: center;
  overflow: hidden;
  border-bottom: 1px solid #1f2937;
}
.badge {
  display:inline-block;
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid rgba(16,185,129,0.4);
  background: rgba(16,185,129,0.1);
  color: #34d399;
  font-weight: 600;
  margin-bottom: 18px;
}
.h1-grad {
  font-weight: 800;
  line-height: 1.05;
  background: linear-gradient(90deg, #34d399, #60a5fa, #facc15);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-cta button {
  border-radius: 12px;
  padding: 12px 22px;
  font-weight: 700;
  cursor: pointer;
  border: 1px solid #27272a;
}
.btn-primary {
  background: linear-gradient(90deg,#0ea5e9,#1e40af);
  color:#f9fafb;
  border:none;
  box-shadow: 0 6px 26px rgba(14,165,233,.35);
}
.btn-ghost {
  background:#0b0b0b;
  border:1px solid #1e3a8a;
  color:#93c5fd;
}
.btn-ghost:hover {
  background: rgba(30,64,175,0.25);
}

.kpi-card {
  background: #111827;
  border:1px solid #1f2937;
  border-radius: 14px;
  padding: 16px;
  box-shadow: inset 0 0 20px rgba(255,255,255,0.02);
}
.kpi-title { color:#94a3b8; font-size:0.9rem; margin-bottom:6px;}
.kpi-value { font-size:1.9rem; font-weight:800; color:#f9fafb; }

.section {
  padding: 56px 8px;
  border-bottom: 1px solid #1f2937;
  background: #0b0b0b;
}
.section.alt { background: #0d0d0d; }
.h2-grad {
  text-align:center;
  font-weight:800;
  margin-bottom: 6px;
  background: linear-gradient(90deg, #34d399, #60a5fa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.section p.lead {
  color:#d1d5db;
  text-align:center;
  font-size:1.15rem;
  max-width: 860px;
  margin: 0 auto 26px auto;
}
.card {
  background: #111827;
  border:1px solid #1f2937;
  border-radius: 16px;
  padding: 18px;
}
.card:hover { border-color: rgba(99,102,241,0.45); }

.footer {
  padding: 20px 8px;
  background:#0a0a0a;
  border-top: 1px solid #1f2937;
  color:#94a3b8;
  text-align:center;
}
.legend-dot {
  display:inline-block;
  width:12px;
  height:12px;
  border-radius:999px;
  margin-right:6px;
}
.small-muted { color:#94a3b8; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ====== DATOS DEMO PARA CHARTS ======
feature_importance = pd.DataFrame({
    "name": ['Posesión','OBV Team t0','OBV pg','Passing Ratio','Posesión×xGD','Pases pg','xGChain 90','PPDA'],
    "value":[20.47, 5.94, 4.95, 4.90, 4.60, 4.21, 3.36, 3.37],
    "category":['meso','meso','meso','meso','meta','meso','micro','meso']
})
perf_metrics = [{"metric":"R²","value":0.449,"max":1.0},{"metric":"MAE","value":0.195,"max":0.5}]

# ====== ESTADO ======
if "show_predictor" not in st.session_state:
    st.session_state.show_predictor = False

# ====== UTILIDADES PREDICCIÓN (HEURÍSTICA CLIENTE) ======
def calculate_prediction(player, team):
    possession_factor = team["possession"] / 100.0
    ppda_factor = max(0.0, 1.0 - (team["ppda"] / 20.0))
    age_factor = 1.0 - abs(player["age"] - 27.0) / 20.0

    possession_interaction = possession_factor * team["xgd_pg"]
    intensity_diff = player["pressure_regains_90"] - (1.0 / team["ppda"])
    style_fit = team["passing_ratio"] * (player["xgchain_90"] / 0.3)

    base_obv = (player["obv_90"] * 0.15
                + player["xgchain_90"] * 0.20
                + player["xa_90"] * 0.10)

    context_multiplier = (possession_factor * 0.35
                          + ppda_factor * 0.15
                          + (team["obv_pg"] / 2.0) * 0.25
                          + possession_interaction * 0.15
                          + age_factor * 0.10)

    interaction_bonus = (style_fit * 0.08
                         + max(0.0, intensity_diff) * 0.05)

    predicted_obv = max(0.0, min(2.0, base_obv + context_multiplier + interaction_bonus))
    confidence = min(95.0, 70.0 + (age_factor * 10.0) + (possession_factor * 15.0))
    compatibility = min(100.0, (possession_interaction * 100.0) + (style_fit * 20.0) + max(0.0, 30.0 - abs(intensity_diff * 20.0)))

    return {
        "predictedOBV": round(predicted_obv, 3),
        "confidence": round(confidence, 1),
        "compatibility": int(round(compatibility)),
        "factors": {
            "possession": int(round(possession_factor * 100.0)),
            "intensity": int(round(ppda_factor * 100.0)),
            "age": int(round(age_factor * 100.0)),
            "style": int(round(min(100.0, style_fit * 50.0)))
        }
    }

def radar_figure(factors_dict):
    cats = ["Posesión", "Intensidad", "Edad", "Estilo"]
    vals = [factors_dict["possession"], factors_dict["intensity"], factors_dict["age"], factors_dict["style"]]
    cats_closed = cats + [cats[0]]
    vals_closed = vals + [vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill='toself', name='Score',
        line=dict(width=2), opacity=0.7
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=320, margin=dict(l=10,r=10,t=10,b=10)
    )
    return fig

# ====== CARGA MODELOS REALES (CatBoost + Huber) ======
@st.cache_resource(show_spinner=False)
def load_models():
    models = {"cat": None, "meta": None, "feature_cols": None}
    try:
        from catboost import CatBoostRegressor, Pool
        import joblib

        if os.path.exists(CATBOOST_PATH):
            cat_model = CatBoostRegressor()
            cat_model.load_model(CATBOOST_PATH)
            models["cat"] = cat_model

        if os.path.exists(META_PATH):
            models["meta"] = joblib.load(META_PATH)

        # Lista de features principales usadas en tu entrenamiento (las que tenemos chance de mapear)
        player_feats = [
            "player_season_obv_90","player_season_xgchain_90","player_season_xa_90",
            "player_season_carry_length","player_season_pressure_regains_90","Age"
        ]
        team_feats = [
            "team_season_possession","team_season_ppda","team_season_passing_ratio",
            "team_season_xgd_pg","team_season_obv_pg","obv_for_to_team_t0"
        ]
        # Interacciones disponibles con sliders
        inter_feats = ["possession_xgd_interaction","intensity_diff","style_fit_direct","pass_quality_context"]
        # Categóricas opcionales
        cats_optional = ["primary_position","competition_name"]

        models["feature_cols"] = player_feats + team_feats + cats_optional + inter_feats
    except Exception as e:
        st.warning(f"No pude inicializar carga de modelos: {e}")
    return models

models = load_models()

def build_feature_row_from_sliders(player, team):
    row = {
        "player_season_obv_90": player["obv_90"],
        "player_season_xgchain_90": player["xgchain_90"],
        "player_season_xa_90": player["xa_90"],
        "player_season_carry_length": player["carry_length"],
        "player_season_pressure_regains_90": player["pressure_regains_90"],
        "Age": player["age"],

        "team_season_possession": team["possession"],     # %
        "team_season_ppda": team["ppda"],
        "team_season_passing_ratio": team["passing_ratio"],
        "team_season_xgd_pg": team["xgd_pg"],
        "team_season_obv_pg": team["obv_pg"],
        "obv_for_to_team_t0": team["obv_pg"],  # proxy si no tienes el t0 real

        "primary_position": "NA",
        "competition_name": "NA",

        # Interacciones “disponibles” con sliders
        "possession_xgd_interaction": (team["possession"]/100.0) * team["xgd_pg"],
        "intensity_diff": player["pressure_regains_90"] - team["ppda"],  # proxy

        # ⬇️ Campos del entrenamiento que no están en UI -> NaN (o define una heurística si quieres)
        "style_fit_direct": np.nan,              # requería directness × f3_lbp_ratio
        "pass_quality_context": np.nan,          # requería obv_pass_90 × possession
        "player_season_f3_lbp_ratio": np.nan,    # <--- EL QUE TE FALTABA
        "player_season_obv_pass_90": np.nan,     # por si el modelo lo pidió para pass_quality_context
        "team_season_directness": np.nan,        # por si el modelo lo pidió para style_fit_direct
    }
    return pd.DataFrame([row])

def predict_with_trained_models(df_features):
    """
    Predice con CatBoost + (opcional) Huber.
    1) Obtiene el ORDEN REAL esperado por el modelo (models['required']).
    2) Crea columnas faltantes con NaN.
    3) Reordena columnas exactamente como el modelo las espera.
    4) Elimina columnas de más para evitar desalineaciones.
    """
    try:
        from catboost import Pool
        cat_model = models["cat"]
        meta = models["meta"]
        required = models.get("required")

        if (cat_model is None) or (required is None):
            return None

        df = df_features.copy()

        # Asegura TODAS las columnas requeridas
        missing = [c for c in required if c not in df.columns]
        for col in missing:
            df[col] = np.nan

        # Reordena exactamente como el modelo lo pide y tira extras
        df = df[required]

        # Trata categóricas si algunas son requeridas
        cat_candidates = ["primary_position","competition_name"]
        cat_idx = [df.columns.get_loc(c) for c in cat_candidates if c in df.columns]

        pool = Pool(df, cat_features=cat_idx if len(cat_idx)>0 else None)
        base_pred = cat_model.predict(pool).reshape(-1,)

        if meta is not None:
            z = base_pred.reshape(-1,1)
            stacked = meta.predict(z)
            return stacked
        return base_pred

    except Exception as e:
        st.warning(f"No pude usar el modelo entrenado (fallback a demo). Detalle: {e}")
        return None
# ===========================================================
#                  MODO PREDICTOR (pantalla)
# ===========================================================
if st.session_state.show_predictor:
    st.markdown("""
    <div style="position:sticky;top:0;z-index:50;background:#0b1220dd;border-bottom:1px solid #1f2937;padding:10px 0;">
      <div style="max-width:1160px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;padding:0 10px;">
        <div style="display:flex;gap:10px;align-items:center;color:#34d399;font-weight:800">🧠 ITQ • Predictor</div>
        <a href="#" onClick="return false;" style="text-decoration:none;">
          <span style="background:#111827;border:1px solid #334155;color:#e5e7eb;padding:8px 12px;border-radius:10px;">← Volver</span>
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # Botón Volver
    if st.button("← Volver al Landing", type="secondary", use_container_width=True):
        st.session_state.show_predictor = False
        st.rerun()

    st.markdown("""
    <div style="text-align:center;margin-top:10px">
      <span class='badge'>Simulador de Compatibilidad Táctica</span>
      <h1 class="h1-grad" style="font-size:clamp(28px,4vw,48px)">Predictor de OBV Contextual</h1>
      <p class="small-muted">Ajusta los parámetros del jugador y del equipo para proyectar el rendimiento esperado.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.07, 1.07], gap="large")

    # --- Perfil del Jugador (sliders)
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 👤 Perfil del Jugador")
        obv_90 = st.slider("OBV por 90 min", 0.00, 0.50, 0.15, 0.01)
        xgchain_90 = st.slider("xGChain por 90", 0.00, 0.80, 0.25, 0.01)
        xa_90 = st.slider("xA por 90", 0.00, 0.30, 0.08, 0.01)
        carry_length = st.slider("Longitud de Conducción (m)", 0.0, 20.0, 8.5, 0.1)
        pressure_regains_90 = st.slider("Recuperaciones por Presión", 0.0, 5.0, 1.2, 0.1)
        age = st.slider("Edad", 18, 38, 25, 1)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Perfil del Equipo (sliders)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🟢 Perfil del Equipo")
        possession = st.slider("Posesión (%)", 30, 75, 55, 1)
        ppda = st.slider("PPDA (Intensidad, ↓=más presión)", 5.0, 20.0, 10.5, 0.1)
        passing_ratio = st.slider("Passing Ratio", 0.5, 3.0, 1.8, 0.1)
        xgd_pg = st.slider("xGD por Partido", -1.0, 2.0, 0.30, 0.05)
        obv_pg = st.slider("OBV por Partido", 0.0, 3.0, 1.20, 0.05)
        st.markdown("</div>", unsafe_allow_html=True)

        # === NUEVO: Usar modelo entrenado + uploader ===
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        use_model = st.toggle("🎯 Usar modelo entrenado (CatBoost + Huber)", value=False, help="Requiere archivos en /models. Recomiendo subir un CSV con una fila de features completas para máxima compatibilidad.")
        uploaded = st.file_uploader("Sube un CSV con **una fila** de features (opcional pero recomendado)", type=["csv"])
        st.caption("Si no se sube CSV o faltan columnas, intentaremos predecir con las variables del simulador y completados automáticos. Si falla, usaremos la fórmula demo.")
        st.markdown("</div>", unsafe_allow_html=True)

        run = st.button("🚀 Calcular Predicción", use_container_width=True)

    # --- Resultados
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Resultado y Confianza")

        if run:
            # 1) Resultado demo (heurística) SIEMPRE disponible como respaldo
            player = dict(obv_90=obv_90, xgchain_90=xgchain_90, xa_90=xa_90,
                          carry_length=carry_length, pressure_regains_90=pressure_regains_90, age=age)
            team = dict(possession=possession, ppda=ppda, passing_ratio=passing_ratio, xgd_pg=xgd_pg, obv_pg=obv_pg)
            demo_res = calculate_prediction(player, team)

            final_res = None
            reason = "demo"

            # 2) Intento de predicción con modelo real
            if use_model and (models["cat"] is not None):
                try:
                    if uploaded is not None:
                        df_up = pd.read_csv(uploaded)
                        if len(df_up) > 1:
                            st.warning("El CSV tiene más de una fila; usaré la primera.")
                            df_up = df_up.iloc[[0]].copy()
                        yhat = predict_with_trained_models(df_up)
                        if yhat is not None:
                            ypred = float(yhat[0])
                            final_res = {
                                "predictedOBV": round(ypred, 3),
                                "confidence": 80.0,   # puedes enriquecer desde backend (p.ej. varianza entre seeds)
                                "compatibility": demo_res["compatibility"],  # mantenemos tu lectura táctica
                                "factors": demo_res["factors"]
                            }
                            reason = "modelo_subido"
                    else:
                        # Construimos una fila con sliders y tratamos de predecir
                        df_feat = build_feature_row_from_sliders(player, team)
                        yhat = predict_with_trained_models(df_feat)
                        if yhat is not None:
                            ypred = float(yhat[0])
                            final_res = {
                                "predictedOBV": round(ypred, 3),
                                "confidence": 78.0,
                                "compatibility": demo_res["compatibility"],
                                "factors": demo_res["factors"]
                            }
                            reason = "modelo_sliders"
                except Exception as e:
                    st.warning(f"No se pudo predecir con el modelo real: {e}")

            # 3) Fallback si no hubo modelo
            if final_res is None:
                final_res = demo_res

            # 4) Render
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.metric("OBV proyectado", f"{final_res['predictedOBV']}")
            with c2:
                st.metric("Confianza", f"{final_res['confidence']}%")
            with c3:
                st.metric("Compatibilidad", f"{final_res['compatibility']}%")

            if reason.startswith("modelo"):
                st.caption("✅ Predicción generada con el **modelo entrenado** " + ("(CSV subido)" if reason=="modelo_subido" else "(a partir de sliders)"))
            else:
                st.caption("🔧 Predicción generada con la **fórmula demo** (sube una fila CSV o agrega más columnas para usar el modelo real).")

            st.markdown("---")
            st.markdown("#### 🧭 Factores de Compatibilidad (Radar)")
            fig_radar = radar_figure(final_res["factors"])
            st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("#### 📝 Interpretación")
            comp = final_res["compatibility"]
            if comp > 70:
                st.success("**Alta compatibilidad.** El perfil del jugador se alinea fuertemente con la identidad táctica del equipo.")
            elif comp < 50:
                st.warning("**Compatibilidad moderada.** Hay fricciones entre el estilo del jugador y las características del equipo.")
            else:
                st.info("**Compatibilidad intermedia.** La adaptación depende de ajustes tácticos específicos (rol, alturas, pressing triggers).")

            st.caption("Nota: proyección basada en un esquema jerárquico que explica ~45% de la varianza del OBV contextual (holdout temporal).")
        else:
            st.info("Ajusta los parámetros y pulsa **Calcular Predicción** para ver resultados.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ===========================================================
#                     MODO LANDING (portada)
# ===========================================================
# --- HERO ---
st.markdown("<div class='hero'>", unsafe_allow_html=True)
st.markdown("<div>", unsafe_allow_html=True)
st.markdown("<span class='badge'>ITAM • Investigación 2025</span>", unsafe_allow_html=True)
st.markdown("<h1 class='h1-grad' style='font-size:clamp(32px,6vw,72px)'>El Fútbol como Sistema<br/>Predictivamente Inteligible</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#cbd5e1; font-size:1.2rem; max-width:920px; margin: 14px auto 20px auto;'>Una arquitectura jerárquica para modelar la inteligencia colectiva y la compatibilidad táctica mediante Machine Learning.</p>", unsafe_allow_html=True)

ccta1, ccta2 = st.columns([1,1], gap="small")
with ccta1:
    if st.button("🧮 Hacer Predicción", use_container_width=True):
        st.session_state.show_predictor = True
        st.rerun()
with ccta2:
    pdf_path = "Modelo_Prediccion_OBV.pdf"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        label="📄 Descargar Paper Completo",
        data=pdf_bytes,
        file_name="Modelo_Prediccion_OBV.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.write("")
kc1,kc2,kc3,kc4 = st.columns(4)
with kc1:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>R² Score</div><div class='kpi-value'>0.449</div></div>", unsafe_allow_html=True)
with kc2:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>MAE</div><div class='kpi-value'>0.195</div></div>", unsafe_allow_html=True)
with kc3:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Variables</div><div class='kpi-value'>45+</div></div>", unsafe_allow_html=True)
with kc4:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Temporadas</div><div class='kpi-value'>4</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- ARQUITECTURA ---
st.markdown("<div class='section alt'>", unsafe_allow_html=True)
st.markdown("<h2 class='h2-grad' style='font-size:clamp(26px,3.6vw,44px)'>Arquitectura del Modelo</h2>", unsafe_allow_html=True)
st.markdown("<p class='lead'>Un stacked ensemble jerárquico que integra CatBoost y Huber Regression para capturar interacciones no lineales multiescala.</p>", unsafe_allow_html=True)

colA, colB, colC = st.columns(3)
with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:34px'>⚽</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#60a5fa; font-weight:800'>Nivel Micro</h3>", unsafe_allow_html=True)
    st.caption("Jugador Individual")
    st.markdown("• OBV_90  \n• xGChain_90  \n• Carry Length  \n• Age")
    st.markdown("</div>", unsafe_allow_html=True)
with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:34px'>👥</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#34d399; font-weight:800'>Nivel Meso</h3>", unsafe_allow_html=True)
    st.caption("Equipo Colectivo")
    st.markdown("• Posesión  \n• PPDA  \n• Passing Ratio  \n• xGD_pg")
    st.markdown("</div>", unsafe_allow_html=True)
with colC:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:34px'>🔗</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#fbbf24; font-weight:800'>Nivel Meta</h3>", unsafe_allow_html=True)
    st.caption("Interacciones")
    st.markdown("• Posesión × xGD  \n• Intensity Diff  \n• Style Fit  \n• Pass Quality")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; font-weight:800'>Pipeline de Aprendizaje Jerárquico</h3>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; gap:18px; justify-content: center; flex-wrap: wrap; margin-top:10px;">
  <div style="text-align:center">
    <div style="width:60px;height:60px;border-radius:999px;background:linear-gradient(135deg,#10b981,#60a5fa);display:flex;align-items:center;justify-content:center;font-weight:800;">1</div>
    <div style="margin-top:6px;color:#cbd5e1">Datos Raw</div>
  </div>
  <div style="font-size:24px;color:#34d399;align-self:center">→</div>
  <div style="text-align:center">
    <div style="width:60px;height:60px;border-radius:999px;background:linear-gradient(135deg,#10b981,#60a5fa);display:flex;align-items:center;justify-content:center;font-weight:800;">2</div>
    <div style="margin-top:6px;color:#cbd5e1">Feature Engineering</div>
  </div>
  <div style="font-size:24px;color:#34d399;align-self:center">→</div>
  <div style="text-align:center">
    <div style="width:60px;height:60px;border-radius:999px;background:linear-gradient(135deg,#10b981,#60a5fa);display:flex;align-items:center;justify-content:center;font-weight:800;">3</div>
    <div style="margin-top:6px;color:#cbd5e1">CatBoost Ensemble</div>
  </div>
  <div style="font-size:24px;color:#34d399;align-self:center">→</div>
  <div style="text-align:center">
    <div style="width:60px;height:60px;border-radius:999px;background:linear-gradient(135deg,#10b981,#60a5fa);display:flex;align-items:center;justify-content:center;font-weight:800;">4</div>
    <div style="margin-top:6px;color:#cbd5e1">Huber Meta-Model</div>
  </div>
  <div style="font-size:24px;color:#34d399;align-self:center">→</div>
  <div style="text-align:center">
    <div style="width:60px;height:60px;border-radius:999px;background:linear-gradient(135deg,#10b981,#60a5fa);display:flex;align-items:center;justify-content:center;font-weight:800;">5</div>
    <div style="margin-top:6px;color:#cbd5e1">OBV Predicho</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- RESULTADOS Y VALIDACIÓN ---
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<h2 class='h2-grad' style='font-size:clamp(26px,3.6vw,44px)'>Resultados y Validación</h2>", unsafe_allow_html=True)
st.markdown("<p class='lead'>El modelo alcanza un desempeño predictivo sólido, explicando ≈45% de la varianza del OBV futuro (holdout temporal).</p>", unsafe_allow_html=True)

cL, cR = st.columns(2)
with cL:
    st.markdown("<div class='card'><h3>Métricas de desempeño</h3>", unsafe_allow_html=True)
    for item in perf_metrics:
        pct = (item["value"]/item["max"])*100
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;color:#cbd5e1;margin-top:8px">
          <span>{item['metric']}</span><span style="color:#34d399;font-weight:800">{item['value']:.3f}</span>
        </div>
        <div style="width:100%;height:12px;background:#1f2937;border-radius:999px;overflow:hidden;margin-top:4px">
          <div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,#10b981,#60a5fa)"></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with cR:
    st.markdown("<div class='card'><h3>Hallazgo central</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#e5e7eb;font-size:1.05rem">
    “El valor en el fútbol no es una propiedad individual, sino una emergencia estadística del sistema.”
    </p>
    <p style="color:#cbd5e1">La posesión y las \\
    interacciones contextuales (p.ej. Posesión×xGD) superan consistentemente
    la relevancia marginal de métricas exclusivamente individuales.</p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3>Importancia de Variables (Top 8)</h3>", unsafe_allow_html=True)
fig_bar = px.bar(
    feature_importance.sort_values("value", ascending=True),
    x="value", y="name", orientation="h",
    color="category",
    color_discrete_map={"micro":"#3b82f6","meso":"#10b981","meta":"#f59e0b"},
    labels={"value":"Importancia","name":"Variable"},
    height=430
)
fig_bar.update_layout(margin=dict(l=10,r=10,t=20,b=10), showlegend=True)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("""
<div style="display:flex;gap:16px;justify-content:center;margin-top:8px;color:#cbd5e1">
  <span><span class="legend-dot" style="background:#3b82f6"></span>Micro</span>
  <span><span class="legend-dot" style="background:#10b981"></span>Meso</span>
  <span><span class="legend-dot" style="background:#f59e0b"></span>Meta</span>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- METODOLOGÍA ---
st.markdown("<div class='section alt'>", unsafe_allow_html=True)
st.markdown("<h2 class='h2-grad' style='font-size:clamp(26px,3.6vw,44px)'>Metodología e Innovación</h2>", unsafe_allow_html=True)
m1,m2 = st.columns(2)
with m1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### CatBoost Ensemble")
    st.write("Tres semillas (13, 42, 777), 3000 iteraciones, depth=6, LR=0.03, L2=20. Captura no-linealidades y reduce varianza.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Validación temporal")
    st.write("GroupKFold por temporada (4 años → 3 train / 1 holdout). Evita leakage y garantiza independencia temporal.")
    st.markdown("</div>", unsafe_allow_html=True)

with m2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Huber Meta-Regression")
    st.write("Meta-modelo robusto que calibra outliers sin perder sensibilidad estructural (stacking).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Feature Engineering")
    st.write("Interacciones sintéticas de 2º orden (p.ej. Posesión×xGD, Intensity Diff, Style Fit, Pass Quality).")
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- AUTORES / FOOTER ---
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<h2 class='h2-grad' style='font-size:clamp(26px,3.6vw,44px)'>Autores</h2>", unsafe_allow_html=True)
a1, a2 = st.columns(2)
with a1:
    st.markdown("<div class='card' style='text-align:center'>", unsafe_allow_html=True)
    st.markdown("<div style='width:84px;height:84px;border-radius:999px;margin:0 auto 10px auto;background:linear-gradient(135deg,#10b981,#60a5fa);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:26px;color:#0b1220;'>CMM</div>", unsafe_allow_html=True)
    st.markdown("**Cuauhtémoc Maya Maldonado**  \nLicenciatura en Economía, ITAM")
    st.markdown("</div>", unsafe_allow_html=True)
with a2:
    st.markdown("<div class='card' style='text-align:center'>", unsafe_allow_html=True)
    st.markdown("<div style='width:84px;height:84px;border-radius:999px;margin:0 auto 10px auto;background:linear-gradient(135deg,#10b981,#60a5fa);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:26px;color:#0b1220;'>AUC</div>", unsafe_allow_html=True)
    st.markdown("**Alexis Uriel Cano Bernabé**  \nLicenciatura en Economía, ITAM")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class='card' style='text-align:center; margin-top:18px;'>
  <p style='font-weight:700'>Instituto Tecnológico Autónomo de México (ITAM)</p>
  <p style='color:#cbd5e1'>Departamento Académico de Economía</p>
  <p style='color:#94a3b8;margin-top:6px;'>2025</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='footer'>© 2025 ITAM - Inteligencia Táctica Cuantitativa · Datos: Hudl StatsBomb 360 | Liga MX 2021–2025</div>", unsafe_allow_html=True)