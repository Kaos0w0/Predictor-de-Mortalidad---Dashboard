"""
Tablero de riesgo de mortalidad intrahospitalaria en adultos mayores (≥ 75 años).

Usa el consenso reducido que exporta Entrenamiento_corregido.ipynb (secciones 10–12):
    salida_entrenamiento/modelo_tablero.joblib   modelos calibrados, variables y umbral
    salida_entrenamiento/ficha_tablero.json      formulario, rangos, desempeño y paciente de referencia

Ejecutar desde la carpeta del proyecto, en el MISMO entorno de Python donde se entrenó el modelo:
    streamlit run app.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

try:
    import shap
except ImportError:  # la explicación local es opcional
    shap = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DIR_MODELO = BASE_DIR / "salida_entrenamiento"
NO_DISPONIBLE = "No disponible"
COLOR_SUBE, COLOR_BAJA, TINTA = "#e34948", "#2a78d6", "#52514e"

SECCIONES = [
    ("Valoración geriátrica", {"mental_cognitivo", "nutricion", "barthel_actual", "frail", "lawton", "actividades_basicas"}),
    ("Signos vitales al ingreso", {"signos_vitales"}),
    ("Laboratorio (primeras 48 h)", {"laboratorio"}),
    ("Diagnósticos", {"diagnosticos", "general"}),
    ("Antecedentes", {"antecedentes"}),
    ("Datos sociodemográficos", {"edad", "sociodemografica"}),
]

st.set_page_config(page_title="Riesgo de mortalidad intrahospitalaria", page_icon="🏥", layout="wide")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    ficha = json.loads((DIR_MODELO / "ficha_tablero.json").read_text(encoding="utf-8"))
    artefacto = joblib.load(DIR_MODELO / "modelo_tablero.joblib")
    assert artefacto["variables"] == [v["nombre"] for v in ficha["variables"]], "La ficha no corresponde al modelo"
    return artefacto, ficha


try:
    ARTEFACTO, FICHA = cargar_modelo()
except FileNotFoundError as e:
    st.error(f"No se encontró el modelo: {e}. Ejecuta Entrenamiento_corregido.ipynb hasta la sección 12.")
    st.stop()

VARIABLES = FICHA["variables"]
NOMBRES = [v["nombre"] for v in VARIABLES]
MIEMBROS = ARTEFACTO["miembros"]
UMBRAL = float(ARTEFACTO["umbral"])
MORTALIDAD_BASE = float(FICHA["mortalidad_base"])

version_entreno = FICHA["versiones"].get("scikit-learn")
if version_entreno and version_entreno != sklearn.__version__:
    st.warning(f"El modelo se entrenó con scikit-learn {version_entreno} y este entorno tiene {sklearn.__version__}. "
               "Ejecuta el tablero en el mismo entorno del entrenamiento para evitar resultados incorrectos.")


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def a_dataframe(registros):
    """Lista de diccionarios {variable: valor} → DataFrame con los tipos que espera el pipeline."""
    df = pd.DataFrame(registros, columns=NOMBRES)
    for v in VARIABLES:
        if v["tipo"] == "categorica":
            df[v["nombre"]] = df[v["nombre"]].astype(object).where(df[v["nombre"]].notna(), np.nan)
        else:
            df[v["nombre"]] = pd.to_numeric(df[v["nombre"]], errors="coerce").astype(float)
    return df


def probabilidades(df):
    """Probabilidad calibrada de cada modelo y del consenso (promedio simple)."""
    por_modelo = {nombre: m.predict_proba(df[NOMBRES])[:, 1] for nombre, m in MIEMBROS.items()}
    return por_modelo, np.mean(list(por_modelo.values()), axis=0)


# Codificación numérica para el explicador (las categorías pasan a índices)
_CODIGOS = {v["nombre"]: {c: i for i, c in enumerate(v["categorias"])} for v in VARIABLES if v["tipo"] == "categorica"}


def codificar(registro):
    fila = []
    for v in VARIABLES:
        x = registro.get(v["nombre"], np.nan)
        if v["tipo"] == "categorica":
            fila.append(np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else _CODIGOS[v["nombre"]][x])
        else:
            fila.append(np.nan if x is None else float(x))
    return np.array(fila, dtype=float)


def decodificar(Z):
    registros = []
    for fila in np.atleast_2d(Z):
        r = {}
        for valor, v in zip(fila, VARIABLES):
            if v["tipo"] == "categorica":
                r[v["nombre"]] = np.nan if np.isnan(valor) else v["categorias"][int(round(valor))]
            else:
                r[v["nombre"]] = valor
        registros.append(r)
    return a_dataframe(registros)


def explicar(registro):
    """Valores de Shapley del consenso frente al paciente de referencia (mediana/moda de la cohorte),
    en puntos porcentuales de probabilidad."""
    referencia = codificar({v["nombre"]: v["referencia"] for v in VARIABLES})
    explicador = shap.PermutationExplainer(lambda Z: probabilidades(decodificar(Z))[1],
                                           shap.maskers.Independent(referencia.reshape(1, -1), max_samples=1))
    resultado = explicador(codificar(registro).reshape(1, -1), max_evals=max(500, 2 * len(VARIABLES) + 1),
                           silent=True)
    return float(np.ravel(resultado.base_values)[0]), resultado.values[0]


def texto_valor(v, x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "no disponible"
    if v["tipo"] == "binaria":
        return "sí" if x == 1 else "no"
    if v["tipo"] == "numerica":
        return f"{x:.0f}" if v.get("entera") else f"{x:.2f}".rstrip("0").rstrip(".")
    return str(x)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
p = FICHA["prueba"]
ic = FICHA.get("prueba_ic95", {})
with st.sidebar:
    st.header("Sobre el modelo")
    st.markdown(f"""
**Población:** {FICHA['poblacion']}.
**Desenlace:** {FICHA['desenlace'].lower()}.

**Modelo:** consenso de {', '.join(MIEMBROS)}; {FICHA['combinacion']}.

**Variables:** {len(VARIABLES)}, elegidas con un {FICHA['seleccion_variables']}.

**Desempeño en prueba** ({FICHA['datos']['prueba']} hospitalizaciones, {FICHA['datos']['fallecidos_prueba']} fallecidos):
- AUROC: {ic.get('AUROC', f"{p['AUROC']:.3f}")}
- Sensibilidad: {p['Sensibilidad'] * 100:.0f} % · Especificidad: {p['Especificidad'] * 100:.0f} %
- VPP: {p['VPP'] * 100:.0f} % · VPN: {p['VPN'] * 100:.0f} %
- Umbral de alerta: {UMBRAL * 100:.1f} % (índice de Youden)
""")
    st.warning("**Herramienta de apoyo con fines de investigación.** Validada solo internamente, en un centro. "
               "No reemplaza el juicio clínico ni debe usarse sola para decidir el tratamiento.")
    st.caption("Desarrollado por: **Brayan Andres Sanchez Lozano**")

# ---------------------------------------------------------------------------
# Formulario
# ---------------------------------------------------------------------------
st.title("Riesgo de mortalidad intrahospitalaria")
st.markdown("Adultos ≥ 75 años hospitalizados por el servicio de geriatría · datos de las primeras 48 horas")
st.caption("Los campos marcados con * son obligatorios. En los demás puede elegir *No disponible*: "
           "el modelo usará el valor típico de la cohorte, y la estimación será menos precisa.")

valores = {}
with st.form("paciente"):
    for titulo, familias in SECCIONES:
        grupo = [v for v in VARIABLES if v["familia"] in familias]
        if not grupo:
            continue
        st.subheader(titulo)
        columnas = st.columns(3)
        for i, v in enumerate(grupo):
            etiqueta = v["etiqueta"] + (" *" if v["obligatoria"] else "")
            ayuda = v.get("ayuda") or None
            if v.get("temporalidad_incierta"):
                ayuda = ((ayuda + " ") if ayuda else "") + "Registrar solo si ya se conoce en las primeras 48 horas."
            with columnas[i % 3]:
                if v["tipo"] == "numerica":
                    entero = v.get("entera", False)
                    paso = 1.0 if entero else (0.01 if v["max"] <= 20 else 0.1)
                    valores[v["nombre"]] = st.number_input(
                        etiqueta, min_value=v["min"], max_value=v["max"], value=None, step=paso,
                        format="%.0f" if entero else ("%.2f" if paso == 0.01 else "%.1f"),
                        placeholder=f"{texto_valor(v, v['min'])} – {texto_valor(v, v['max'])}",
                        help=ayuda, key=v["nombre"])
                elif v["tipo"] == "binaria":
                    opciones = ["No", "Sí"] + ([] if v["obligatoria"] else [NO_DISPONIBLE])
                    eleccion = st.selectbox(etiqueta, opciones, index=0, help=ayuda, key=v["nombre"])
                    valores[v["nombre"]] = {"No": 0.0, "Sí": 1.0}.get(eleccion, np.nan)
                else:
                    opciones = v["categorias"] + ([] if v["obligatoria"] else [NO_DISPONIBLE])
                    eleccion = st.selectbox(etiqueta, opciones, index=None, placeholder="Seleccione…",
                                            help=ayuda, key=v["nombre"])
                    valores[v["nombre"]] = np.nan if eleccion in (None, NO_DISPONIBLE) else eleccion
    enviado = st.form_submit_button("Calcular riesgo", type="primary")

if not enviado:
    st.stop()

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
faltan_obligatorias = [v["etiqueta"] for v in VARIABLES if v["obligatoria"]
                       and (valores[v["nombre"]] is None or (isinstance(valores[v["nombre"]], float)
                                                             and np.isnan(valores[v["nombre"]])))]
if faltan_obligatorias:
    st.error("Faltan datos obligatorios: " + ", ".join(faltan_obligatorias) + ".")
    st.stop()

registro = {k: (np.nan if x is None else x) for k, x in valores.items()}
no_disponibles = [v["etiqueta"] for v in VARIABLES if isinstance(registro[v["nombre"]], float) and np.isnan(registro[v["nombre"]])]
por_modelo, consenso = probabilidades(a_dataframe([registro]))
riesgo = float(consenso[0])
alerta = riesgo >= UMBRAL

st.divider()
col_riesgo, col_contexto = st.columns([1, 2])
with col_riesgo:
    st.metric("Riesgo de mortalidad intrahospitalaria", f"{riesgo * 100:.1f} %",
              delta=f"{riesgo / MORTALIDAD_BASE:.1f} veces el promedio de la cohorte ({MORTALIDAD_BASE * 100:.1f} %)",
              delta_color="off")
    if alerta:
        st.error(f"**Por encima del umbral de alerta** ({UMBRAL * 100:.1f} %).")
    else:
        st.success(f"**Por debajo del umbral de alerta** ({UMBRAL * 100:.1f} %).")

with col_contexto:
    banda = next((b for b in FICHA["bandas_riesgo_prueba"] if b["desde"] <= riesgo <= b["hasta"]), None)
    if banda and banda["n"] >= 20:
        st.markdown(f"En el conjunto de prueba, de **{banda['n']}** pacientes con riesgo estimado entre "
                    f"{banda['desde'] * 100:.0f} % y {banda['hasta'] * 100:.0f} %, falleció el "
                    f"**{banda['mortalidad_observada'] * 100:.0f} %**.")
    st.markdown("**Estimación de cada modelo del consenso**")
    st.dataframe(pd.DataFrame({"Modelo": list(por_modelo),
                               "Probabilidad": [f"{float(v[0]) * 100:.1f} %" for v in por_modelo.values()]}),
                 hide_index=True, width="stretch")
    rango = max(float(v[0]) for v in por_modelo.values()) - min(float(v[0]) for v in por_modelo.values())
    if rango > 0.10:
        st.info(f"Los modelos difieren en {rango * 100:.0f} puntos porcentuales: interprete la estimación con cautela.")
    if no_disponibles:
        st.warning("Se usaron valores típicos de la cohorte para: " + ", ".join(no_disponibles) + ".")

st.subheader("¿Qué explica esta estimación?")
if shap is None:
    st.info("Instale `shap` para ver la explicación de cada predicción.")
else:
    with st.spinner("Calculando la contribución de cada variable…"):
        base, contribuciones = explicar(registro)
    tabla = pd.DataFrame({"variable": [v["etiqueta"] for v in VARIABLES],
                          "valor": [texto_valor(v, registro[v["nombre"]]) for v in VARIABLES],
                          "aporte": contribuciones * 100})
    tabla = tabla.reindex(tabla["aporte"].abs().sort_values(ascending=False).index).head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(tabla) + 1.2))
    ax.barh([f"{r.variable} = {r.valor}" for r in tabla.itertuples()], tabla["aporte"],
            color=np.where(tabla["aporte"] > 0, COLOR_SUBE, COLOR_BAJA), height=0.6)
    ax.axvline(0, color="#9a9994", linewidth=1)
    for i, a in enumerate(tabla["aporte"]):
        ax.text(a, i, f" {a:+.1f} ", va="center", ha="left" if a > 0 else "right", fontsize=8, color=TINTA)
    ax.set_xlabel("Puntos porcentuales que suma (rojo) o resta (azul) al riesgo")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TINTA)
    ax.margins(x=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption(f"Comparación con un paciente de referencia de la cohorte (valores típicos), cuyo riesgo estimado es "
               f"{base * 100:.1f} %. Las barras suman la diferencia entre ese riesgo y el de este paciente "
               f"({riesgo * 100:.1f} %). Son asociaciones aprendidas de los datos, no efectos causales.")
