# -*- coding: utf-8 -*-
"""
Mi plan 70.3 — Demostrador Streamlit del TFM
=============================================
Sistema de Análisis Fisiológico y Planificación Adaptativa para Triatlón 70.3.

Lee los resultados persistidos por el notebook (salidas/triatlon.db y
salidas/parquet/) y reproduce el motor adaptativo: perfil del atleta,
macrociclo con fecha de competencia, validación de ritmos con el historial,
plan semanal detallado, botón «Actualizar hoy», glosario pedagógico y plan
de competencia km a km.

Ejecución:  streamlit run app.py
Requisitos: streamlit, pandas, numpy, matplotlib (pip install -r requirements)
"""

import os
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from stravalib.client import Client as StravaClient
    STRAVA_AVAILABLE = True
except ImportError:
    STRAVA_AVAILABLE = False

try:
    from garminconnect import Garmin
    GARMIN_AVAILABLE = True
except ImportError:
    GARMIN_AVAILABLE = False

# ----------------------------------------------------------------------------
# Configuración general y estilos
# ----------------------------------------------------------------------------
st.set_page_config(page_title='Mi plan 70.3', page_icon='🔴', layout='centered')

st.markdown("""
<style>
  /* Ocultar el "andamiaje" de Streamlit (sin tocar el header: en móvil contiene
     el botón para abrir la barra lateral). toolbarMode="minimal" limpia el resto. */
  #MainMenu, footer {visibility: hidden;}
  .block-container {padding-top: 1.2rem; padding-bottom: 2rem;
                    max-width: 820px;}
  /* Paleta unificada (azul #1d6fa5 + acento cálido), coherente con el tema */
  .chip {display:inline-block; background:#eef3f7; border:1px solid #d8e3ec;
         border-radius:12px; padding:10px 16px; margin:4px; text-align:center;}
  .chip .lab {color:#1d6fa5; font-weight:700; font-size:0.8rem;}
  .chip .val {color:#16202b; font-weight:700; font-size:1.0rem;}
  .chip.off {background:#f3f4f6; border-color:#dfe3e8;}
  .chip.off .lab, .chip.off .val {color:#9aa3ad; text-decoration:line-through;}
  .badge {display:inline-block; border-radius:14px; padding:4px 14px;
          font-weight:700; font-size:0.85rem;}
  div[data-testid="stMetric"] {background:#ffffff; border:1px solid #e2e8ee;
          border-radius:12px; padding:14px;}
  /* Botones más "app": esquinas suaves y peso */
  .stButton>button, .stDownloadButton>button {border-radius:10px; font-weight:700;}
  /* Imágenes (gráficos) que se adaptan al ancho del móvil */
  div[data-testid="stImage"] img {max-width:100%; height:auto;}
</style>
""", unsafe_allow_html=True)

DB_PATH = 'salidas/triatlon.db'
PARQUET_DIR = 'salidas/parquet'

GARMIN_COLORS = {'Z1': '#9aa0a6', 'Z2': '#3b82f6', 'Z3': '#22a06b',
                 'Z4': '#f59e0b', 'Z5': '#ef4444'}

SEMANAS_MINIMAS = {'nuevo': 24, 'experimentado': 12}
CTL_OBJETIVO_70_3 = 70
SEMANA_ACTUAL_DEFAULT = 6
DIAS_ES = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# Strava OAuth (sincronización en vivo). Las credenciales van en .streamlit/secrets.toml
STRAVA_CLIENT_ID = st.secrets.get("strava_client_id", "")
STRAVA_CLIENT_SECRET = st.secrets.get("strava_client_secret", "")
# URL a la que Strava devuelve tras autorizar. En local es localhost; al desplegar
# en Streamlit Cloud, define `redirect_uri = "https://TU-APP.streamlit.app"` en Secrets.
STRAVA_REDIRECT_URI = st.secrets.get("redirect_uri", "http://localhost:8501")

# ----------------------------------------------------------------------------
# i18n: idioma de la interfaz (Español / English). El contenido dinámico
# (glosario, detalle de sesiones, explicaciones IA) permanece en español.
# ----------------------------------------------------------------------------
IDIOMAS = {'Español': 'es', 'English': 'en'}
idioma = 'es'  # se actualiza con el selector de la barra lateral

TR = {
    'sync_datos':   {'es': 'Sincronización de datos', 'en': 'Data source'},
    'de_donde':     {'es': '¿De dónde obtener datos?', 'en': 'Where to get data from?'},
    'opt_strava':   {'es': '📱 Strava (en vivo)', 'en': '📱 Strava (live)'},
    'opt_garmin':   {'es': '⌚ Garmin Connect', 'en': '⌚ Garmin Connect'},
    'tu_perfil':    {'es': 'Tu perfil', 'en': 'Your profile'},
    'nuevo':        {'es': 'Nuevo en triatlón', 'en': 'New to triathlon'},
    'experto':      {'es': 'Con experiencia', 'en': 'Experienced'},
    'sexo':         {'es': 'Sexo', 'en': 'Sex'},
    'hombre':       {'es': 'Hombre', 'en': 'Male'},
    'mujer':        {'es': 'Mujer', 'en': 'Female'},
    'tu_objetivo':  {'es': 'Tu objetivo', 'en': 'Your goal'},
    'fecha_inicio': {'es': 'Fecha de inicio del plan', 'en': 'Plan start date'},
    'reiniciar':    {'es': '🔄 Reiniciar', 'en': '🔄 Reset'},
    'tengo_fecha':  {'es': 'Ya tengo fecha de competencia', 'en': 'I have a race date'},
    'cuando_703':   {'es': '¿Cuándo es tu 70.3?', 'en': 'When is your 70.3?'},
    'hola':         {'es': '👋 ¡Hola, {n}! Bienvenido a tu plan de hoy.',
                     'en': '👋 Hi, {n}! Welcome to today’s plan.'},
    'h_hoy':        {'es': 'Cómo estás hoy', 'en': 'How you are today'},
    'h_forma':      {'es': 'Tu forma a lo largo de las semanas', 'en': 'Your fitness over the weeks'},
    'h_rumbo':      {'es': 'Tu rumbo hacia la carrera', 'en': 'Your road to race day'},
    'h_zonas':      {'es': 'Tus zonas, como las muestra tu reloj', 'en': 'Your zones, as your watch shows them'},
    'h_semana':     {'es': 'Tu semana, día a día', 'en': 'Your week, day by day'},
    'h_porque':     {'es': '¿Por qué este plan?', 'en': 'Why this plan?'},
    'h_progreso':   {'es': 'Tu progreso por disciplina', 'en': 'Your progress by discipline'},
    'm_fuente':     {'es': 'Fuente de datos', 'en': 'Data source'},
    'm_acts':       {'es': 'Actividades usadas', 'en': 'Activities used'},
    'm_modelos':    {'es': 'Modelos IA (XGBoost/KMeans)', 'en': 'AI models (XGBoost/KMeans)'},
}


def t(key, **kw):
    s = TR.get(key, {}).get(idioma) or TR.get(key, {}).get('es') or key
    return s.format(**kw) if kw else s

GLOSARIO = {
    'Body Battery': "Reserva de energía (0-100) que estima el reloj combinando HRV, estrés, sueño y actividad. Por debajo de 20-30, el cuerpo pide recuperación, no entrenamiento.",
    'HRV': "Variabilidad de la frecuencia cardiaca: los milisegundos que cambian entre latido y latido. Alta y estable = bien recuperado; si cae de golpe, fatiga, estrés o enfermedad.",
    'Estrés': "Índice 0-100 derivado de la HRV. Sostenido por encima de 70, el sistema nervioso está en alerta y asimila peor el entrenamiento.",
    'Sueño': "El sueño profundo es donde ocurre la reparación muscular y hormonal. Menos de ~1 h por noche de forma repetida frena la adaptación.",
    'Sleep Score': "Puntuación 0-100 que resume la CALIDAD del sueño (no las horas): combina sueño profundo y REM, despertares, duración y estabilidad de la FC. Excelente 90+, Buena 80-89, Regular 60-79, Mala <60. Puedes dormir muchas horas con un score bajo y amanecer cansado; por eso ajusta el entrenamiento aunque las horas parezcan suficientes.",
    'FC': "Frecuencia cardiaca, en pulsaciones por minuto (ppm). Es el cuentarrevoluciones del entrenamiento.",
    'CTL / ATL / TSB': "CTL = Fitness (forma acumulada de ~6 semanas, media exponencial de la carga). ATL = Fatiga (carga de los últimos ~7 días). TSB = Frescura = CTL − ATL. La referencia es 0: TSB positivo = descansado (fresco para competir); TSB negativo = cargado. Bandas orientativas: por encima de +25 demasiado fresco (riesgo de desentrenar); de +5 a +25 a punto para competir; de −10 a +5 en equilibrio/mantenimiento; de −30 a −10 carga productiva (estás construyendo forma, fatiga normal y buscada); por debajo de −30 sobrecarga (riesgo, toca aliviar).",
    'Zonas Z1-Z5': "Intensidades por FC: Z1 muy suave (recuperar), Z2 aeróbico (la base, puedes hablar), Z3 medio (te cuesta hablar), Z4 umbral (frases cortas), Z5 máximo.",
    'D+': "Desnivel positivo acumulado: metros totales de subida de la ruta. Más D+ = más carga muscular y cardiaca al mismo ritmo.",
    'brick': "Bici seguida inmediatamente de carrera. Entrena la transición T2: los primeros minutos con 'piernas de cemento' que sentirás el día del 70.3.",
    'series': "Repeticiones de un esfuerzo con recuperación entre ellas (ej. 6x800 m). Permiten acumular minutos a intensidades que de seguido no aguantarías.",
    'FTP': "Potencia máxima sostenible ~1 h en bici (vatios). Con potenciómetro, las series se hacen al 90-100% del FTP; sin él, por FC y velocidad.",
    'cadencia': "Pedaladas (rpm) o zancadas por minuto. En bici, 85-95 rpm protege las piernas para la carrera posterior; en cuestas, 70-75.",
    'taper': "Reducción del volumen (~50%) las 2-3 semanas previas a la carrera manteniendo toques de intensidad: la fatiga se va, la forma se queda.",
    'drafting': "Nadar a los pies de otro: legal en la natación y ahorra ~5% de energía.",
}

# ----------------------------------------------------------------------------
# Utilidades de formato
# ----------------------------------------------------------------------------
ESTADO_PLAN_PATH = '.plan_estado.json'


def _estado_plan_load():
    """Carga el dict {user_key: fecha_inicio_iso} persistido (o {})."""
    try:
        import json
        with open(ESTADO_PLAN_PATH, encoding='utf-8') as f:
            data = json.load(f)
        # Compatibilidad con el formato antiguo (clave única 'fecha_inicio')
        if 'fecha_inicio' in data and isinstance(data['fecha_inicio'], str):
            return {'demo': data['fecha_inicio']}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def leer_inicio_plan(user_key='demo'):
    """Fecha de inicio persistida PARA ESE USUARIO (o None si no hay)."""
    d = _estado_plan_load().get(user_key)
    try:
        return date.fromisoformat(d) if d else None
    except Exception:
        return None


def guardar_inicio_plan(d, user_key='demo'):
    """Fija la fecha de inicio del plan de ese usuario. Es el ANCLA del plan:
    no cambia sola al pasar los días; solo se reescribe al reiniciar el plan."""
    try:
        import json
        data = _estado_plan_load()
        data[user_key] = pd.Timestamp(d).date().isoformat()
        with open(ESTADO_PLAN_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        pass


def pace_run(v_kmh):
    if pd.isna(v_kmh) or v_kmh <= 0:
        return 'N/D'
    m = 60.0 / v_kmh
    s = int(round((m - int(m)) * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{int(m)}:{s:02d} min/km"


def pace_swim(v_kmh):
    if pd.isna(v_kmh) or v_kmh <= 0:
        return 'N/D'
    s = 360.0 / v_kmh
    return f"{int(s // 60)}:{int(s % 60):02d}/100 m"


def t2min(txt):
    p = [int(v) for v in str(txt).split(':')]
    return p[0] * 60 + p[1]


def min2t(m):
    h, mm = divmod(int(round(m)), 60)
    return f"{h}:{mm:02d}"


# ----------------------------------------------------------------------------
# Integración con Gemini API para explicaciones pedagógicas dinámicas
# ----------------------------------------------------------------------------
@st.cache_resource
def inicializar_gemini():
    """Inicializa Gemini API con clave de st.secrets."""
    if not GEMINI_AVAILABLE:
        return None
    try:
        api_key = st.secrets.get("gemini_api_key", os.getenv("GEMINI_API_KEY", ""))
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        # 'gemini-pro' quedó descontinuado; se usa el modelo vigente (como en el v6),
        # con respaldo a otros nombres si la cuenta tiene distinta disponibilidad.
        for nombre in ('gemini-1.5-flash', 'gemini-1.5-flash-latest',
                       'gemini-1.5-pro', 'gemini-2.0-flash'):
            try:
                return genai.GenerativeModel(nombre)
            except Exception:
                continue
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def generar_explicacion_dia(atleta, dia_semana, decision, motivos, zonas, fase,
                            sesion_planificada=''):
    """Explica en vivo cómo queda la sesión de HOY tras el recálculo adaptativo."""
    model = inicializar_gemini()
    if not model:
        return None
    try:
        prompt = f"""Eres un entrenador personal de triatlón. En español, 2-3 frases,
        tono motivador y pedagógico, tuteando a {atleta}.

        Hoy es {dia_semana}, fase del plan: {fase}.
        Sesión que tenía planificada hoy: {sesion_planificada or 'no especificada'}.
        Métricas de esta mañana: {', '.join(motivos)}.
        El sistema recomienda: "{decision}".

        Dile EXACTAMENTE qué hacer hoy con esa sesión:
        - Si es DESCANSO HOY: justifica el descanso y qué hacer en su lugar (dormir, hidratar).
        - Si es SUAVIZAR HOY: indica reducir volumen y mantener todo en Z1-Z2
          ({zonas['hr']['Z1'][0]}-{zonas['hr']['Z2'][1]} ppm), sin series ni calidad.
        - Si es MANTENER: confirma que ejecute la sesión planificada tal cual.
        Conecta la lógica con sus métricas; no solo el qué, también el porqué."""
        response = model.generate_content(prompt)
        return response.text if response else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def generar_explicacion_plan(atleta, perfil_tipo, fase, semanas_restantes, ctl_actual):
    """Genera justificación dinámica de por qué este plan es el ideal."""
    model = inicializar_gemini()
    if not model:
        return None

    try:
        prompt = f"""Eres un entrenador de triatlón. Escribe una justificación breve (3-4 frases) de por qué
        el plan actual es ideal para {atleta}, un triatleta {perfil_tipo} que está en fase {fase}
        con {semanas_restantes} semanas para la carrera y forma actual (CTL) de {ctl_actual:.0f}.

        Sé específico y pedagógico: explica qué mejorará y por qué el ritmo es el adecuado para esta fase."""

        response = model.generate_content(prompt)
        return response.text if response else None
    except Exception as e:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def generar_frase_dia(nombre, fase, semilla, fallback):
    """Frase motivadora del día, personalizada con Gemini (una por día).
    Si Gemini no está disponible, usa la frase de respaldo (lista curada)."""
    model = inicializar_gemini()
    if not model:
        return fallback
    try:
        prompt = (f"Genera UNA sola frase motivadora, original y potente (máximo 14 palabras), "
                  f"en español, para {nombre}, un triatleta que hoy entrena en fase {fase} de su "
                  f"plan hacia un 70.3. Tono inspirador, cercano y memorable —la frase que se "
                  f"llevará en la cabeza hoy—. Devuelve SOLO la frase, sin comillas ni explicación.")
        r = model.generate_content(prompt)
        s = (r.text or '').strip().strip('"').strip()
        return s if 3 < len(s) < 160 else fallback
    except Exception:
        return fallback


# ----------------------------------------------------------------------------
# Carga de datos persistidos (con respaldo de demostración)
# ----------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    datos = {}
    if os.path.exists(DB_PATH):
        con = sqlite3.connect(DB_PATH)
        for tabla in ('perfiles', 'planes_semanales', 'sesiones_plan',
                      'predicciones_ciclismo', 'predicciones_carrera'):
            try:
                datos[tabla] = pd.read_sql(f'SELECT * FROM {tabla}', con)
            except Exception:
                datos[tabla] = pd.DataFrame()
        con.close()
    else:
        datos = {t: pd.DataFrame() for t in ('perfiles', 'planes_semanales',
                                             'sesiones_plan', 'predicciones_ciclismo',
                                             'predicciones_carrera')}
    fp = os.path.join(PARQUET_DIR, 'fitness_fatiga.parquet')
    datos['fitness'] = pd.read_parquet(fp) if os.path.exists(fp) else pd.DataFrame()
    fa = os.path.join(PARQUET_DIR, 'actividades.parquet')
    datos['actividades'] = pd.read_parquet(fa) if os.path.exists(fa) else pd.DataFrame()
    return datos


def datos_demo():
    """Atleta sintético para explorar la app sin ejecutar el notebook."""
    perfil = pd.DataFrame([{'athlete': 'Demo', 'resting_hr': 52,
                            'pace_running_kmh': 10.5, 'pace_cycling_kmh': 24,
                            'pace_swimming_kmh': 2.8}])
    fechas = pd.date_range(end=pd.Timestamp.today(), periods=120)
    rng = np.random.default_rng(7)
    fitness = pd.DataFrame({'athlete': 'Demo', 'date': fechas,
                            'CTL': np.linspace(35, 50, 120) + rng.normal(0, 1, 120),
                            'ATL': np.linspace(35, 55, 120) + rng.normal(0, 3, 120)})
    fitness['TSB'] = fitness['CTL'] - fitness['ATL']
    # Actividades con FECHA y las 3 disciplinas, con mejora gradual del ritmo
    # a lo largo de ~16 semanas (para ilustrar el progreso en el gráfico).
    filas = []
    hoy = pd.Timestamp.today().normalize()
    base = {'RUNNING': (9.6, 11.2, 0.35, 162, 186),
            'CYCLING': (23.0, 28.0, 0.9, 150, 178),
            'SWIMMING': (2.55, 2.95, 0.08, 150, 170)}  # (v_ini, v_fin, ruido, hr, hrmax)
    for sport, (v0, v1, ns, hr, hrmax) in base.items():
        n = 16 if sport != 'SWIMMING' else 14
        for k in range(n):
            frac = k / max(n - 1, 1)
            d_ = hoy - pd.Timedelta(days=int((1 - frac) * 112))   # ~16 semanas atrás → hoy
            filas.append({'athlete': 'Demo', 'sport': sport, 'date': d_,
                          'avg_speed_kmh': (v0 + (v1 - v0) * frac) + rng.normal(0, ns),
                          'avg_hr': rng.normal(hr, 5), 'max_hr': rng.normal(hrmax, 4),
                          'moving_time': rng.normal(55, 12)})
    acts = pd.DataFrame(filas)
    return perfil, fitness, acts


def acts_demo_progresivo(per, sexo='M', fecha_inicio=None):
    """Actividades de demostración que REFLEJAN las semanas ya transcurridas del
    plan: a más semanas completadas, más actividades acumuladas y mayor mejora.
    Las fechas se generan DESDE EL INICIO DEL PLAN hacia adelante (semana 1, 2, 3…),
    para que coincidan con el filtro "solo desde el inicio" del gráfico de progreso.
    Así evoluciona al avanzar (o simular) las semanas, igual que los datos reales."""
    sem_act = int(per.get('semana_actual', 1))
    sem_tot = max(int(per.get('semanas_totales', 24)), 1)
    # Semanas transcurridas desde el inicio del plan: si empezaste hoy, apenas hay
    # datos; el historial crece semana a semana (con datos reales pasaría igual).
    n_weeks = min(max(sem_act, 1), 24)
    sx = 1.0 if sexo == 'M' else 0.93
    if fecha_inicio is not None:
        ini = pd.Timestamp(fecha_inicio).normalize()
    else:
        ini = pd.Timestamp.today().normalize() - pd.Timedelta(days=(n_weeks - 1) * 7)
    rng = np.random.default_rng(7)
    base = {'RUNNING': (9.4, 11.6, 0.30, 162, 186),
            'CYCLING': (22.0, 29.5, 0.8, 150, 178),
            'SWIMMING': (2.5, 3.05, 0.07, 150, 170)}   # (v_ini, v_fin, ruido, hr, hrmax)
    filas = []
    for sport, (v0, v1, ns, hr, hrmax) in base.items():
        for w in range(n_weeks):
            frac = w / max(sem_tot - 1, 1)            # mejora proporcional al avance del plan
            d_sem = ini + pd.Timedelta(days=w * 7)    # semana w DESDE el inicio
            for _ in range(2):                        # ~2 sesiones/semana por disciplina
                filas.append({'athlete': 'Demo', 'sport': sport,
                              'date': d_sem + pd.Timedelta(days=int(rng.integers(0, 5))),
                              'avg_speed_kmh': (v0 + (v1 - v0) * frac) * sx + rng.normal(0, ns),
                              'avg_hr': rng.normal(hr, 5), 'max_hr': rng.normal(hrmax, 4),
                              'moving_time': rng.normal(55, 12)})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Integración Strava OAuth (sincronización de actividades en vivo)
# ----------------------------------------------------------------------------
@st.cache_resource
def init_strava_client():
    """Cliente Strava compartido (sin token hasta autenticar)."""
    if STRAVA_AVAILABLE and STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET:
        return StravaClient()
    return None


def _strava_client_id_int():
    """Strava exige client_id numérico (en secrets puede venir como texto)."""
    try:
        return int(STRAVA_CLIENT_ID)
    except (ValueError, TypeError):
        return STRAVA_CLIENT_ID


def get_strava_auth_url():
    """URL de autorización de Strava."""
    client = init_strava_client()
    if not client:
        return None
    return client.authorization_url(
        client_id=_strava_client_id_int(),
        redirect_uri=STRAVA_REDIRECT_URI,
        scope=['activity:read_all'],
        approval_prompt='force',  # muestra siempre la pantalla de autorización
    )


def desconectar_strava():
    """Borra la sesión de Strava de la app para poder cambiar de cuenta."""
    for k in ('strava_token', 'strava_refresh_token', 'strava_expires_at',
              'actividades_strava'):
        st.session_state.pop(k, None)
    try:
        fetch_strava_athlete_name.clear()
        fetch_strava_activities.clear()
        init_strava_client.clear()
    except Exception:
        pass


def handle_strava_callback():
    """Captura el ?code=... que devuelve Strava y lo canjea por el token.

    Sin esto, la app vuelve a la pantalla inicial sin conectarse.
    """
    if 'strava_token' in st.session_state:
        return
    if not (STRAVA_AVAILABLE and STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET):
        return
    code = st.query_params.get('code')
    if not code:
        return
    client = init_strava_client()
    if not client:
        return
    try:
        token = client.exchange_code_for_token(
            client_id=_strava_client_id_int(),
            client_secret=STRAVA_CLIENT_SECRET,
            code=code,
        )
        if isinstance(token, dict):   # stravalib 2.x devuelve AccessInfo (dict)
            st.session_state['strava_token'] = token.get('access_token')
            st.session_state['strava_refresh_token'] = token.get('refresh_token')
            st.session_state['strava_expires_at'] = token.get('expires_at')
        else:
            st.session_state['strava_token'] = token
        # Deja la app en la fuente Strava y sincroniza ya, para que no se quede en Local
        st.session_state['fuente_sel'] = 'strava'
        try:
            st.session_state['actividades_strava'] = fetch_strava_activities(
                st.session_state['strava_token'], days=90)
        except Exception:
            pass
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error intercambiando código de Strava: {e}")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_strava_activities(access_token, days=90):
    """Descarga actividades recientes de Strava y las normaliza al esquema
    que usa el motor (sport, avg_hr, max_hr, avg_speed_kmh, distance, date).
    """
    try:
        client = init_strava_client()
        if not client:
            return None
        client.access_token = access_token
        # stravalib 2.x exige un datetime en `after` (hace assert isinstance datetime);
        # pasar un epoch int o un date provoca el error de sincronización.
        activities = list(client.get_activities(
            after=datetime.now() - timedelta(days=days),
            limit=200,
        ))
        # Fallback: si no hay nada en la ventana de días, trae las más recientes
        if not activities:
            activities = list(client.get_activities(limit=30))

        sport_map = {
            'Run': 'RUNNING', 'TrailRun': 'RUNNING', 'VirtualRun': 'RUNNING',
            'Ride': 'CYCLING', 'VirtualRide': 'CYCLING', 'GravelRide': 'CYCLING',
            'MountainBikeRide': 'CYCLING',
            'Swim': 'SWIMMING',
        }

        def _sport(activity):
            st_obj = getattr(activity, 'sport_type', None)
            raw = (getattr(st_obj, 'root', None) or str(st_obj)) if st_obj else None
            return sport_map.get(raw, (raw or 'OTHER').upper())

        data = []
        for activity in activities:
            mt = activity.moving_time  # Duration(int): segundos en stravalib 2.x
            data.append({
                'sport': _sport(activity),
                'avg_hr': float(activity.average_heartrate) if activity.average_heartrate else np.nan,
                'max_hr': float(activity.max_heartrate) if activity.max_heartrate else np.nan,
                'avg_speed_kmh': float(activity.average_speed) * 3.6 if activity.average_speed else np.nan,
                'distance': float(activity.distance) / 1000 if activity.distance else np.nan,
                'moving_time': float(mt) / 60 if mt else np.nan,
                'date': activity.start_date,
            })
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error sincronizando Strava: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_strava_athlete_name(access_token):
    """Devuelve el nombre de pila del atleta autenticado en Strava."""
    try:
        client = init_strava_client()
        if not client:
            return None
        client.access_token = access_token
        ath = client.get_athlete()
        return getattr(ath, 'firstname', None) or getattr(ath, 'username', None)
    except Exception:
        return None


def fitness_desde_actividades(acts, fc_rep, fc_max, sexo='M'):
    """Reconstruye la curva CTL/ATL/TSB (Performance Management Chart) a partir
    de las actividades reales, usando la carga de Banister (TRIMP).

    TRIMP = duración(min) · HRr · 0.64 · e^(1.92·HRr)  (HRr = reserva cardiaca usada)
    CTL = media exponencial 42 días (forma) · ATL = 7 días (fatiga) · TSB = CTL−ATL.
    Es la misma lógica del notebook, aplicada aquí en vivo sobre Strava.
    """
    if acts is None or acts.empty or 'date' not in acts.columns:
        return pd.DataFrame()
    df = acts.dropna(subset=['date']).copy()
    if df.empty:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['date'])
    df['date'] = df['date'].dt.normalize()
    fcr = max(float(fc_max) - float(fc_rep), 1.0)
    # Coeficientes de Banister específicos por sexo (distinta respuesta de lactato)
    coef, k = (0.64, 1.92) if sexo == 'M' else (0.86, 1.67)

    def _trimp(row):
        dur = row.get('moving_time')
        if pd.isna(dur) or dur <= 0:
            return 0.0
        hr = row.get('avg_hr')
        ratio = 0.65 if pd.isna(hr) else float(np.clip((hr - fc_rep) / fcr, 0.1, 1.0))
        return float(dur) * ratio * coef * float(np.exp(k * ratio))

    df['trimp'] = df.apply(_trimp, axis=1)
    daily = df.groupby('date')['trimp'].sum()
    if daily.empty:
        return pd.DataFrame()
    idx = pd.date_range(daily.index.min(), pd.Timestamp.today().normalize(), freq='D')
    daily = daily.reindex(idx, fill_value=0.0)
    ctl = daily.ewm(span=42, adjust=False).mean()
    atl = daily.ewm(span=7, adjust=False).mean()
    out = pd.DataFrame({'date': idx, 'CTL': ctl.values, 'ATL': atl.values})
    out['TSB'] = out['CTL'] - out['ATL']
    return out


# ----------------------------------------------------------------------------
# Inferencia con los modelos entrenados del notebook (XGBoost + KMeans)
# ----------------------------------------------------------------------------
MODELOS_DIR = 'salidas/modelos'


@st.cache_resource(show_spinner=False)
def cargar_modelos():
    """Carga los modelos .joblib del notebook si están presentes.
    Devuelve None si falta joblib/sklearn/xgboost o algún archivo.
    """
    try:
        import joblib
        req = ['xgb_run2cyc.joblib', 'xgb_cyc2run.joblib',
               'kmeans_perfiles.joblib', 'preprocesadores.joblib']
        paths = {f: os.path.join(MODELOS_DIR, f) for f in req}
        if not all(os.path.exists(p) for p in paths.values()):
            return None
        return {
            'run2cyc': joblib.load(paths['xgb_run2cyc.joblib']),
            'cyc2run': joblib.load(paths['xgb_cyc2run.joblib']),
            'kmeans': joblib.load(paths['kmeans_perfiles.joblib']),
            'pp': joblib.load(paths['preprocesadores.joblib']),
        }
    except Exception:
        return None


def estado_modelos():
    """Diagnóstico legible del estado de los modelos, sin cargarlos."""
    import importlib.util
    faltan_lib = [m for m in ('joblib', 'sklearn', 'xgboost')
                  if importlib.util.find_spec(m) is None]
    req = ['xgb_run2cyc.joblib', 'xgb_cyc2run.joblib',
           'kmeans_perfiles.joblib', 'preprocesadores.joblib']
    faltan_arch = [f for f in req if not os.path.exists(os.path.join(MODELOS_DIR, f))]
    if faltan_lib:
        libs = ' '.join('scikit-learn' if m == 'sklearn' else m for m in faltan_lib)
        return ('sin_librerias', f'faltan librerías — pip install {libs}')
    if faltan_arch:
        return ('sin_archivos', f'faltan {len(faltan_arch)} de 4 modelos en salidas/modelos/')
    return ('activo', 'modelos del TFM cargados')


def features_atleta(acts, zonas, fitness, perfil_row, sexo='M'):
    """Construye el vector de características del atleta a partir de los datos
    disponibles (Strava o locales). Lo que no se conoce queda NaN: XGBoost lo
    maneja nativamente y el imputer del KMeans lo rellena con la mediana."""
    f = {}
    fc_rep, fc_max = float(zonas['fc_rep']), float(zonas['fc_max'])
    fcr = max(fc_max - fc_rep, 1.0)
    coef_trimp, k_trimp = (0.64, 1.92) if sexo == 'M' else (0.86, 1.67)

    def g(s, k):
        try:
            return s.get(k, np.nan)
        except Exception:
            return np.nan

    for k in ('resting_hr', 'weight_kg', 'age', 'vo2max'):
        f[k] = g(perfil_row, k)
    # sexo como variable del modelo de transferencia real (1 = hombre, 0 = mujer)
    f['sexo_M'] = 1.0 if sexo == 'M' else 0.0

    if acts is not None and not acts.empty:
        cols = set(acts.columns)
        runs = acts[acts['sport'] == 'RUNNING'] if 'sport' in cols else acts.iloc[0:0]
        cyc = acts[acts['sport'] == 'CYCLING'] if 'sport' in cols else acts.iloc[0:0]
        swm = acts[acts['sport'] == 'SWIMMING'] if 'sport' in cols else acts.iloc[0:0]
        if 'avg_speed_kmh' in cols:
            if not runs.empty:
                f['pace_run_kmh'] = runs['avg_speed_kmh'].median()
                f['pace_running_kmh'] = f['pace_run_kmh']
                f['running_speed_kmh'] = f['pace_run_kmh']   # nombre usado por el modelo real
            if not cyc.empty:
                f['pace_cycling_kmh'] = cyc['avg_speed_kmh'].median()
            if not swm.empty:
                f['swim_speed_kmh'] = swm['avg_speed_kmh'].median()
        if 'avg_hr' in cols:
            f['avg_hr'] = acts['avg_hr'].mean()
            if not runs.empty:
                f['avg_hr_run'] = runs['avg_hr'].mean()
                f['fc_umbral_run'] = runs['avg_hr'].quantile(0.85)
            if not cyc.empty:
                f['fc_umbral_cyc'] = cyc['avg_hr'].quantile(0.85)
        if 'avg_cadence' in cols:
            f['avg_cadence'] = acts['avg_cadence'].mean()
        if 'moving_time' in cols:
            tmp = acts.dropna(subset=['moving_time']).copy()

            def _tr(r):
                hr, dur = r.get('avg_hr'), r.get('moving_time')
                if pd.isna(dur) or dur <= 0:
                    return 0.0
                ratio = 0.65 if pd.isna(hr) else float(np.clip((hr - fc_rep) / fcr, 0.1, 1.0))
                return float(dur) * ratio * coef_trimp * float(np.exp(k_trimp * ratio))

            tmp['trimp'] = tmp.apply(_tr, axis=1)
            f['training_load'] = tmp['trimp'].mean()
            if 'date' in cols:
                d = pd.to_datetime(acts['date'], utc=True, errors='coerce').dropna()
                wk = max((d.max() - d.min()).days / 7.0, 1.0) if len(d) > 1 else 1.0
            else:
                wk = max(len(acts) / 3.0, 1.0)
            f['weekly_trimp'] = tmp['trimp'].sum() / wk
            f['sessions_per_week'] = len(acts) / wk
            if 'avg_hr' in cols:
                a = acts.dropna(subset=['avg_hr']).copy()
                if not a.empty:
                    w = a['moving_time'].fillna(1.0).clip(lower=1.0)
                    tot = w.sum()
                    if tot > 0:
                        f['pct_z12'] = float(w[a['avg_hr'] <= zonas['hr']['Z2'][1]].sum() / tot * 100)
                        f['pct_z45'] = float(w[a['avg_hr'] >= zonas['hr']['Z4'][0]].sum() / tot * 100)

    if fitness is not None and not fitness.empty and {'CTL', 'TSB'} <= set(fitness.columns):
        f['ctl_current'] = float(fitness['CTL'].iloc[-1])
        f['tsb_current'] = float(fitness['TSB'].iloc[-1])
        f['ctl_cycling'] = f['ctl_current']
    return f


def inferir_con_modelos(modelos, feats):
    """Aplica los modelos entrenados al vector de características."""
    out = {}
    pp = modelos['pp']
    try:
        fc = pp.get('feature_cols', [])
        if fc:
            row = np.array([[feats.get(c, np.nan) for c in fc]], dtype=float)
            out['bici_pred_kmh'] = float(modelos['run2cyc'].predict(row)[0])
    except Exception:
        pass
    try:
        fr = pp.get('feat_cyc2run', [])
        if fr:
            row = np.array([[feats.get(c, np.nan) for c in fr]], dtype=float)
            out['run_pred_kmh'] = float(modelos['cyc2run'].predict(row)[0])
    except Exception:
        pass
    try:
        cf = pp.get('cluster_features', [])
        if cf:
            row = np.array([[feats.get(c, np.nan) for c in cf]], dtype=float)
            row = pp['imputer'].transform(row)
            row = pp['scaler'].transform(row)
            cl = int(modelos['kmeans'].predict(row)[0])
            names = pp.get('profile_names', {})
            out['cluster'] = cl
            out['perfil_nombre'] = names.get(cl) or names.get(cl % 4) or f'Perfil {cl}'
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------
# Motor (réplica ligera del notebook, secciones 7-8)
# ----------------------------------------------------------------------------
def calcular_zonas(perfil_row, acts, perfil_tipo='nuevo', pred_bici=np.nan):
    p = perfil_row
    fc_max = np.nan
    if not acts.empty and 'max_hr' in acts.columns:
        fc_max = acts['max_hr'].dropna().quantile(0.98)
    if pd.isna(fc_max):
        fc_max = 185.0
    fc_rep = p.get('resting_hr', np.nan)
    if pd.isna(fc_rep):
        fc_rep = 55.0
    # Zonas por % de FC MÁXIMA (convención por defecto de Garmin/Polar/Coros), para
    # que coincidan con lo que el atleta ve en su reloj. Antes se usaba % de FC de
    # reserva (Karvonen), que da valores ~15-20 ppm más altos y NO casaba con el
    # reloj: inflaba Z1/Z2 y dejaba fuera las carreras fáciles reales, lo que
    # estimaba un ritmo de rodaje demasiado lento.
    def _hr(lo, hi):
        return (int(fc_max * lo), int(fc_max * hi))

    hr = {'Z1': _hr(.50, .60), 'Z2': _hr(.60, .70), 'Z3': _hr(.70, .80),
          'Z4': _hr(.80, .90), 'Z5': _hr(.90, 1.0)}

    # ------------------------------------------------------------------
    # Ritmos por disciplina derivados del HISTORIAL REAL (modelo de coach):
    #   · ritmo FÁCIL  = mediana de actividades en banda aeróbica de FC
    #     (carrera 65-78 % FCmáx ≈ zona "Easy" de Daniels; bici 60-75 %).
    #   · ritmo UMBRAL = mediana en banda 83-92 % FCmáx (o p80 de velocidad).
    #   El resto de zonas se derivan del umbral, como en el sistema VDOT.
    #   Si no hay FC suficiente, se usan cuantiles de velocidad (robusto).
    # ------------------------------------------------------------------
    def _ritmos(d, easy=(0.65, 0.78), thr=(0.83, 0.92), q_thr=0.80, q_easy=0.35):
        if d is None or d.empty or 'avg_speed_kmh' not in d.columns:
            return np.nan, np.nan, 0
        s = d['avg_speed_kmh'].dropna()
        s = s[s > 0]
        if s.empty:
            return np.nan, np.nan, 0
        # Ventana RECIENTE (8 semanas) para el ritmo FÁCIL → refleja tu forma
        # ACTUAL: al re-sincronizar Strava cada semana, el ritmo sube con tu
        # progreso real. El UMBRAL se toma del histórico completo (necesita
        # esfuerzos fuertes, más escasos). Si hay poca data reciente, usa todo.
        d_rec = d
        if 'date' in d.columns:
            dt = pd.to_datetime(d['date'], utc=True, errors='coerce')
            ref = dt.max()
            if pd.notna(ref):
                m = dt >= (ref - pd.Timedelta(days=56))
                if m.sum() >= 4:
                    d_rec = d[m]
        v_e = v_t = np.nan
        n_e = 0
        if 'avg_hr' in d.columns and fc_max:
            if d_rec['avg_hr'].notna().sum() >= 3:
                pe = d_rec['avg_hr'] / fc_max
                e = d_rec.loc[pe.between(*easy), 'avg_speed_kmh'].dropna()
                n_e = len(e)
                if len(e) >= 3:
                    v_e = float(e.median())
            if d['avg_hr'].notna().sum() >= 5:
                pt = d['avg_hr'] / fc_max
                tt = d.loc[pt.between(*thr), 'avg_speed_kmh'].dropna()
                if len(tt) >= 3:
                    v_t = float(tt.median())
        if np.isnan(v_t):
            v_t = float(s.quantile(q_thr))
        if np.isnan(v_e):
            s_rec = d_rec['avg_speed_kmh'].dropna()
            s_rec = s_rec[s_rec > 0]
            v_e = float(min((s_rec if not s_rec.empty else s).quantile(q_easy), v_t * 0.85))
        v_e = float(min(v_e, v_t * 0.95))   # coherencia: fácil siempre < umbral
        return v_e, v_t, n_e

    runs = cyc = swims = pd.DataFrame()
    if not acts.empty and 'sport' in acts.columns:
        runs = acts[acts['sport'] == 'RUNNING']
        cyc = acts[acts['sport'] == 'CYCLING']
        swims = acts[acts['sport'] == 'SWIMMING']

    # CARRERA
    v_run_easy, v_run_thr, n_e_run = _ritmos(runs)
    if np.isnan(v_run_easy):
        v0 = p.get('pace_running_kmh', np.nan)
        v0 = 9.2 if pd.isna(v0) else v0
        v_run_thr, v_run_easy = v0 * 1.05, v0 * 0.90
        fuente_z2 = 'estimado a partir de tu perfil (sin FC en el historial)'
    elif n_e_run >= 3:
        fuente_z2 = f'validado con {n_e_run} carreras reales en banda aeróbica (65-78 % FCmáx)'
    else:
        fuente_z2 = 'estimado por cuantiles de velocidad de tu historial'

    # CICLISMO
    v_bike_easy, v_bike_thr, _ = _ritmos(cyc, easy=(0.60, 0.75), thr=(0.83, 0.92))
    if np.isnan(v_bike_easy):
        vb = pred_bici if not pd.isna(pred_bici) else p.get('pace_cycling_kmh', np.nan)
        vb = 25.0 if pd.isna(vb) else vb
        v_bike_thr, v_bike_easy = vb, vb * 0.82

    # NATACIÓN (la FC en piscina es poco fiable → cuantiles de velocidad)
    if not swims.empty and 'avg_speed_kmh' in swims.columns:
        ss = swims['avg_speed_kmh'].dropna()
        ss = ss[ss > 0]
        v_nat_easy = float(ss.median()) if not ss.empty else np.nan
        v_nat_thr = float(ss.quantile(0.80)) if not ss.empty else np.nan
    else:
        v_nat_easy = v_nat_thr = np.nan
    if np.isnan(v_nat_easy):
        vn = p.get('pace_swimming_kmh', np.nan)
        vn = 2.6 if pd.isna(vn) else vn
        v_nat_easy, v_nat_thr = vn, vn * 1.08

    # ------------------------------------------------------------------
    # GUARDRAILS fisiológicos orientados al 70.3: un historial escaso o no
    # representativo (p. ej. pocas salidas de bici, todas muy lentas) no debe
    # producir objetivos imposibles para completar la prueba (90 km en bici no
    # se entrenan a 9 km/h). Son pisos suaves hacia las demandas de la carrera.
    # ------------------------------------------------------------------
    if len(cyc) < 6:   # casi sin historial de bici → objetivo orientado a la meta
        base_thr = pred_bici if not pd.isna(pred_bici) else 27.0
        v_bike_thr = max(v_bike_thr, base_thr)
        v_bike_easy = v_bike_thr * 0.82
    v_run_easy = max(v_run_easy, 7.0)                 # ≥ ~8:34/km (no objetivos de caminar)
    v_run_thr = max(v_run_thr, v_run_easy * 1.08)
    v_bike_easy = max(v_bike_easy, 18.0)              # piso de rodaje de bici para 70.3
    v_bike_thr = max(v_bike_thr, v_bike_easy * 1.08, 22.0)
    v_nat_easy = max(v_nat_easy, 2.2)
    v_nat_thr = max(v_nat_thr, v_nat_easy * 1.05)

    v_bici_obj = pred_bici if not pd.isna(pred_bici) else v_bike_thr * 0.95
    v_bici_obj = max(v_bici_obj, 22.0)                # ritmo de bici mínimo de carrera

    # Ritmo objetivo de carrera del 70.3: entre el fácil y el umbral (esfuerzo
    # sostenible de media-larga distancia). Garantiza fácil < objetivo < umbral.
    v_run_obj = float(min(v_run_thr * 0.96, v_run_easy * 1.04))

    return {'fc_max': int(fc_max), 'fc_rep': int(fc_rep), 'hr': hr,
            'fuente_z2': fuente_z2, 'perfil': perfil_tipo,
            'run_z2': v_run_easy, 'run_z3': v_run_thr * 0.92, 'run_z4': v_run_thr,
            'run_obj': v_run_obj, 'bici_obj': v_bici_obj, 'nat_obj': v_nat_easy,
            'bici_z2': v_bike_easy, 'bici_z4': v_bike_thr,
            'nat_z2': v_nat_easy, 'nat_z4': v_nat_thr}


def calcular_fase(fecha_comp, perfil_tipo, fecha_inicio=None, semana_sim=None):
    """Calcula fase, semana actual y semanas restantes a partir de fechas reales.

    - semana_actual = semanas transcurridas desde la fecha de inicio + 1
      (si empiezas hoy, estás en la semana 1).
    - semanas_totales = de la fecha de inicio a la competencia.
    - semanas_restantes = de hoy a la competencia (determina la fase).
    - semana_sim: si se indica, FUERZA la semana actual (modo simulación/demo);
      las semanas restantes se derivan de ella, no del calendario.
    """
    sem_min = SEMANAS_MINIMAS[perfil_tipo]
    hoy = pd.Timestamp.today().normalize()
    ini = hoy if fecha_inicio is None else pd.Timestamp(fecha_inicio).normalize()
    if ini > hoy:
        ini = hoy  # no permitir inicio en el futuro

    if fecha_comp:
        fc = pd.Timestamp(fecha_comp).normalize()
        sem_rest = max(int(np.ceil((fc - hoy).days / 7)), 0)
        sem_tot = max(int(np.ceil((fc - ini).days / 7)), 1)
    else:
        sem_tot = sem_min
        sem_rest = sem_tot

    if semana_sim:  # modo simulación: la semana la fija el usuario
        sem_actual = int(min(max(semana_sim, 1), sem_tot))
        sem_rest = max(sem_tot - sem_actual, 0)
    else:
        sem_actual = int((hoy - ini).days // 7) + 1
        sem_actual = min(max(sem_actual, 1), sem_tot)
    viable = sem_tot >= sem_min

    if sem_rest <= 1:
        fase = 'COMPETENCIA'
    elif sem_rest <= 3:
        fase = 'TAPERING'
    elif sem_rest <= 6:
        fase = 'PICO'
    elif sem_rest <= 12:
        fase = 'CONSTRUCCION'
    else:
        fase = 'BASE'
    return {'fase': fase, 'semanas_restantes': sem_rest, 'semanas_totales': sem_tot,
            'semana_actual': sem_actual, 'plan_viable': viable, 'semanas_minimas': sem_min}


def analizar_objetivo(anterior, objetivo_total, sem_prep):
    """anterior = dict con 'nat','bici','run','total' en 'h:mm'."""
    try:
        ant = {k: t2min(v) for k, v in anterior.items()}
        t_obj = t2min(objetivo_total)
    except Exception:
        return None
    mejora = ant['total'] - t_obj
    if mejora <= 0:
        return None
    transiciones = max(ant['total'] - (ant['nat'] + ant['bici'] + ant['run']), 4)
    reparto = {'nat': 0.15, 'bici': 0.50, 'run': 0.35}
    splits = {k: ant[k] - mejora * w for k, w in reparto.items()}
    mejora_pct = mejora / ant['total'] * 100
    return {'anterior_total': min2t(ant['total']), 'objetivo_total': objetivo_total,
            'mejora_min': round(mejora, 1), 'mejora_pct': round(mejora_pct, 1),
            'viable': mejora_pct <= max(sem_prep, 1) * 0.5,
            'transiciones_min': round(transiciones, 1),
            'splits_obj': {k: min2t(v) for k, v in splits.items()},
            'v_nat_obj': 1.9 / (splits['nat'] / 60),
            'v_bici_obj': 90 / (splits['bici'] / 60),
            'v_run_obj': 21.1 / (splits['run'] / 60)}


def sleep_score_label(score):
    """Etiqueta de calidad del sueño según el Sleep Score (estilo Garmin)."""
    if score is None:
        return ''
    if score >= 90:
        return 'Excelente'
    if score >= 80:
        return 'Buena'
    if score >= 60:
        return 'Regular'
    return 'Mala'


def actualizar_hoy(sesiones_hoy, z, sueno_h, body_battery, stress, enfermo,
                   sleep_score=None):
    graves, moderados = [], []
    if enfermo:
        graves.append('enfermedad o malestar')
    if body_battery < 20:
        graves.append(f'Body Battery muy baja ({body_battery:.0f})')
    if sueno_h < 5:
        graves.append(f'noche muy corta ({sueno_h:.1f} h)')
    # Sleep Score muy bajo: aunque las horas sean correctas, el sueño no reparó
    if sleep_score is not None and sleep_score < 40:
        graves.append(f'calidad de sueño muy mala (score {sleep_score:.0f}/100)')
    if not graves:
        if body_battery < 35:
            moderados.append(f'Body Battery baja ({body_battery:.0f})')
        if sueno_h < 6.5:
            moderados.append(f'sueño insuficiente ({sueno_h:.1f} h)')
        if sleep_score is not None and sleep_score < 60:
            moderados.append(f'calidad de sueño baja (score {sleep_score:.0f}/100, '
                             f'«{sleep_score_label(sleep_score)}»)')
        if stress > 70:
            moderados.append(f'estrés alto ({stress:.0f})')
    if graves:
        return ('DESCANSO HOY', graves,
                'Descanso completo: caminar suave, hidratarte y priorizar dormir. '
                'La sesión prevista se reprograma, no se pierde.')
    if moderados:
        partes = [f"{s['deporte']}: versión suave de {int(s['dur_min'] * 0.6)} min, "
                  f"todo en Z1-Z2 ({z['hr']['Z1'][0]}-{z['hr']['Z2'][1]} ppm), sin calidad."
                  for s in sesiones_hoy if s['deporte'] not in ('Descanso', 'Movilidad')]
        return ('SUAVIZAR HOY', moderados,
                ' '.join(partes) or 'Hoy ya era descanso: mantenlo.')
    return ('MANTENER', ['métricas en rango'],
            'Tus métricas de esta mañana están en tu rango: el plan sigue tal cual.')


def plan_competencia(z, objetivo=None):
    hr = z['hr']
    v_b = objetivo['v_bici_obj'] if objetivo else z['bici_obj']
    v_r = objetivo['v_run_obj'] if objetivo else z['run_obj']
    v_n = objetivo['v_nat_obj'] if objetivo else z['nat_obj']
    seg = [
        ('NATACIÓN 1,9 km', '0-300 m',
         'Salida controlada al 90%: deja pasar al grupo rápido y busca agua libre. No dispares la FC.'),
        ('NATACIÓN 1,9 km', '300-1.600 m',
         f'Crucero a {pace_swim(v_n)}. Pies de alguien un punto más rápido (drafting legal: ahorra ~5%).'),
        ('NATACIÓN 1,9 km', '1.600-1.900 m',
         'Aumenta el pataleo: sangre a las piernas antes de ponerte de pie. Visualiza la T1.'),
        ('T1', '—',
         f"Saldrás con la FC en {hr['Z4'][0]}+ ppm aunque no hayas apretado (estrés, neopreno, "
         f"horizontal→vertical). Camina rápido, no esprintes."),
        ('BICI 90 km', 'km 0-5',
         f"Arranque PROGRESIVO al 85% ({v_b * 0.85:.0f} km/h), 90-95 rpm. No persigas la FC: "
         f"espera a que baje de {hr['Z3'][0]} ppm. Bebe ya."),
        ('BICI 90 km', 'km 5-15',
         f"Sube hacia {v_b:.0f} km/h SOLO cuando la FC se asiente en {hr['Z2'][0]}-{hr['Z3'][0]} ppm. "
         f"Primera comida km 10-15."),
        ('BICI 90 km', 'km 15-70',
         f"Crucero {v_b:.0f} km/h, FC tope {hr['Z3'][1]} ppm. 60-90 g de carbohidrato/hora. "
         f"Subidas sentado aunque te adelanten: lo devuelves corriendo."),
        ('BICI 90 km', 'km 70-85',
         f"El tramo del juicio: si vas bien, MANTÉN {v_b:.0f} km/h — no subas. "
         f"Última comida sólida antes del km 80."),
        ('BICI 90 km', 'km 85-90',
         f"Afloja al 95% ({v_b * 0.95:.0f} km/h), cadencia 95-100 rpm: piernas sueltas para la T2."),
        ('T2', '—',
         'La FC sube al desmontar: normal. Cambio rápido sin pánico; gorra, dorsal y a correr.'),
        ('CARRERA 21,1 km', 'km 0-3',
         f"BRICK: piernas de cemento. Sal a {pace_run(v_r * 0.92)} aunque te sientas capaz de más: "
         f"la FC debe asentarse."),
        ('CARRERA 21,1 km', 'km 3-10',
         f"Ritmo objetivo {pace_run(v_r)}, FC tope {hr['Z3'][1]} ppm. Gel en el km 7-8."),
        ('CARRERA 21,1 km', 'km 10-16',
         f"Mantén {pace_run(v_r)}. Si la FC entra en Z4 ({hr['Z4'][0]}+) sin subir el ritmo, "
         f"afloja 10-15 s/km y reevalúa."),
        ('CARRERA 21,1 km', 'km 16-19',
         f"La parte dura. Permitido Z4 bajo ({hr['Z4'][0]}-{int((hr['Z4'][0] + hr['Z4'][1]) / 2)} ppm) "
         f"si las sensaciones acompañan."),
        ('CARRERA 21,1 km', 'km 19-21,1',
         'Vacía el tanque: todo lo que quede. La meta de un 70.3 se recuerda toda la vida.'),
    ]
    t_nat = 1.9 / v_n * 60
    t_bici = (5 / (v_b * 0.85) + 10 / (v_b * 0.95) + 70 / v_b + 5 / (v_b * 0.95)) * 60
    t_run = (3 / (v_r * 0.92) + 18.1 / v_r) * 60
    trans = objetivo['transiciones_min'] if objetivo else 8
    total = t_nat + trans + t_bici + t_run
    resumen = (f"Natación {min2t(t_nat)} + T1/T2 {min2t(trans)} + bici {min2t(t_bici)} "
               f"+ carrera {min2t(t_run)} = **{min2t(total)}**")
    return seg, resumen


def frecuencia_y_disciplinas(acts):
    """Sesiones/semana y disciplinas practicadas según el historial reciente."""
    if acts is None or getattr(acts, 'empty', True):
        return None, set()
    disc = set(acts['sport'].dropna().unique()) if 'sport' in acts.columns else set()
    if 'date' in acts.columns:
        d = pd.to_datetime(acts['date'], utc=True, errors='coerce').dropna()
        if len(d) > 1:
            semanas = max((d.max() - d.min()).days / 7.0, 1.0)
            return len(acts) / semanas, disc
    return None, disc


def factor_mejora(per, perfil_tipo):
    """Factor de mejora del ritmo según el avance del plan (mismo pulso, ritmo más
    rápido por adaptación aeróbica). Mayor en principiantes. Se usa tanto en los
    ritmos que se MUESTRAN como en los del plan, para que sean coherentes."""
    sem_tot = max(int(per.get('semanas_totales', 24)), 1)
    sem_idx = min(max(int(per.get('semana_actual', 1)), 1), sem_tot)
    prog = sem_idx / sem_tot
    tope = 0.10 if perfil_tipo == 'nuevo' else 0.05
    return 1.0 + tope * prog


def generar_plan_semanal(perfil_tipo, per, zonas, acts=None, sexo='M'):
    """Plan semanal ADAPTATIVO orientado a COMPLETAR un 70.3 (1,9-90-21,1 km).

    - Volúmenes reales que progresan hacia la distancia de carrera: el fondo de
      bici crece hasta acercarse a los 90 km y la tirada larga hacia los 21 km.
    - Incluye BRICK (bici+carrera encadenadas), la sesión clave del 70.3, en
      Construcción y Pico.
    - Frecuencia y carga suben de forma gradual según la semana y la fase.
    - Los ritmos objetivo PROGRESAN con las semanas (mismo pulso, ritmo más
      rápido) y salen de los datos del propio atleta y del modelo (que ya
      incluye el sexo); no se les aplica una penalización extra por sexo.
    - 1-2 bricks/semana en construcción y pico: largo el sábado y mini-brick en
      pico el miércoles, con la carrera del brick creciendo hacia la distancia.
    """
    fase = per.get('fase', 'BASE')
    sem_tot = max(int(per.get('semanas_totales', 24)), 1)
    sem_idx = int(per.get('semana_actual',
                          max(sem_tot - int(per.get('semanas_restantes', 0)), 1)))
    sem_idx = min(max(sem_idx, 1), sem_tot)
    prog = sem_idx / sem_tot  # avance del plan, 0→1

    freq_act, disciplinas = frecuencia_y_disciplinas(acts)
    if perfil_tipo == 'nuevo':
        f_ini = int(min(max(round(freq_act), 3), 4)) if freq_act else 3
        f_fin = 6
    else:
        f_ini = int(min(max(round(freq_act), 4), 6)) if freq_act else 5
        f_fin = 6
    n_ses = int(round(f_ini + (f_fin - f_ini) * min(prog * 1.3, 1.0)))
    if fase == 'TAPERING':
        n_ses -= 1
    if fase == 'COMPETENCIA':
        n_ses = 3
    n_ses = int(min(max(n_ses, 3), 6))

    # --- Volumen: rampa hacia el pico × fase × nivel del atleta ---
    ramp = 0.45 + 0.55 * min(prog / 0.85, 1.0)        # 0.45 (inicio) → 1.0 (~85 % del plan)
    fase_mult = {'BASE': 0.80, 'CONSTRUCCION': 1.0, 'PICO': 1.0,
                 'TAPERING': 0.55, 'COMPETENCIA': 0.35}.get(fase, 1.0)
    nivel = 0.80 if perfil_tipo == 'nuevo' else 1.0
    vol = ramp * fase_mult * nivel
    intens = fase in ('CONSTRUCCION', 'PICO')

    # Semana de DESCARGA (recuperación) cada 4ª semana — patrón clásico 3:1.
    # Se reduce el volumen ~40 % y se baja la intensidad para asimilar la carga
    # acumulada (supercompensación) y prevenir el sobreentrenamiento.
    es_descarga = (sem_idx % 4 == 0) and fase in ('BASE', 'CONSTRUCCION', 'PICO') \
        and sem_idx < sem_tot - 2
    if es_descarga:
        vol *= 0.6
        intens = False

    # Progresión de ritmos: a igual pulso, el ritmo mejora con las semanas por
    # adaptación aeróbica (la esencia de un plan de entrenamiento). El tope de
    # mejora es mayor en perfiles nuevos. Los ritmos base ya provienen de los
    # datos del propio atleta y del modelo (que incluye el sexo), por lo que NO
    # se aplica aquí un factor de sexo: hacerlo penalizaba dos veces a la mujer.
    mejora = factor_mejora(per, perfil_tipo)
    v_b2, v_b4 = zonas['bici_z2'] * mejora, zonas['bici_z4'] * mejora
    # Ritmos de carrera DIFERENCIADOS por tipo de sesión (no todos al mismo ritmo):
    #   fácil/rodaje y tirada larga = Z2; objetivo 70.3 (brick y final de la larga);
    #   tempo = Z3; series = Z4. Todos progresan con las semanas (factor mejora).
    v_run = zonas['run_z2'] * mejora        # fácil / rodaje / base de la tirada larga
    v_run3 = zonas['run_z3'] * mejora       # tempo (Z3)
    v_run4 = zonas['run_z4'] * mejora       # series (Z4)
    v_robj = zonas['run_obj'] * mejora      # ritmo OBJETIVO de carrera 70.3
    v_nat = zonas['nat_z2'] * mejora
    hrz2 = f"{zonas['hr']['Z2'][0]}-{zonas['hr']['Z2'][1]} ppm"
    rz2 = pace_run(v_run)
    robj = pace_run(v_robj)
    rz4 = pace_run(v_run4)
    nz2 = pace_swim(v_nat)

    def dur(peak):
        return max(int(round(peak * vol / 5.0) * 5), 20)   # min, múltiplos de 5
    def kb(m, v):
        return round(m / 60 * v, 1)
    def kr(m):
        return round(m / 60 * v_run, 1)
    def ks(m):
        return round(m / 60 * v_nat, 2)

    nat_intro = bool(disciplinas) and 'SWIMMING' not in disciplinas

    # --- Miércoles: bici de calidad (+ mini-BRICK de transición en PICO) ---
    # Segundo estímulo bici→carrera de la semana. Los expertos recomiendan 1-2
    # bricks por semana en construcción/pico: uno largo (sábado) y, en pico, un
    # mini-brick de transición (10-15 min) para reforzar el patrón neuromuscular.
    mie_bike = dur(85)
    # Mini-brick en TODA Construcción y Pico (no solo Pico): así hay 2 bricks/semana
    # (sábado largo + miércoles corto), dentro del rango 1-2/sem que recomiendan los
    # entrenadores en las fases específicas. La transición se corre a ritmo objetivo.
    mini_brick = dur(15) if intens else 0
    if intens and mini_brick:
        mie = {'dia': 'Miércoles', 'deporte': 'Ciclismo', 'tipo': 'series Z4 + mini-BRICK',
               'dur_min': mie_bike + mini_brick, 'km': kb(mie_bike, v_b4),
               'detalle': (f'4x8 min en Z4 a ~{v_b4:.0f} km/h (4 min suaves entre series) y, '
                           f'al bajar de la bici, {mini_brick} min de transición a {robj} '
                           f'(ritmo objetivo). 2º estímulo bici→carrera de la semana.')}
    elif intens:
        mie = {'dia': 'Miércoles', 'deporte': 'Ciclismo', 'tipo': 'series Z4',
               'dur_min': mie_bike, 'km': kb(mie_bike, v_b4),
               'detalle': f'4x8 min en Z4 a ~{v_b4:.0f} km/h, 4 min suaves entre series.'}
    else:
        mie = {'dia': 'Miércoles', 'deporte': 'Ciclismo', 'tipo': 'rodaje Z2',
               'dur_min': mie_bike, 'km': kb(mie_bike, v_b2),
               'detalle': f'Rodaje a {v_b2:.0f} km/h en Z2, cadencia 85-95 rpm.'}

    # --- Sábado: FONDO de bici + BRICK de carrera ---
    # El brick NO se limita a Construcción/Pico: se introduce ya en Base tardío
    # (suave) y se mantiene casi todas las semanas, también en descarga (más corto,
    # porque es una sesión de ESPECIFICIDAD y técnica, no solo de carga). La carrera
    # del brick crece de ~15-20 min (Base) a ~45-50 min (Pico) — rango 30-60 min
    # recomendado para 70.3; 20 min en Pico no prepara una carrera de >2 h.
    bike_long = dur(220)                       # pico ≈ 3,5 h → cerca de los 90 km
    hay_brick = (fase in ('BASE', 'CONSTRUCCION', 'PICO')) and prog >= 0.25
    if hay_brick:
        brick_peak = 55 if intens else 30      # carrera del brick objetivo (min)
        brick_run = dur(brick_peak)
        clave = '🔑 ' if intens else ''
        modo = 'sesión clave del 70.3' if intens else 'brick introductorio suave'
        sab = {'dia': 'Sábado', 'deporte': 'Ciclismo',
               'tipo': 'fondo + BRICK' if intens else 'fondo + brick suave',
               'dur_min': bike_long + brick_run, 'km': kb(bike_long, v_b2),
               'detalle': (f'{clave}BRICK ({modo}): fondo de '
                           f'{kb(bike_long, v_b2):.0f} km de bici a {v_b2:.0f} km/h en Z2 '
                           f'({hrz2}) y, SIN parar, {brick_run} min de carrera a {robj} '
                           f'(ritmo OBJETIVO 70.3, {brick_run/60*v_robj:.1f} km) con "piernas '
                           f'de cemento": aquí ensayas tu ritmo de carrera con fatiga. Crece '
                           f'semana a semana. Come e hidrátate como en competencia.')}
    else:
        brick_run = 0
        sab = {'dia': 'Sábado', 'deporte': 'Ciclismo', 'tipo': 'fondo Z2',
               'dur_min': bike_long, 'km': kb(bike_long, v_b2),
               'detalle': (f'Fondo de {kb(bike_long, v_b2):.0f} km a {v_b2:.0f} km/h en Z2 '
                           f'({hrz2}). Construye la resistencia aeróbica de base hacia los 90 km; '
                           f'practica comida e hidratación de carrera.')}

    # --- Domingo: TIRADA LARGA orientada a distancia (tiempo en pie) ---
    # Construye hacia ~18 km en pico (recomendación 70.3: 16-20 km, no media maratón
    # completa), con tope de 140 min para no castigar a corredores lentos. Es
    # progresiva: arranca fácil (Z2) y termina los últimos km a ritmo objetivo.
    larga_km = 18.0 * min(prog / 0.85, 1.0)
    if es_descarga:
        larga_km *= 0.7
    larga_min = int(round(min(max(larga_km / max(v_run, 1) * 60, 40), 140) / 5) * 5)
    larga_km = round(larga_min / 60 * v_run, 1)
    finish_km = max(round(min(larga_km * 0.3, 6.0), 1), 1.0)
    dom = {'dia': 'Domingo', 'deporte': 'Carrera', 'tipo': 'tirada larga progresiva',
           'dur_min': larga_min, 'km': larga_km,
           'detalle': (f'Tirada larga de {larga_km:.1f} km ({larga_min} min): arranca cómodo '
                       f'a {rz2} en Z2 ({hrz2}) y corre los últimos ~{finish_km:.0f} km a '
                       f'{robj} (ritmo objetivo del 70.3). Construye resistencia y enseña a '
                       f'tu cuerpo a correr a ritmo de carrera ya cansado, hacia los 21,1 km.')}

    catalogo = [
        ('Sábado', sab),
        ('Jueves', {'dia': 'Jueves', 'deporte': 'Natación', 'tipo': 'resistencia',
                    'dur_min': dur(55), 'km': ks(dur(55)),
                    'detalle': f'Series de resistencia a {nz2} (técnica y respiración bilateral). '
                               f'Construye los 1,9 km de la carrera.'
                               + (' Empieza por aquí: aún no registras natación.' if nat_intro else '')}),
        ('Domingo', dom),
        ('Martes', {'dia': 'Martes', 'deporte': 'Carrera',
                    'tipo': 'series Z4' if intens else 'rodaje Z2',
                    'dur_min': dur(55), 'km': kr(dur(55)),
                    'detalle': (f'Calienta y haz 5x4 min en Z4 a {rz4} (ritmo de series), '
                                f'2 min suaves entre repeticiones.'
                                if intens else f'Rodaje cómodo a {rz2} en Z2 ({hrz2}).')}),
        ('Miércoles', mie),
        ('Viernes', {'dia': 'Viernes', 'deporte': 'Natación', 'tipo': 'técnica',
                     'dur_min': dur(45), 'km': ks(dur(45)),
                     'detalle': f'Técnica de brazada y pies; series cortas a {nz2}.'}),
    ]
    por_dia = {dia: s for dia, s in catalogo[:n_ses]}

    if fase == 'COMPETENCIA':
        por_dia['Domingo'] = {'dia': 'Domingo', 'deporte': 'Competencia', 'tipo': '70.3',
                              'km': 0, 'dur_min': 0,
                              'detalle': '¡Día de carrera! Aplica el plan km a km de más abajo.'}

    orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    plan = []
    for dia in orden:
        if dia in por_dia:
            plan.append(por_dia[dia])
        else:
            plan.append({'dia': dia, 'deporte': 'Descanso', 'tipo': 'recuperación',
                         'km': 0, 'dur_min': 0,
                         'detalle': 'Descanso o 15 min de movilidad. La adaptación ocurre hoy.'})
    meta = {'n_ses': n_ses, 'sem_idx': sem_idx, 'sem_tot': sem_tot,
            'freq_act': freq_act, 'disciplinas': disciplinas, 'nat_intro': nat_intro,
            'tiene_brick': bool(hay_brick and brick_run),
            'tiene_mini_brick': bool(intens and mini_brick), 'descarga': es_descarga}
    return plan, meta


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def grafico_forma(fd):
    fig, ax = plt.subplots(figsize=(10, 3))
    fd = fd.tail(365)
    ax.plot(fd['date'], fd['CTL'], color='#21a366', lw=2, label='Fitness (CTL)')
    ax.plot(fd['date'], fd['ATL'], color='#f97316', lw=2, label='Fatiga (ATL)')
    ax.plot(fd['date'], fd['TSB'], color='#3b82f6', lw=1.6, ls='--', label='Frescura (TSB)')
    ax.axhline(0, color='#d8d5cd', lw=1)
    ax.legend(loc='upper left', frameon=False, fontsize=9, ncol=3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors='#8a877f')
    fig.tight_layout()
    return fig


def grafico_forma_plotly(fd):
    """Curva CTL/ATL/TSB interactiva (Plotly), responsiva y con hover unificado."""
    fd = fd.tail(365)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fd['date'], y=fd['CTL'], name='Fitness (CTL)',
                             mode='lines', line=dict(color='#21a366', width=2.4)))
    fig.add_trace(go.Scatter(x=fd['date'], y=fd['ATL'], name='Fatiga (ATL)',
                             mode='lines', line=dict(color='#f97316', width=2.4)))
    fig.add_trace(go.Scatter(x=fd['date'], y=fd['TSB'], name='Frescura (TSB)',
                             mode='lines', line=dict(color='#1d6fa5', width=1.8, dash='dot')))
    fig.add_hline(y=0, line=dict(color='#d8d5cd', width=1))
    fig.update_layout(height=300, margin=dict(l=8, r=8, t=10, b=8),
                      legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0,
                                  font=dict(size=12)),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      hovermode='x unified', font=dict(size=12, color='#5b6670'))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)', zeroline=False)
    return fig


def grafico_progreso(acts):
    """Evolución del ritmo medio por disciplina a lo largo del tiempo, con su
    tendencia. Muestra el progreso del atleta y devuelve (fig, % mejora por disciplina)."""
    deps = [('RUNNING', 'Carrera', '#f59e0b'), ('CYCLING', 'Ciclismo', '#22a06b'),
            ('SWIMMING', 'Natación', '#3b82f6')]
    if acts is None or 'sport' not in acts.columns or 'date' not in acts.columns:
        return None, {}
    pres = [(s, n, c) for s, n, c in deps
            if not acts[acts['sport'] == s].dropna(subset=['avg_speed_kmh']).empty]
    if not pres:
        return None, {}
    fig, axes = plt.subplots(1, len(pres), figsize=(10, 2.7))
    if len(pres) == 1:
        axes = [axes]
    mejoras = {}
    for ax, (s, n, c) in zip(axes, pres):
        df = acts[acts['sport'] == s].dropna(subset=['avg_speed_kmh', 'date']).copy()
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None)
        df = df.dropna(subset=['date']).sort_values('date')
        if len(df) < 2:
            ax.axis('off'); continue
        x, y = df['date'], df['avg_speed_kmh'].values
        ax.scatter(x, y, s=14, color=c, alpha=0.45)
        xi = (df['date'] - df['date'].min()).dt.days.values.astype(float)
        if xi.max() > 0:
            m, b = np.polyfit(xi, y, 1)
            ax.plot(x, m * xi + b, color=c, lw=2.2)
            y0, y1 = b, m * xi[-1] + b
            mejoras[n] = (y1 - y0) / y0 * 100 if y0 > 0 else 0.0
        ax.set_title(n, fontsize=10, color=c, fontweight='bold')
        ax.set_ylabel('km/h', fontsize=8)
        ax.tick_params(labelsize=7, colors='#8a877f')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30); lbl.set_ha('right')
    fig.tight_layout()
    return fig, mejoras


def progreso_semanal(acts, desde=None):
    """Media SEMANAL del ritmo por disciplina + KPIs (ritmo actual y % de mejora).

    Solo considera actividades a partir de `desde` (fecha de inicio del plan), porque
    el gráfico refleja la evolución DESDE QUE EMPEZASTE el plan. Devuelve
    (df_indice, kpis): índice de progreso (primera semana = 100) por disciplina.
    """
    deps = [('RUNNING', 'Carrera'), ('CYCLING', 'Ciclismo'), ('SWIMMING', 'Natación')]
    if acts is None or 'sport' not in acts.columns or 'date' not in acts.columns:
        return None, {}
    a = acts.dropna(subset=['avg_speed_kmh']).copy()
    a['date'] = pd.to_datetime(a['date'], utc=True, errors='coerce').dt.tz_localize(None)
    a = a.dropna(subset=['date'])
    if desde is not None:
        a = a[a['date'] >= pd.Timestamp(desde).normalize()]   # solo desde el inicio del plan
    if a.empty:
        return None, {}
    # Semana DESDE EL INICIO (1, 2, 3…), no por calendario: así la semana 1 es
    # un único punto y el gráfico solo aparece con ≥2 semanas de actividad.
    inicio = a['date'].min()
    a['semana'] = ((a['date'] - inicio).dt.days // 7) + 1
    series, kpis = {}, {}
    for s, n in deps:
        d = a[a['sport'] == s]
        if d.empty:
            continue
        wk = d.groupby('semana')['avg_speed_kmh'].mean().sort_index()
        if len(wk) < 2:
            continue
        series[n] = wk
        v0, v1 = float(wk.iloc[0]), float(wk.iloc[-1])
        kpis[n] = (v1, (v1 - v0) / v0 * 100 if v0 > 0 else 0.0)
    if not series:
        return None, {}
    idx = pd.DataFrame({n: s / s.iloc[0] * 100 for n, s in series.items()})
    idx.index.name = 'Semana'
    return idx, kpis


def grafico_zonas(z):
    fig, ax = plt.subplots(figsize=(10, 1.9))
    for zona, (lo, hi) in z['hr'].items():
        ax.barh(0, hi - lo, left=lo, height=0.62, color=GARMIN_COLORS[zona])
        ax.text((lo + hi) / 2, 0, f"{zona}\n{lo}-{hi}", ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
    ax.set_xlim(z['hr']['Z1'][0] - 8, z['hr']['Z5'][1] + 8)
    ax.set_yticks([])
    ax.set_xlabel('pulsaciones por minuto (ppm)', fontsize=9)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def grafico_zonas_plotly(z):
    """Barra de zonas de FC interactiva (Plotly): cada tramo Z1-Z5 con su rango."""
    fig = go.Figure()
    for zona, (lo, hi) in z['hr'].items():
        fig.add_trace(go.Bar(
            y=['FC'], x=[hi - lo], base=[lo], orientation='h', width=0.6,
            marker=dict(color=GARMIN_COLORS[zona]),
            text=[f"{zona}<br>{lo}-{hi}"], textposition='inside',
            insidetextanchor='middle', textfont=dict(color='white', size=12),
            hovertemplate=f"{zona}: {lo}-{hi} ppm<extra></extra>", showlegend=False))
    fig.update_layout(barmode='overlay', height=130, margin=dict(l=8, r=8, t=6, b=28),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(size=12, color='#5b6670'),
                      xaxis=dict(title='pulsaciones por minuto (ppm)',
                                 range=[z['hr']['Z1'][0] - 8, z['hr']['Z5'][1] + 8],
                                 showgrid=False),
                      yaxis=dict(showticklabels=False))
    return fig


# Macrociclo: color y carga relativa (0-1) de cada fase, con fines pedagógicos
FASES_INFO = [
    ('Base',         '#3b82f6', 0.55, 'volumen alto · intensidad baja'),
    ('Construcción', '#22a06b', 0.80, 'volumen alto · intensidad media (sube la carga)'),
    ('Pico',         '#ef4444', 1.00, 'intensidad máxima · volumen medio'),
    ('Taper',        '#f59e0b', 0.35, 'descarga: baja el volumen, llega la frescura'),
    ('70.3',         '#7c3aed', 0.25, 'día de carrera'),
]
NOMBRE_FASE = {'BASE': 'Base', 'CONSTRUCCION': 'Construcción', 'PICO': 'Pico',
               'TAPERING': 'Taper', 'COMPETENCIA': '70.3'}


def grafico_fases(fase_actual, sem_actual, sem_tot):
    """Barra de macrociclo cuyos tramos respetan los MISMOS umbrales por semanas
    que definen la fase (base >12 sem, construcción 7-12, pico 4-6, taper 2-3,
    70.3 ≤1), de modo que el marcador SIEMPRE cae sobre la fase activa.
    El color y la altura de cada tramo representan la carga relativa de la fase.
    """
    sem_tot = max(int(sem_tot), 1)

    def frac(semanas_desde_inicio):
        return min(max(semanas_desde_inicio / sem_tot, 0.0), 1.0)

    # Fin de cada fase como fracción del plan (derivado de las semanas restantes)
    fin = {
        'Base': frac(sem_tot - 12),
        'Construcción': frac(sem_tot - 6),
        'Pico': frac(sem_tot - 3),
        'Taper': frac(sem_tot - 1),
        '70.3': 1.0,
    }
    ini = 0.0
    activa = NOMBRE_FASE.get(fase_actual, '')
    fig, ax = plt.subplots(figsize=(10, 1.5))
    for nom, color, carga, _desc in FASES_INFO:
        x1 = fin[nom]
        w = x1 - ini
        if w > 0.001:
            es_act = (nom == activa)
            # la altura de la barra refleja la carga relativa de la fase
            h = 0.25 + 0.55 * carga
            ax.barh(0, w, left=ini, height=h, color=color,
                    alpha=1.0 if es_act else 0.40,
                    edgecolor='#111111' if es_act else 'none',
                    linewidth=2.2 if es_act else 0)
            ax.text(ini + w / 2, 0, nom, ha='center', va='center', color='white',
                    fontsize=8.5, fontweight='bold')
        ini = x1
    pos = min(max(sem_actual / sem_tot, 0.0), 1.0)
    ax.plot([pos], [0.62], marker='v', color='#111111', markersize=12)
    ax.text(pos, 0.80, f'semana {sem_actual}', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 1.05)
    ax.axis('off')
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
datos = cargar_datos()
hay_datos = not datos['perfiles'].empty

# Procesa el retorno de Strava (?code=...) ANTES de pintar la interfaz
handle_strava_callback()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown('### 🔴 Mi plan 70.3')

    # --- Selector de idioma (interfaz) ---
    idioma = IDIOMAS[st.radio('🌐 Idioma / Language', list(IDIOMAS.keys()),
                              horizontal=True, key='idioma_sel')]

    # --- Fuente de datos: Strava / Garmin ---
    st.divider()
    st.markdown(f'**{t("sync_datos")}**')
    _FUENTES = ['strava', 'garmin']
    if st.session_state.get('fuente_sel') not in _FUENTES:
        st.session_state.pop('fuente_sel', None)   # limpia valores antiguos (p. ej. Local)
    fuente = st.radio(
        t('de_donde'), _FUENTES,
        format_func=lambda v: {'strava': t('opt_strava'), 'garmin': t('opt_garmin')}[v],
        key='fuente_sel')

    if fuente == 'strava':
        if not STRAVA_AVAILABLE:
            st.caption('Instala `stravalib` para sincronizar con Strava.')
        elif not (STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET):
            st.caption('Configura `strava_client_id` y `strava_client_secret` en secrets.toml.')
        else:
            if 'strava_token' not in st.session_state:
                auth_url = get_strava_auth_url()
                if auth_url:
                    st.link_button('🔗 Conectar con Strava', auth_url,
                                   use_container_width=True)
                    st.caption('Tras autorizar, tus zonas y ritmos se calcularán '
                               'con tus actividades reales de Strava.')
                    st.caption('¿Conectar con OTRA cuenta? Strava recuerda tu sesión del '
                               'navegador y reusa esa cuenta. Cierra sesión en Strava (o '
                               'revoca el acceso) y luego pulsa Conectar:')
                    cc1, cc2 = st.columns(2)
                    cc1.link_button('Cerrar sesión en Strava',
                                    'https://www.strava.com/logout', use_container_width=True)
                    cc2.link_button('Revocar acceso (apps)',
                                    'https://www.strava.com/settings/apps', use_container_width=True)
                    st.caption('Alternativa más rápida: abre la app en una ventana de incógnito.')
            else:
                _nom = fetch_strava_athlete_name(st.session_state['strava_token'])
                st.success(f'✅ Strava conectado{f" · {_nom}" if _nom else ""}')
                dias_sync = st.slider('Días a sincronizar', 30, 365, 90, 30)
                if st.button('🔄 Sincronizar actividades', use_container_width=True):
                    a = fetch_strava_activities(st.session_state['strava_token'], days=dias_sync)
                    st.session_state['actividades_strava'] = a
                    if a is not None and not a.empty:
                        st.toast(f'✅ {len(a)} actividades sincronizadas')
                    else:
                        st.warning('Strava no devolvió actividades en ese rango. '
                                   'Prueba a ampliar los días o revisa que tu cuenta '
                                   'tenga entrenamientos con pulsómetro.')
                _sa = st.session_state.get('actividades_strava')
                if _sa is not None and not _sa.empty:
                    st.caption(f'Usando {len(_sa)} actividades reales de Strava.')
                if st.button('🔓 Desconectar / cambiar de cuenta', use_container_width=True):
                    desconectar_strava()
                    st.rerun()
                st.caption('Para cambiar de atleta: desconecta aquí y, si Strava te '
                           'reconecta solo, revoca el acceso en strava.com → Ajustes → '
                           'Mis aplicaciones, o inicia sesión con la otra cuenta.')

    elif fuente == 'garmin':
        st.info('🔭 **Próximo paso del proyecto.** La sincronización directa con '
                'Garmin Connect daría acceso a Body Battery, HRV y sueño — métricas '
                'que hoy se cargan desde los .fit locales.')
        with st.expander('Vista previa de la conexión Garmin'):
            st.text_input('Email Garmin', key='garmin_email_demo',
                          placeholder='tucorreo@ejemplo.com')
            st.text_input('Contraseña', type='password', key='garmin_pass_demo')
            st.button('🔐 Conectar con Garmin', disabled=True,
                      use_container_width=True,
                      help='Deshabilitado en la demo: la API de Garmin requiere '
                           'aprobación oficial (Garmin Developer Program).')
            estado = ('`garminconnect` instalado ✅' if GARMIN_AVAILABLE
                      else '`garminconnect` no instalado')
            st.caption(f'Roadmap técnico: vía OAuth oficial o `garminconnect`. '
                       f'Estado actual: {estado}.')

    st.divider()

    # Sin selector de atleta ni carga local: la app trabaja con TUS datos (Strava)
    # o en modo demo. El perfil base es de demostración; Strava aporta el historial.
    perfil_df, fitness, acts = datos_demo()
    perfil_row = perfil_df.set_index('athlete').loc['Demo']
    pred_bici = 26.5
    atleta = 'Demo'

    perfil_tipo = st.radio(t('tu_perfil'), ['nuevo', 'experimentado'],
                           format_func=lambda v: t('nuevo') if v == 'nuevo' else t('experto'),
                           key='perfil_tipo_sel', help=GLOSARIO['Zonas Z1-Z5'])
    st.caption(f"Plan mínimo: {SEMANAS_MINIMAS[perfil_tipo]} semanas "
               f"({'6 meses' if perfil_tipo == 'nuevo' else '3 meses'})")
    sexo = st.radio(t('sexo'), ['M', 'F'], horizontal=True, key='sexo_sel',
                    format_func=lambda v: t('hombre') if v == 'M' else t('mujer'))

    st.divider()
    st.markdown(f'**{t("tu_objetivo")}**')

    # Fecha de inicio = ANCLA del plan. Se elige una vez, se FIJA (se bloquea) y no
    # cambia sola al pasar los días; así la semana avanza de verdad y queda
    # correlacionada con la fecha de competencia. Solo se cambia al reiniciar el plan.
    # Persistencia POR USUARIO (por atleta de Strava; 'demo' si no hay conexión).
    _uk = 'demo'
    if st.session_state.get('strava_token'):
        import hashlib
        _uk = 'u' + hashlib.sha1(str(st.session_state['strava_token']).encode()).hexdigest()[:10]
    if 'inicio_fijado' not in st.session_state:
        _persisted = leer_inicio_plan(_uk)
        st.session_state['inicio_plan_val'] = _persisted or date.today()
        st.session_state['inicio_fijado'] = _persisted is not None

    cI, cR = st.columns([2, 1])
    if st.session_state['inicio_fijado']:
        fecha_inicio = st.session_state['inicio_plan_val']
        cI.date_input(t('fecha_inicio'), value=fecha_inicio, disabled=True,
                      help='Fecha ancla del plan: fija y BLOQUEADA para que la semana avance '
                           'día a día y siga correlacionada con tu 70.3. Pulsa «Reiniciar plan» '
                           'para elegir otra.')
        if cR.button(t('reiniciar'), help='Desbloquear para elegir una nueva fecha de inicio'):
            st.session_state['inicio_fijado'] = False
            st.rerun()
        st.caption(f'🔒 Plan anclado al **{fecha_inicio:%d/%m/%Y}** (no se mueve solo). '
                   'Reinícialo solo si vas a empezar de cero.')
    else:
        _sel = cI.date_input(t('fecha_inicio'), value=st.session_state['inicio_plan_val'],
                             max_value=date.today(),
                             help='Elige el día en que empiezas (o empezaste) el plan. '
                                  'Al fijarla quedará bloqueada como ancla.')
        fecha_inicio = _sel
        if cR.button('🔒 Fijar', help='Bloquear esta fecha como inicio del plan'):
            st.session_state['inicio_plan_val'] = _sel
            st.session_state['inicio_fijado'] = True
            guardar_inicio_plan(_sel, _uk)
            st.rerun()
        st.caption('Elige tu fecha de inicio y pulsa **🔒 Fijar** para anclar el plan.')

    usa_fecha = st.toggle(t('tengo_fecha'), value=True)
    if usa_fecha:
        fecha_comp = st.date_input(t('cuando_703'),
                                   value=fecha_inicio + timedelta(weeks=SEMANAS_MINIMAS[perfil_tipo]))
    else:
        meses = st.slider('Meses de plan', 3, 12,
                          6 if perfil_tipo == 'nuevo' else 3)
        fecha_comp = fecha_inicio + timedelta(weeks=meses * 4)

    # Total de semanas reales del plan (límite del slider de simulación)
    sem_tot_plan = max(int(np.ceil((pd.Timestamp(fecha_comp) -
                                    pd.Timestamp(fecha_inicio)).days / 7)), 1)
    with st.expander('⚙️ Opciones avanzadas (demo)', expanded=False):
        sim_on = st.toggle('🎛️ Simular semana', value=False,
                           help='Para la presentación: fuerza en qué semana del plan estás, '
                                'sin depender de la fecha real. Recalcula fase, carga y plan.')
        semana_sim = (st.slider('Semana actual (simulada)', 1, sem_tot_plan,
                                min(6, sem_tot_plan)) if sim_on else None)

    objetivo = None
    if perfil_tipo == 'experimentado':
        st.divider()
        st.markdown('**Tu último 70.3** (orientado a marca)')
        c1, c2 = st.columns(2)
        t_nat = c1.text_input('Natación', '0:42')
        t_bici = c2.text_input('Bici', '2:52')
        t_run = c1.text_input('Carrera', '2:01')
        t_tot = c2.text_input('Total', '5:45')
        t_objetivo = st.text_input('Tiempo objetivo', '5:30')

# ---------------- Cabecera + Actualizar hoy ----------------
st.title('Mi plan 70.3')

# --- Bienvenida personalizada + frase del día ---
nombre = None
if 'strava_token' in st.session_state:
    nombre = fetch_strava_athlete_name(st.session_state['strava_token'])
if not nombre:
    nombre = (str(atleta).split()[0].capitalize()
              if atleta and str(atleta).lower() != 'demo' else 'triatleta')
FRASES_DIA = [
    "el plan no es el que más duele, sino el que sí cumples.",
    "la constancia de hoy es la marca de tu próximo 70.3.",
    "entrena por pulso, no por ego: la FC manda sobre el ritmo.",
    "cada Z2 bien hecha es un ladrillo de tu base aeróbica.",
    "descansar también es entrenar: la adaptación ocurre durmiendo.",
    "la transición se entrena; el día de la carrera no se improvisa.",
    "nada lento para correr rápido: la paciencia es velocidad diferida.",
    "el 70.3 se gana en la bici y se sufre corriendo: dosifica.",
    "mejor una semana sostenible que tres heroicas y una lesión.",
    "tu mejor sesión es la que te deja con ganas de la siguiente.",
]
frase_base = FRASES_DIA[date.today().timetuple().tm_yday % len(FRASES_DIA)]
st.markdown(f"#### {t('hola', n=nombre)}")

# Si se eligió Strava y hay actividades sincronizadas, reemplazan al historial
# local (el motor calcula FC máx, zonas y valida ritmos con datos reales)
usando_strava = False
if fuente == 'strava':
    _sa = st.session_state.get('actividades_strava')
    if _sa is not None and not _sa.empty:
        acts = _sa
        usando_strava = True
        st.success(f'📱 Datos en vivo: {len(_sa)} actividades de Strava. '
                   'Zonas y ritmos calculados con tu entrenamiento real.')

per = calcular_fase(fecha_comp, perfil_tipo, fecha_inicio, semana_sim)

# --- Frase del día (banner destacado, generada por Gemini si está disponible) ---
_frase = generar_frase_dia(nombre, per['fase'], date.today().toordinal(), frase_base)
st.markdown(
    '<div style="background:linear-gradient(100deg,#0B2A4A 0%,#13456F 55%,#1d6fa5 100%);'
    'border-radius:16px;padding:18px 22px;margin:2px 0 16px;'
    'box-shadow:0 2px 12px rgba(11,42,74,.25);">'
    '<div style="color:#8fd0ff;font-size:.72rem;font-weight:700;letter-spacing:2px;">'
    'FRASE DEL DÍA</div>'
    f'<div style="color:#fff;font-size:1.35rem;font-weight:800;line-height:1.3;margin-top:4px;">'
    f'&ldquo;{_frase}&rdquo;</div>'
    f'<div style="color:#FF8A5B;font-size:.85rem;font-weight:700;margin-top:8px;">— {nombre} 💪</div>'
    '</div>', unsafe_allow_html=True)

# En demo, las actividades reflejan las semanas ya transcurridas del plan
if not usando_strava:
    acts = acts_demo_progresivo(per, sexo, fecha_inicio)
zonas = calcular_zonas(perfil_row, acts, perfil_tipo, pred_bici)

# Con Strava, la curva de forma (CTL/ATL/TSB) también se calcula en vivo
if usando_strava:
    _fit_str = fitness_desde_actividades(acts, zonas['fc_rep'], zonas['fc_max'], sexo)
    if not _fit_str.empty:
        fitness = _fit_str

# Inferencia con los modelos entrenados (si están disponibles en salidas/modelos/)
modelos = cargar_modelos()
infer = {}
if modelos:
    infer = inferir_con_modelos(modelos, features_atleta(acts, zonas, fitness, perfil_row, sexo))
    # El modelo alimenta SOLO los objetivos de carrera (90 km / 21,1 km);
    # las zonas de entrenamiento semanal salen del historial/FC (Karvonen),
    # iguales para ambas disciplinas. Así el sexo influye de forma coherente
    # en los objetivos de carrera y la carga, no en las zonas de entreno.
    vb = infer.get('bici_pred_kmh')
    if vb and 10 < vb < 45:
        zonas['bici_obj'] = vb
    vr = infer.get('run_pred_kmh')
    if vr and 6 < vr < 20:
        zonas['run_obj'] = vr
if perfil_tipo == 'experimentado':
    objetivo = analizar_objetivo({'nat': t_nat, 'bici': t_bici, 'run': t_run,
                                  'total': t_tot}, t_objetivo,
                                 per['semanas_restantes'])
    if objetivo:
        zonas['run_obj'] = objetivo['v_run_obj']
        zonas['bici_obj'] = objetivo['v_bici_obj']
        zonas['nat_obj'] = objetivo['v_nat_obj']

# --- Tarjeta de la SESIÓN DE HOY (lo primero que el atleta quiere ver) ---
_plan_hoy_card, _ = generar_plan_semanal(perfil_tipo, per, zonas, acts, sexo)
_dia_hoy = DIAS_ES[pd.Timestamp.today().dayofweek]
_ses_hoy_card = next((s for s in _plan_hoy_card if str(s['dia']).startswith(_dia_hoy)), None)
if _ses_hoy_card:
    _emo = {'Natación': '🏊', 'Ciclismo': '🚴', 'Carrera': '🏃', 'Descanso': '😴',
            'Movilidad': '🧘', 'Competencia': '🏁'}.get(_ses_hoy_card['deporte'], '•')
    _meta_line = (f"{_ses_hoy_card['km']} km · {_ses_hoy_card['dur_min']} min"
                  if _ses_hoy_card.get('dur_min') else 'Día de descanso / recuperación')
    st.markdown(
        '<div style="background:#ffffff;border:1px solid #d8e3ec;border-left:6px solid #1d6fa5;'
        'border-radius:14px;padding:16px 18px;margin:2px 0 18px;box-shadow:0 1px 6px rgba(11,42,74,.08);">'
        '<div style="color:#1d6fa5;font-size:.72rem;font-weight:700;letter-spacing:2px;">TU SESIÓN DE HOY</div>'
        f'<div style="font-size:1.25rem;font-weight:800;color:#16202b;margin-top:2px;">'
        f'{_emo} {_ses_hoy_card["deporte"]} — {_ses_hoy_card["tipo"]}</div>'
        f'<div style="color:#1d6fa5;font-weight:700;margin:2px 0 6px;">{_meta_line}</div>'
        f'<div style="color:#3a4650;font-size:.9rem;line-height:1.35;">{_ses_hoy_card["detalle"]}</div>'
        '</div>', unsafe_allow_html=True)

# --- Estado del sistema (técnico): colapsado, para no abrumar al usuario final ---
if usando_strava:
    fuente_ico, fuente_txt = '📱', t('opt_strava').replace('📱 ', '')
elif fuente == 'garmin':
    fuente_ico, fuente_txt = '⌚', 'Garmin (roadmap)'
else:
    fuente_ico, fuente_txt = '🧪', 'Demo'
_em_estado, _em_msg = estado_modelos()
_mod_val = {'activo': 'Activos ✅', 'sin_archivos': 'No encontrados ⚠️',
            'sin_librerias': 'Sin librerías ⛔'}[_em_estado]
with st.expander('ⓘ Estado del sistema (fuente de datos y modelos)', expanded=False):
    _s1, _s2, _s3 = st.columns(3)
    _s1.metric(t('m_fuente'), f'{fuente_ico} {fuente_txt}')
    _n_act = len(acts) if (acts is not None and not getattr(acts, 'empty', True)) else 0
    _s2.metric(t('m_acts'), f'{_n_act}' if _n_act else '—',
               help='Tus actividades determinan: FC máxima y zonas, validación de ritmos, '
                    'la curva CTL/ATL/TSB, la frecuencia base del plan (sesiones/semana) y '
                    'las predicciones de los modelos. Más historial = plan más afinado.')
    _s3.metric(t('m_modelos'), _mod_val, help=_em_msg)
    if _em_estado == 'activo' and infer.get('perfil_nombre'):
        st.caption(f"🧠 Inferencia activa — perfil **{infer['perfil_nombre']}** · "
                   f"bici estimada por XGBoost **{zonas['bici_obj']:.1f} km/h**.")
    elif _em_estado == 'sin_librerias':
        st.caption(f"⛔ Modelos inactivos: {_em_msg}. (Las zonas/ritmos siguen siendo reales.)")
    elif _em_estado == 'sin_archivos':
        st.caption(f"⚠️ Modelos inactivos: {_em_msg}. Coloca la carpeta `salidas/` "
                   "(con `modelos/`) junto a `app.py`.")
    _gem_ok = inicializar_gemini() is not None
    st.caption('🤖 Explicaciones IA (Gemini): **configurado** — el «¿Por qué este plan?» y el '
               'ajuste del día se redactan en vivo.' if _gem_ok else
               '🤖 Explicaciones IA (Gemini): inactivo — se usan plantillas (revisa `gemini_api_key`).')
    _sx = 'Mujer' if sexo == 'F' else 'Hombre'
    _sx_efecto = ('y las predicciones de ritmo (modelo XGBoost)' if (_em_estado == 'activo' and infer)
                  else '(sus predicciones se verán al activar los modelos)')
    st.caption(f'🚻 Sexo: **{_sx}** — ajusta la carga de entrenamiento (TRIMP de Banister) '
               f'{_sx_efecto}. La estructura del plan y las zonas no cambian por sexo '
               '(la FC máxima es independiente del sexo, Tanaka 2001).')

if not per['plan_viable']:
    st.warning(f"⚠ Para un atleta {perfil_tipo} se recomiendan al menos "
               f"{per['semanas_minimas']} semanas de preparación y este plan tiene "
               f"{per['semanas_totales']}. Considera una carrera más lejana o ajusta "
               f"el objetivo (terminar bien, no competir).")
if objetivo:
    if objetivo['viable']:
        st.success(f"✓ Objetivo viable: {objetivo['anterior_total']} → "
                   f"{objetivo['objetivo_total']} (-{objetivo['mejora_min']:.0f} min). "
                   f"Splits objetivo — natación {objetivo['splits_obj']['nat']} · "
                   f"bici {objetivo['splits_obj']['bici']} · carrera {objetivo['splits_obj']['run']}.")
    else:
        st.warning(f"⚠ Objetivo ambicioso ({objetivo['mejora_pct']}%): el plan lo "
                   f"apunta, pero la estimación realista queda por encima. "
                   f"La mejora se concentra en la bici.")

# ---------------- Cómo estás hoy ----------------
recalc_hoy = None  # se rellena si el atleta recalcula su día
st.subheader(t('h_hoy'))

# métricas (reflejan los valores recalculados o los de referencia)
bb_hoy = float(st.session_state.get('bb', 78))
sueno_hoy = float(st.session_state.get('sueno', 7.6))
stress_hoy = float(st.session_state.get('stress', 28))
sleep_hoy = float(st.session_state.get('sleep_score', 75))
# Tarjetas HTML (rejilla responsiva: 2 en móvil, varias en escritorio), con
# estado en color y tooltip de glosario al pasar el cursor.
_GREEN, _AMBER, _RED, _NEU = '#1d9e75', '#ba7517', '#a32d2d', '#5b6670'
_bb_s, _bb_c = (('bien cargado', _GREEN) if bb_hoy >= 60
                else (('baja', _RED) if bb_hoy < 35 else ('media', _AMBER)))
_st_s, _st_c = (('bajo', _GREEN) if stress_hoy < 40
                else (('alto', _RED) if stress_hoy > 70 else ('medio', _AMBER)))
_su_s, _su_c = (('suficiente', _GREEN) if sueno_hoy >= 7
                else (('corto', _RED) if sueno_hoy < 6 else ('justo', _AMBER)))
_sueno_txt = f"{int(sueno_hoy)}h {int((sueno_hoy % 1) * 60):02d}"
_hoy_items = [
    ('Body Battery', f'{bb_hoy:.0f}', _bb_s, _bb_c, GLOSARIO['Body Battery']),
    ('Estrés', f'{stress_hoy:.0f}', _st_s, _st_c, GLOSARIO['Estrés']),
    ('HRV', '62 ms', 'estable', _NEU, GLOSARIO['HRV']),
    ('Sueño', _sueno_txt, _su_s, _su_c, GLOSARIO['Sueño']),
    ('Sleep Score', f'{sleep_hoy:.0f}', sleep_score_label(sleep_hoy), _NEU, GLOSARIO['Sleep Score']),
]
_cards = ''
for _lab, _val, _stt, _clr, _hlp in _hoy_items:
    _h = str(_hlp).replace('"', "'")
    _cards += (f'<div title="{_h}" style="background:#fff;border:1px solid #e2e8ee;'
               'border-radius:10px;padding:11px 13px;">'
               f'<div style="font-size:12px;color:#5b6670;">{_lab}</div>'
               f'<div style="font-size:22px;font-weight:800;color:#16202b;line-height:1.1;'
               f'margin-top:2px;">{_val}</div>'
               f'<div style="font-size:12px;font-weight:700;color:{_clr};margin-top:3px;">{_stt}</div></div>')
st.markdown('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));'
            f'gap:10px;margin-bottom:6px;">{_cards}</div>', unsafe_allow_html=True)

# Formulario persistente: queda visible y recuerda lo último introducido
with st.expander('⟳ Actualizar mi día — ¿cómo amaneciste hoy?',
                 expanded=st.session_state.get('recalc_done', False)):
    with st.form('hoy'):
        st.caption('Introduce tus métricas de esta mañana y pulsa «Recalcular mi día».')
        c1, c2, c3, c4 = st.columns(4)
        sueno_in = c1.number_input('Horas de sueño', 0.0, 12.0, sueno_hoy, 0.1)
        bb_in = c2.number_input('Body Battery', 0, 100, int(bb_hoy))
        stress_in = c3.number_input('Estrés', 0, 100, int(stress_hoy))
        enfermo_in = c4.checkbox('Me siento enfermo',
                                 value=bool(st.session_state.get('enfermo', False)))
        score_in = st.slider('Sleep Score / calidad del sueño (0-100)', 0, 100,
                             int(sleep_hoy),
                             help='El número de tu reloj (Garmin Sleep Score): mide la '
                                  'CALIDAD del sueño, no las horas. Mala < 60.')
        st.caption(f'Calidad estimada: **{sleep_score_label(score_in)}**')
        if st.form_submit_button('Recalcular mi día', type='primary'):
            st.session_state.update(sueno=sueno_in, bb=bb_in, stress=stress_in,
                                    enfermo=enfermo_in, sleep_score=score_in,
                                    recalc_done=True)
            st.rerun()

if st.session_state.get('recalc_done'):
    dia_hoy = DIAS_ES[pd.Timestamp.today().dayofweek]
    ses_df = datos['sesiones_plan']
    ses_hoy = (ses_df[(ses_df['athlete'] == atleta) &
                      (ses_df['dia'].str.startswith(dia_hoy))].to_dict('records')
               if hay_datos and not ses_df.empty else [])

    # Descripción de la sesión planificada de hoy (real o generada) para Gemini
    sesion_hoy_txt = ''
    if ses_hoy:
        r = ses_hoy[0]
        sesion_hoy_txt = f"{r.get('deporte', '')} · {r.get('tipo', '')}" + \
                         (f" · {r.get('dur_min')} min" if r.get('dur_min') else '')
    else:
        _plan_hoy, _ = generar_plan_semanal(perfil_tipo, per, zonas, acts, sexo)
        for s in _plan_hoy:
            if s['dia'] == dia_hoy:
                sesion_hoy_txt = f"{s['deporte']} · {s['tipo']}" + \
                                 (f" · {s['dur_min']} min" if s['dur_min'] else '')
                break

    decision, motivos, detalle = actualizar_hoy(
        ses_hoy, zonas, st.session_state['sueno'], st.session_state['bb'],
        st.session_state['stress'], st.session_state.get('enfermo', False),
        st.session_state.get('sleep_score'))

    # Explicación en vivo con Gemini, basada en la sesión planificada de hoy
    explicacion_gemini = generar_explicacion_dia(
        atleta, dia_hoy, decision, motivos, zonas, per['fase'], sesion_hoy_txt)
    if explicacion_gemini:
        detalle = explicacion_gemini

    recalc_hoy = {'dia': dia_hoy, 'decision': decision, 'detalle': detalle,
                  'sesion': sesion_hoy_txt}
    _ss = st.session_state.get('sleep_score')
    st.caption(f"✓ Recalculado con: sueño {st.session_state['sueno']:.1f} h · "
               f"Body Battery {st.session_state['bb']} · estrés {st.session_state['stress']}"
               + (f" · Sleep Score {int(_ss)} («{sleep_score_label(_ss)}»)" if _ss is not None else '')
               + (' · te sientes enfermo' if st.session_state.get('enfermo') else '')
               + (f" · sesión de hoy: {sesion_hoy_txt}" if sesion_hoy_txt else ''))
    nivel = {'MANTENER': st.success, 'SUAVIZAR HOY': st.warning,
             'DESCANSO HOY': st.error}[decision]
    nivel(f"**Recomendación recalculada — {dia_hoy} · {decision}**\n\n"
          f"Motivos: {', '.join(motivos)}.\n\n{detalle}\n\n"
          f"_Abajo, en «Tu semana», la sesión de hoy queda marcada con este ajuste._")

# ---------------- Forma y rumbo ----------------
st.subheader(t('h_forma'))
if not fitness.empty:
    if PLOTLY_OK:
        st.plotly_chart(grafico_forma_plotly(fitness), use_container_width=True,
                        config={'displayModeBar': False})
    else:
        st.pyplot(grafico_forma(fitness), use_container_width=True)
    ctl = fitness['CTL'].iloc[-1]
    atl = fitness['ATL'].iloc[-1] if 'ATL' in fitness.columns else float('nan')
    tsb = fitness['TSB'].iloc[-1] if 'TSB' in fitness.columns else ctl - atl
    if tsb > 25:
        banda, col_fn = 'demasiado fresco — riesgo de desentrenar si se mantiene', st.warning
    elif tsb > 5:
        banda, col_fn = 'fresco, a punto para competir (zona de taper)', st.success
    elif tsb >= -10:
        banda, col_fn = 'en equilibrio — carga de mantenimiento', st.info
    elif tsb >= -30:
        banda, col_fn = 'carga productiva — estás construyendo forma; esta fatiga es normal y buscada', st.info
    else:
        banda, col_fn = 'sobrecarga — fatiga alta, conviene aliviar para no arriesgar lesión', st.error
    col_fn(f"**Tu frescura hoy: TSB = {tsb:+.1f}** (referencia 0). {banda}.\n\n"
           f"CTL/Fitness {ctl:.0f} · ATL/Fatiga {atl:.0f}. El TSB es la resta (forma − fatiga): "
           f"por encima de 0 estás descansado, por debajo estás cargado.")
else:
    st.caption('Sin datos de fitness: ejecuta el notebook para generar salidas/.')
    ctl = 50.0

st.subheader(t('h_rumbo'))
# Barra de macrociclo segmentada (HTML): fase hecha · fase actual · fases futuras.
_orden_fases = ['Base', 'Construcción', 'Pico', 'Taper', '70.3']
_fase_nom = NOMBRE_FASE.get(per['fase'], 'Base')
_idx_fase = _orden_fases.index(_fase_nom) if _fase_nom in _orden_fases else 0
_segs = ''
for _i, _nom in enumerate(_orden_fases):
    if _i < _idx_fase:
        _bg, _col, _w = '#E6F1FB', '#0c447c', '500'        # ya recorrida
    elif _i == _idx_fase:
        _bg, _col, _w = '#1d6fa5', '#ffffff', '700'        # fase actual
    else:
        _bg, _col, _w = '#eef3f7', '#5b6670', '500'        # por venir
    _segs += (f'<div style="flex:1;text-align:center;font-size:12px;font-weight:{_w};'
              f'color:{_col};background:{_bg};border-radius:6px;padding:9px 2px;">{_nom}</div>')
# Barra fina de progreso por semanas dentro del plan
_prog_pct = int(min(max(per['semana_actual'] / max(per['semanas_totales'], 1), 0), 1) * 100)
st.markdown(
    f'<div style="display:flex;gap:5px;margin:2px 0 8px;">{_segs}</div>'
    '<div style="height:7px;background:#eef3f7;border-radius:6px;overflow:hidden;">'
    f'<div style="height:100%;width:{_prog_pct}%;background:#1d6fa5;border-radius:6px;"></div>'
    '</div>', unsafe_allow_html=True)
_sim_txt = (' · 🎛️ modo simulación (semana forzada para la demo)' if semana_sim
            else ' · la semana avanza sola con el calendario desde tu fecha de inicio')
st.caption(f"📅 Estás en la **semana {per['semana_actual']} de {per['semanas_totales']}** "
           f"del plan · faltan {per['semanas_restantes']} semanas para tu 70.3 "
           f"({fecha_comp:%d %b %Y}){_sim_txt}.")
_desc_fase = next((d for n, _c, _ca, d in FASES_INFO if n == NOMBRE_FASE.get(per['fase'], '')), '')
st.markdown(
    f"**Fase actual: {NOMBRE_FASE.get(per['fase'], per['fase'])}** — {_desc_fase}.  \n"
    "Carga característica de cada fase: "
    "🔵 **Base** volumen alto, intensidad baja · "
    "🟢 **Construcción** sube la intensidad · "
    "🔴 **Pico** carga máxima · "
    "🟠 **Taper** descarga para llegar fresco · "
    "🟣 **70.3** carrera.")
ramp = (CTL_OBJETIVO_70_3 - ctl) / max(per['semanas_restantes'], 1)
if per['fase'] in ('TAPERING', 'COMPETENCIA'):
    progreso = 'fase final: mantener, no construir'
elif ramp <= 0:
    progreso = 'objetivo de forma alcanzado'
elif ramp <= 3:
    progreso = 'en ruta'
elif ramp <= 6:
    progreso = 'justo: evitar semanas perdidas'
else:
    progreso = 'riesgo: forma lejos del objetivo'
st.caption(f"CTL {ctl:.0f} → objetivo {CTL_OBJETIVO_70_3} · rampa necesaria "
           f"{max(ramp, 0):+.1f}/semana · **{progreso}** · faltan "
           f"{per['semanas_restantes']} semanas ({fecha_comp:%d %b %Y})")

# ---------------- Progreso por disciplina (interactivo) ----------------
_idx_prog, _kpis = progreso_semanal(acts, fecha_inicio)
if _idx_prog is not None:
    st.subheader(f"📈 {t('h_progreso')}")
    # KPIs: ritmo actual y mejora por disciplina
    cols = st.columns(len(_kpis))
    for col, (n, (va, pct)) in zip(cols, _kpis.items()):
        unidad = '/100m' if n == 'Natación' else 'km/h'
        val = (pace_swim(va) if n == 'Natación' else f'{va:.1f} km/h')
        col.metric(n, val, f'{pct:+.0f}% vs inicio')
    st.caption('**Índice de progreso semanal** (primera semana de uso = 100), construido con tus '
               'actividades **desde que empezaste el plan**. Cada línea sube a medida que mejora tu '
               'ritmo medio de esa disciplina; pasa el cursor por la línea para ver cada semana. '
               'Una pendiente ascendente significa que el plan te acerca a tu objetivo de forma gradual.')
    st.line_chart(_idx_prog, height=280, y_label='Índice de progreso (sem 1 = 100)',
                  x_label='Semana')
    _suben = [n for n, (_, p) in _kpis.items() if p > 1]
    if _suben:
        st.success('📈 Mejorando en ' + ', '.join(_suben) +
                   '. La tendencia al alza confirma que el plan funciona: progreso gradual, '
                   'sin saltos bruscos, hacia tu 70.3.')
else:
    st.subheader(f"📈 {t('h_progreso')}")
    st.info('📊 Tu evolución por disciplina aparecerá aquí a medida que entrenes. El gráfico recoge '
            'tus actividades **desde que empezaste el plan**, así que al principio (semana 1) aún no '
            'hay datos suficientes: necesitas al menos **2 semanas** de actividad para ver la tendencia. '
            'Vuelve tras acumular algunas sesiones.')

# ---------------- Zonas ----------------
st.subheader(t('h_zonas'))
if PLOTLY_OK:
    st.plotly_chart(grafico_zonas_plotly(zonas), use_container_width=True,
                    config={'displayModeBar': False})
else:
    st.pyplot(grafico_zonas(zonas), use_container_width=True)
st.info(f"✓ **Ritmos validados con tu historial** — {zonas['fuente_z2']}.")
if infer:
    partes = []
    if infer.get('perfil_nombre'):
        partes.append(f"perfil **{infer['perfil_nombre']}** (KMeans)")
    if infer.get('bici_pred_kmh'):
        partes.append(f"bici estimada por XGBoost **{infer['bici_pred_kmh']:.1f} km/h**")
    if infer.get('run_pred_kmh'):
        partes.append(f"carrera estimada **{pace_run(infer['run_pred_kmh'])}**")
    if partes:
        st.success('🧠 **Modelos del TFM activos** — ' + ' · '.join(partes) + '.')

# Los ritmos MOSTRADOS progresan igual que en el plan (mismo factor de mejora),
# así que cambian semana a semana en vez de quedarse fijos.
_mej = factor_mejora(per, perfil_tipo)
chips = [('Z2', f"{zonas['hr']['Z2'][0]}-{zonas['hr']['Z2'][1]} ppm", False),
         ('Z4', f"{zonas['hr']['Z4'][0]}-{zonas['hr']['Z4'][1]} ppm", per['fase'] == 'BASE'),
         ('Rodaje', pace_run(zonas['run_z2'] * _mej), False),
         ('Series', pace_run(zonas['run_z4'] * _mej), per['fase'] == 'BASE'),
         ('Bici Z2', f"{zonas['bici_z2'] * _mej:.0f} km/h", False),
         ('Natación', pace_swim(zonas['nat_z2'] * _mej), False)]
st.markdown(''.join(
    f'<span class="chip{" off" if off else ""}"><div class="lab">{lab}</div>'
    f'<div class="val">{val}</div></span>' for lab, val, off in chips),
    unsafe_allow_html=True)
_mej_pico = factor_mejora({'semana_actual': per['semanas_totales'],
                           'semanas_totales': per['semanas_totales']}, perfil_tipo)
st.caption(f"📈 Los ritmos progresan con el plan: tu rodaje pasa de "
           f"**{pace_run(zonas['run_z2'] * _mej)}/km** esta semana a "
           f"**~{pace_run(zonas['run_z2'] * _mej_pico)}/km** en el pico "
           f"(mismo pulso, más rápido por adaptación aeróbica).")
if per['fase'] == 'BASE':
    st.caption('Z4 y series tachadas: en fase de Base aún no se usan.')

# ---------------- Plan semanal ----------------
EMOJI_DEP = {'Natación': '🔵', 'Ciclismo': '🟢', 'Carrera': '🟠',
             'Descanso': '⚪', 'Movilidad': '⚪', 'Competencia': '🏁'}


def _render_sesion(s, recalc=None, hoy=False):
    with st.container(border=True):
        emoji = EMOJI_DEP.get(s['deporte'], '⚪')
        pref = '🎯 **HOY** · '