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
st.set_page_config(page_title='Mi plan 70.3', page_icon='🔴', layout='wide')

st.markdown("""
<style>
  .chip {display:inline-block; background:#fdf6ec; border:1px solid #f1e3c8;
         border-radius:12px; padding:10px 16px; margin:4px; text-align:center;}
  .chip .lab {color:#b45309; font-weight:700; font-size:0.8rem;}
  .chip .val {color:#262626; font-weight:700; font-size:1.0rem;}
  .chip.off {background:#f4f3ef; border-color:#dddbd4;}
  .chip.off .lab, .chip.off .val {color:#8a877f; text-decoration:line-through;}
  .badge {display:inline-block; border-radius:14px; padding:4px 14px;
          font-weight:700; font-size:0.85rem;}
  div[data-testid="stMetric"] {background:#ffffff; border:1px solid #e8e6e1;
          border-radius:12px; padding:14px;}
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


def leer_inicio_plan():
    """Lee la fecha de inicio del plan persistida (o None si no existe)."""
    try:
        import json
        with open(ESTADO_PLAN_PATH, encoding='utf-8') as f:
            return date.fromisoformat(json.load(f)['fecha_inicio'])
    except Exception:
        return None


def guardar_inicio_plan(d):
    """Persiste la fecha de inicio del plan para que no se reinicie cada día."""
    try:
        import json
        with open(ESTADO_PLAN_PATH, 'w', encoding='utf-8') as f:
            json.dump({'fecha_inicio': pd.Timestamp(d).date().isoformat()}, f)
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
        cyc = acts[acts['sport'] == 'CYCLING'] if 'sport' in cols else acts.iloc[