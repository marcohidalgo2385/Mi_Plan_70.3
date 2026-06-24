# Mi plan 70.3 🔴🏊🚴🏃

**Sistema de análisis fisiológico y planificación adaptativa para triatlón Ironman 70.3.**
Demostrador interactivo (Streamlit) del Trabajo Fin de Máster en Big Data y Ciencia de Datos.

La app construye, día a día, un plan de entrenamiento personalizado a partir del historial
real del deportista (Strava): calcula zonas y ritmos, cuantifica la carga (CTL/ATL/TSB),
periodiza el macrociclo hacia la fecha de competencia y explica el *porqué* de cada sesión.

---

## Capturas

> Mockups de la interfaz en `docs/` — ábrelos en el navegador o sustituye estas imágenes
> por tus capturas reales de la app en marcha (`docs/captura_web.png`, `docs/captura_movil.png`).

| Web (escritorio) | Móvil |
|---|---|
| ![Vista web](docs/captura_web.png) | ![Vista móvil](docs/captura_movil.png) |

---

## Características

- **Datos reales vía Strava (OAuth):** zonas y ritmos validados con tu propio entrenamiento.
- **Zonas por % de FC máxima** (como tu reloj) y **ritmos por umbral** (lógica tipo VDOT),
  que **progresan** semana a semana.
- **Carga y forma:** TRIMP de Banister + curva CTL (fitness) / ATL (fatiga) / TSB (frescura).
- **Periodización automática:** Base → Construcción → Pico → Taper → Competencia, con
  semanas de descarga (patrón 3:1) y **bricks** progresivos (incluido mini-brick en pico).
- **Plan diario** con la sesión de hoy destacada y ritmos diferenciados por tipo de sesión.
- **Capa pedagógica:** explica TSB/CTL/ATL y el motivo de cada sesión (Google Gemini, con
  plantillas de respaldo).
- **Modelos del TFM:** XGBoost (transferencia bici↔carrera) + KMeans (perfilado), servidos
  desde `salidas/modelos/`.
- **Fecha de inicio anclada** (se fija una vez y no se mueve sola; solo cambia al reiniciar).
- Interfaz **bilingüe** (ES/EN) y optimizada para móvil.

---

## Stack tecnológico

Python · Streamlit · pandas · numpy · scikit-learn · XGBoost · matplotlib ·
stravalib (OAuth) · google-generativeai (Gemini) · Apache Parquet / SQLite.

---

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Crea `.streamlit/secrets.toml` con tus claves (ver `.streamlit/secrets.toml.ejemplo`):

```toml
gemini_api_key       = "..."
strava_client_id     = "..."
strava_client_secret = "..."
redirect_uri         = "http://localhost:8501"
```

La app funciona también en **modo demo** (sin claves), con datos sintéticos.

---

## Despliegue

- **Web (Streamlit Community Cloud):** sube este repositorio y despliega `app.py`.
  Configura los *Secrets* y pon en Strava el *Authorization Callback Domain* con tu dominio.
  Guía detallada: `README_DESPLIEGUE.md`.
- **Android (app contenedora):** envoltorio WebView que abre la web desplegada como app
  instalable. Guía y código fuente: `../Guia_App_Android_MiPlan703.md` y `../android_app/`.

---

## Estructura

```
app.py                     # Aplicación Streamlit (motor + interfaz)
requirements.txt           # Dependencias
.streamlit/
  config.toml              # Tema y ajustes de interfaz
  secrets.toml.ejemplo     # Plantilla de claves (NO subir el real)
salidas/
  triatlon.db              # Base de datos del notebook
  parquet/                 # Perfiles, actividades, fitness/fatiga
  modelos/                 # XGBoost + KMeans (.joblib)
docs/                      # Mockups y capturas
```

Los artefactos de `salidas/` los genera el notebook del TFM (entrenamiento de modelos y
preparación de datos); la app solo los consume.

---

## Datos, privacidad y límites

- **Strava (nov-2024):** sus datos solo se muestran a su dueño y no se usan para entrenar
  modelos; la app cumple esto (los modelos vienen pre-entrenados con datos abiertos y en vivo
  solo se calculan zonas/ritmos del propio usuario).
- **VO₂máx:** opcional; la API de Strava no lo expone, así que se aproxima con la lógica de
  umbral (VDOT) y se incorpora directo cuando la fuente sí lo aporta (Garmin / `.fit`).
- Demostrador académico: no sustituye el criterio de un entrenador ni de un profesional médico.

---

## Autor

Trabajo Fin de Máster — Máster en Big Data y Ciencia de Datos.
Marco Hidalgo. 2026.
