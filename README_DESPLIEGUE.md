# Desplegar "Mi plan 70.3" en línea (Streamlit Community Cloud)

Guía para poner la app en internet y que tus sujetos de prueba la usen desde
sus propios perfiles de Strava. Es **gratis**.

> Sube a GitHub **solo el contenido de esta carpeta `deploy_online/`** (no el
> resto del proyecto). Ya está limpia: app, requisitos, datos/modelos y config.

---

## 1. Crear el repositorio en GitHub
1. Entra en https://github.com y crea una cuenta (si no tienes).
2. Botón **New repository** → nombre, p. ej. `mi-plan-70-3` → **Private** (recomendado) → **Create**.
3. En la página del repo: **Add file → Upload files** y arrastra **todo el contenido de `deploy_online/`**
   (app.py, requirements.txt, las carpetas `salidas/` y `.streamlit/`, `.gitignore`).
   - ⚠️ Verifica que **NO** subes `secrets.toml` (solo el `.ejemplo`). El `.gitignore` ya lo protege.
4. **Commit changes**.

## 2. Desplegar en Streamlit Community Cloud
1. Entra en https://share.streamlit.io e inicia sesión **con tu cuenta de GitHub**.
2. **Create app → Deploy a public app from GitHub** (o "from existing repo").
3. Selecciona: tu repositorio · rama `main` · archivo principal `app.py`.
4. **Deploy**. Tardará 1–3 min en instalar dependencias.
5. Copia la URL final, será algo como `https://mi-plan-70-3.streamlit.app`.

## 3. Cargar los secretos (claves) en la nube
En la app desplegada: menú **⋮ (arriba derecha) → Settings → Secrets**. Pega:

```toml
gemini_api_key       = "TU_CLAVE_GEMINI"
strava_client_id     = "TU_CLIENT_ID_STRAVA"
strava_client_secret = "TU_CLIENT_SECRET_STRAVA"
redirect_uri         = "https://mi-plan-70-3.streamlit.app"
```

(Usa exactamente la URL que te dio Streamlit en el paso 2, **sin barra al final**.)
Guarda → la app se reinicia sola.

## 4. Ajustar Strava para que el login funcione online
1. Entra en https://www.strava.com/settings/api (tu aplicación de Strava).
2. En **Authorization Callback Domain** pon **solo el dominio**, sin `https://`:
   ```
   mi-plan-70-3.streamlit.app
   ```
3. Guarda. (En local seguías usando `localhost`; ahora apunta al dominio público.)

## 5. Probar y compartir
- Abre la URL, pulsa **Conectar con Strava**, autoriza → debe sincronizar tus actividades.
- Comparte la URL con tus sujetos de prueba: **cada uno entra con su propio Strava**
  y ve su propio plan. (Para limitar quién entra: Settings → Sharing → invitar por email.)

---

## Avisos importantes
- **Límite de atletas de Strava:** una app de Strava nueva tiene un límite de
  deportistas conectados hasta solicitar ampliación. Para un grupo pequeño de
  testers suele bastar; si algún tester no puede conectar, revisa el límite en
  el panel de tu app de Strava y solicita aumento.
- **Política Strava (nov-2024):** los datos de su API solo se muestran a su dueño
  (cada tester ve lo suyo) y no se usan para entrenar modelos. La app ya cumple:
  los modelos vienen pre-entrenados con datos abiertos; en vivo solo se calculan
  zonas/ritmos del propio usuario.
- **Modo demo:** si un tester no quiere conectar Strava, la app funciona igual en
  modo demostración.
- **Datos persistentes:** la fecha de inicio del plan se guarda por sesión; en la
  nube cada visita puede reiniciarla. Para feedback de varios días, pídeles que
  fijen su fecha de inicio cada vez (o lo dejamos persistente por usuario si lo necesitas).
