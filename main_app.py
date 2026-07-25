"""
Bot de Cultura General e Historia Mundial — Groq + Llama 3.3 70B
-------------------------------------------------------------------
Chatbot conversacional para hacer y responder preguntas de cultura
general e historia mundial, usando la API de Groq.

Ejecutar con:
    streamlit run main_app.py
"""

import streamlit as st
from groq import Groq, APIError, AuthenticationError

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Trivia Mundial · Groq + Llama 3.3",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODELOS_DISPONIBLES = {
    "Llama 3.3 70B Versatile (recomendado por el usuario)": "llama-3.3-70b-versatile",
    "GPT-OSS 120B (alternativa sugerida por Groq)": "openai/gpt-oss-120b",
    "Qwen 3.6 27B (alternativa sugerida por Groq)": "qwen/qwen3.6-27b",
}

SYSTEM_PROMPT = """Eres un profesor experto en cultura general e historia mundial, ameno y riguroso.
Reglas de comportamiento:
- Responde de forma clara, correcta y concisa, con contexto histórico relevante cuando aporte valor.
- Si el usuario te pide que le hagas preguntas (modo trivia), formula UNA pregunta a la vez de cultura
  general o historia mundial, espera su respuesta, dile si acertó o no, da la respuesta correcta con una
  breve explicación, y luego pregunta si quiere continuar.
- Varía la dificultad y las temáticas: historia antigua, edad media, historia moderna y contemporánea,
  geografía, arte, ciencia, mitología, personajes históricos, efemérides, etc.
- Si no sabes algo con certeza, dilo honestamente en vez de inventar datos.
- Sé cercano y motivador, como un buen profesor de trivia.
"""

# =========================================================
# CSS LIGERO
# =========================================================
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .stat-pill {
        display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
        background: #1E3A2B; color: #F5EFE0; font-weight: 600; margin-right: 0.5rem;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR: CONFIGURACIÓN
# =========================================================
st.sidebar.title("🌍 Trivia Mundial")
st.sidebar.caption("Cultura general e historia mundial, potenciado por Groq.")

api_key = st.sidebar.text_input(
    "GROQ API Key",
    type="password",
    placeholder="gsk_...",
    help="Tu API key no se guarda ni se envía a ningún lado distinto de Groq.",
)

modelo_label = st.sidebar.selectbox("Modelo", list(MODELOS_DISPONIBLES.keys()), index=0)
modelo_id = MODELOS_DISPONIBLES[modelo_label]

with st.sidebar.expander("⚠️ Nota sobre disponibilidad del modelo"):
    st.write(
        "Groq anunció el 17 de junio de 2026 la baja de `llama-3.3-70b-versatile` "
        "para cuentas gratuitas/developer, recomendando migrar a `openai/gpt-oss-120b` "
        "o `qwen/qwen3.6-27b`. Si tu cuenta no es enterprise y el modelo falla, prueba "
        "una de las alternativas en el selector de arriba."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Parámetros")
temperatura = st.sidebar.slider("Temperatura (creatividad)", 0.0, 1.5, 0.7, 0.1)
max_tokens = st.sidebar.slider("Máx. tokens por respuesta", 128, 2048, 600, 64)

st.sidebar.markdown("---")
modo_trivia = st.sidebar.toggle("🎮 Modo Trivia (con puntaje)", value=False)

if "score" not in st.session_state:
    st.session_state.score = {"aciertos": 0, "intentos": 0}

if modo_trivia:
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Aciertos", st.session_state.score["aciertos"])
    c2.metric("Preguntas", st.session_state.score["intentos"])
    if st.sidebar.button("🔄 Reiniciar puntaje"):
        st.session_state.score = {"aciertos": 0, "intentos": 0}
        st.rerun()

if st.sidebar.button("🗑️ Limpiar conversación"):
    st.session_state.messages = []
    st.session_state.score = {"aciertos": 0, "intentos": 0}
    st.rerun()

# =========================================================
# ESTADO DE LA CONVERSACIÓN
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🌍 Trivia Mundial")
st.caption("Pregúntame lo que quieras de cultura general e historia mundial, o pídeme que te haga preguntas en Modo Trivia.")

if not api_key:
    st.info("👈 Ingresa tu GROQ API Key en la barra lateral para comenzar a chatear.")
    st.stop()

try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"❌ No se pudo inicializar el cliente de Groq: {e}")
    st.stop()

# Mensaje inicial de bienvenida (solo una vez)
if not st.session_state.messages:
    saludo = (
        "¡Bienvenido/a! 🌍📜 Soy tu profesor de cultura general e historia mundial. "
        "Puedes preguntarme lo que quieras, o decirme **'hazme una pregunta de trivia'** "
        "para empezar a jugar (activa el Modo Trivia en la barra lateral para llevar el puntaje)."
    )
    st.session_state.messages.append({"role": "assistant", "content": saludo})

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================================
# ENTRADA DEL USUARIO Y LLAMADA A LA API
# =========================================================
prompt_usuario = st.chat_input("Escribe tu pregunta o pide una trivia...")

if prompt_usuario:
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    # Detección simple de respuesta a trivia para llevar el puntaje
    if modo_trivia:
        st.session_state.score["intentos"] += 1

    system_msg = SYSTEM_PROMPT
    if modo_trivia:
        system_msg += "\nEl usuario tiene el Modo Trivia activado: siempre formula una pregunta nueva al final de tu respuesta."

    mensajes_api = [{"role": "system", "content": system_msg}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        respuesta_completa = ""
        try:
            stream = client.chat.completions.create(
                model=modelo_id,
                messages=mensajes_api,
                temperature=temperatura,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                respuesta_completa += delta
                placeholder.markdown(respuesta_completa + "▌")
            placeholder.markdown(respuesta_completa)

        except AuthenticationError:
            respuesta_completa = ""
            placeholder.error("❌ API Key inválida o sin permisos. Verifica tu GROQ API Key en la barra lateral.")
        except APIError as e:
            respuesta_completa = ""
            placeholder.error(
                f"❌ Error de la API de Groq: {e}. "
                f"Si el modelo `{modelo_id}` fue dado de baja para tu cuenta, prueba otro modelo en la barra lateral."
            )
        except Exception as e:
            respuesta_completa = ""
            placeholder.error(f"❌ Ocurrió un error inesperado: {e}")

    if respuesta_completa:
        st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})

        # Detección simple de acierto para el marcador (heurística por palabras clave)
        if modo_trivia:
            texto_lower = respuesta_completa.lower()
            if any(p in texto_lower for p in ["¡correcto!", "correcto!", "¡acertaste", "respuesta correcta: sí"]):
                st.session_state.score["aciertos"] += 1
                st.rerun()

st.markdown("---")
st.caption("🌍 Trivia Mundial · Impulsado por Groq (LPU) + Llama 3.3 70B · Streamlit")
