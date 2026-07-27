"""
app.py
======
Asistente Interno de Santo Pegasus Soluciones (RAG).

Responde preguntas de las personas colaboradoras sobre la documentación interna:
Manual de Onboarding, Guía de Ingeniería Back-end y Guía de Ingeniería Front-end.

En cada pregunta:
    1. Embebe la consulta (gemini-embedding-001, task_type=RETRIEVAL_QUERY)
    2. Busca los fragmentos más parecidos por similitud coseno (NumPy)
    3. Arma un prompt con esos fragmentos como CONTEXTO
    4. gemini-2.5-flash responde citando documento y página
    5. Muestra las fuentes usadas

Requisito previo: haber ejecutado `python ingest.py` (genera ./data).
"""

import os
import json

import numpy as np
import streamlit as st
from google import genai
from google.genai import types

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
EMBED_DIM = 768
TOP_K = 4                       # estándar RAG interno (Guía Back-end, Sección 5.1)

SYSTEM_PROMPT = """Eres el Asistente Interno de Santo Pegasus Soluciones, una \
empresa de tecnología. Ayudas a las personas colaboradoras a encontrar información \
en la documentación interna: el Manual de Onboarding, la Guía de Ingeniería \
Back-end y la Guía de Ingeniería Front-end. Tu conocimiento se basa EXCLUSIVAMENTE \
en los fragmentos de esos documentos que se te entregan como CONTEXTO en cada consulta.

Reglas:
1. Responde ÚNICAMENTE con base en el CONTEXTO. No inventes políticas, versiones, \
procesos ni valores que no estén ahí.
2. Cita SIEMPRE el documento y la página de cada dato. Ejemplo: \
"(Manual de Onboarding, pág. 12)".
3. Si la información no está en el contexto, dilo con claridad: "No encontré eso en \
la documentación interna disponible" y, si el contexto lo menciona, sugiere a quién \
preguntar (Tech Lead, canal de Slack, People, etc.).
4. Sé claro, directo y práctico, como un buen buddy de onboarding. Responde en español.
5. Si la pregunta es sobre un procedimiento (setup, Git, deploy), da los pasos \
concretos tal como figuran en el contexto."""

EXAMPLE_QUESTIONS = [
    "¿Cuántas aprobaciones necesita un Pull Request para hacer merge?",
    "¿Qué versión de Java y Spring Boot usamos en el back-end?",
    "¿A qué canales de Slack me uno el primer día?",
    "¿Cuál es la cobertura mínima de pruebas obligatoria?",
]


# --------------------------------------------------------------------------- #
# Carga de recursos (cacheada)
# --------------------------------------------------------------------------- #
def _get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:  # noqa: BLE001  (no hay secrets.toml en el server, p.ej. OCI)
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False)
def get_client() -> genai.Client:
    api_key = _get_api_key()
    if not api_key:
        st.error(
            "Falta la API key de Gemini. En local/Streamlit Cloud usá "
            "`.streamlit/secrets.toml` o los *Secrets*; en OCI, la variable de "
            "entorno `GEMINI_API_KEY` (ver el .env del servicio systemd)."
        )
        st.stop()
    return genai.Client(api_key=api_key)


@st.cache_data(show_spinner=False)
def load_index():
    try:
        emb = np.load("data/embeddings.npy")
        with open("data/chunks.json", encoding="utf-8") as f:
            chunks = json.load(f)
        return emb, chunks
    except FileNotFoundError:
        st.error("No encuentro el índice en `./data`. Ejecutá `python ingest.py` primero.")
        st.stop()


# --------------------------------------------------------------------------- #
# Núcleo RAG
# --------------------------------------------------------------------------- #
def embed_query(client: genai.Client, query: str) -> np.ndarray:
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM,
        ),
    )
    vec = np.array(resp.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def retrieve(query_vec: np.ndarray, emb: np.ndarray, chunks: list[dict], k: int = TOP_K):
    scores = emb @ query_vec                      # coseno (vectores normalizados)
    top_idx = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


def build_prompt(query: str, retrieved: list[tuple]) -> str:
    bloques = []
    for chunk, _score in retrieved:
        etiqueta = f"[{chunk['source']} · pág. {chunk['page']}]"
        bloques.append(f"{etiqueta}\n{chunk['text']}")
    contexto = "\n\n---\n\n".join(bloques)
    return (
        f"CONTEXTO (fragmentos de la documentación interna):\n\n{contexto}\n\n"
        f"=====================================\n\n"
        f"PREGUNTA:\n{query}\n\n"
        f"Responde citando el documento y la página de cada dato que uses."
    )


def stream_answer(client: genai.Client, prompt: str):
    stream = client.models.generate_content_stream(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    for chunk in stream:
        text = getattr(chunk, "text", None)
        if text:
            yield text


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Asistente Interno · Santo Pegasus", page_icon="📚", layout="centered")

st.title("📚 Asistente Interno · Santo Pegasus")
st.caption(
    "Preguntá en lenguaje natural sobre el **Onboarding**, la **Guía Back-end** y la "
    "**Guía Front-end**. Responde con base en la documentación y cita la fuente."
)

client = get_client()
emb, chunks = load_index()

with st.sidebar:
    st.subheader("ℹ️ Sobre el agente")
    st.markdown(
        f"- **Base documental:** {len({c['source'] for c in chunks})} documentos, "
        f"{len(chunks)} fragmentos\n"
        f"- **Recuperación:** coseno · top-{TOP_K} (estándar interno)\n"
        "- **Modelos:** `gemini-embedding-001` + `gemini-2.5-flash`\n"
        "- **Arquitectura:** Retrieval-Augmented Generation (RAG)"
    )
    st.divider()
    st.caption("Santo Pegasus Soluciones · Alura Agente (ONE / Oracle)")
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Probá con una pregunta de ejemplo:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, use_container_width=True):
            st.session_state.pending = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Fuentes consultadas"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']} · pág. {s['page']}** "
                                f"(relevancia {s['score']:.2f})\n\n> {s['preview']}")

prompt = st.chat_input("Escribí tu pregunta sobre la documentación de Santo Pegasus...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la documentación..."):
            query_vec = embed_query(client, prompt)
            retrieved = retrieve(query_vec, emb, chunks)
            full_prompt = build_prompt(prompt, retrieved)
        answer = st.write_stream(stream_answer(client, full_prompt))

        sources = [
            {"source": c["source"], "page": c["page"], "score": score,
             "preview": c["text"][:220] + "…"}
            for c, score in retrieved
        ]
        with st.expander("📎 Fuentes consultadas"):
            for s in sources:
                st.markdown(f"**{s['source']} · pág. {s['page']}** "
                            f"(relevancia {s['score']:.2f})\n\n> {s['preview']}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
