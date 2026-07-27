"""
ingest.py
=========
Convierte los PDFs de la documentación interna de Santo Pegasus (Manual de
Onboarding, Guía Back-end, Guía Front-end) en un índice vectorial que el agente
consulta en tiempo real.

Flujo:
    1. Lee todos los PDFs de la carpeta ./docs
    2. Extrae texto página por página (pypdf)
    3. Trocea el texto en fragmentos con solapamiento (chunking)
    4. Genera embeddings con gemini-embedding-001 (task_type=RETRIEVAL_DOCUMENT)
    5. Normaliza los vectores (L2) para poder usar similitud coseno
    6. Guarda ./data/embeddings.npy y ./data/chunks.json

Se ejecuta UNA sola vez. Los artefactos en ./data son los que usa la app.

Nota sobre el estándar RAG interno (Guía Back-end, Sección 5.1): la guía define
chunks de 512 tokens con overlap de 50 y K=4 en la recuperación. Aquí aproximamos
ese estándar con chunking por caracteres (~1600 chars ≈ 512 tokens; overlap 200
≈ 50 tokens). La guía también homologa 'text-embedding-004', pero Google lo
deprecó el 14-ene-2026; usamos su sucesor oficial 'gemini-embedding-001'.

Uso:
    export GEMINI_API_KEY="tu_api_key"     # PowerShell: $env:GEMINI_API_KEY="..."
    python ingest.py
"""

import os
import re
import json
import time
import glob

import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
DOCS_DIR = "docs"
DATA_DIR = "data"
EMBED_MODEL = "gemini-embedding-001"   # sucesor de text-embedding-004 (deprecado)
EMBED_DIM = 768                        # 768 = buen balance calidad/tamaño (MRL)
CHUNK_SIZE = 1600                      # ≈ 512 tokens (estándar interno)
CHUNK_OVERLAP = 200                    # ≈ 50 tokens de solapamiento
BATCH_SIZE = 100                       # fragmentos por llamada a la API
SLEEP_BETWEEN_BATCHES = 1.0            # respetar límites del free tier


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "❌ Falta la variable de entorno GEMINI_API_KEY.\n"
            "   Conseguí una gratis en https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Trocea texto en fragmentos, intentando cortar en fin de frase."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if end < len(text):
            cut = chunk.rfind(". ")
            if cut > size * 0.5:
                chunk = chunk[: cut + 1]
                end = start + cut + 1
        chunk = chunk.strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        start = max(end - overlap, start + 1)
    return chunks


def load_and_chunk() -> list[dict]:
    """Lee los PDFs de ./docs y devuelve una lista de fragmentos con metadata."""
    pdf_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    if not pdf_paths:
        raise SystemExit(f"❌ No hay PDFs en ./{DOCS_DIR}.")

    all_chunks: list[dict] = []
    for path in pdf_paths:
        source = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
        print(f"📄 Leyendo {source} ...")
        reader = PdfReader(path)
        n_before = len(all_chunks)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk in chunk_text(text):
                all_chunks.append(
                    {"id": len(all_chunks), "text": chunk, "source": source, "page": page_num}
                )
        print(f"   → {len(reader.pages)} páginas, {len(all_chunks) - n_before} fragmentos")
    print(f"✅ Total: {len(all_chunks)} fragmentos de {len(pdf_paths)} documento(s)\n")
    return all_chunks


def embed_batch(client: genai.Client, texts: list[str], task_type: str) -> list[list[float]]:
    """Embebe un lote con reintentos y backoff exponencial."""
    for attempt in range(5):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBED_DIM,
                ),
            )
            return [e.values for e in resp.embeddings]
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"   ⚠️  Error (intento {attempt + 1}/5): {exc}. Reintento en {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Falló el embedding tras 5 reintentos. Revisá tu cuota/API key.")


def l2_normalize(vectors: list[list[float]]) -> np.ndarray:
    """Normaliza a longitud 1 para que el producto punto == similitud coseno."""
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def main() -> None:
    t0 = time.time()
    client = get_client()
    chunks = load_and_chunk()

    print("🧠 Generando embeddings...")
    all_vectors: list[list[float]] = []
    texts = [c["text"] for c in chunks]
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        vecs = embed_batch(client, texts[i : i + BATCH_SIZE], task_type="RETRIEVAL_DOCUMENT")
        all_vectors.extend(vecs)
        print(f"   Lote {i // BATCH_SIZE + 1}/{total_batches} listo ({len(all_vectors)}/{len(texts)})")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    matrix = l2_normalize(all_vectors)

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "embeddings.npy"), matrix)
    with open(os.path.join(DATA_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(
        f"\n✅ Índice generado en ./{DATA_DIR}\n"
        f"   • {len(chunks)} fragmentos · {matrix.shape[1]} dims · "
        f"{matrix.nbytes / (1024 * 1024):.1f} MB · {time.time() - t0:.0f}s\n"
        f"   Ya podés correr:  streamlit run app.py 🚀"
    )


if __name__ == "__main__":
    main()
