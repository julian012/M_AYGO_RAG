# Taller: RAG con Deep Agents (LangChain), OpenAI y Pinecone

Implementación de un *Retrieval-Augmented Generator* (RAG) en un Jupyter Notebook, basada en el tutorial
[RAG with Deep Agents](https://docs.langchain.com/oss/python/deepagents/rag) (patrón *retrieve, offload, delegate*):

1. **Parte 1:** RAG con LangChain + OpenAI usando un almacén vectorial en memoria.
2. **Parte 2:** el mismo agente conectado a **Pinecone** (persistente y escalable), con comparación de rendimiento.

## Arquitectura

```
Indexación:  documentos → splitter (chunks) → embeddings → Vector Store (memoria | Pinecone)
Consulta:    pregunta → agente orquestador → tool search_documentation (top-k → archivos /retrieved/*)
             → subagentes chunk-analyst (1 por archivo, en paralelo) → síntesis con fuentes
```

La herramienta devuelve rutas de archivo (no texto) para mantener pequeño el contexto del orquestador.
Pasar de la Parte 1 a la 2 solo cambia el `VectorStore` que recibe `build_agent()`
(`InMemoryVectorStore` → `PineconeVectorStore`). El notebook incluye el diagrama completo, pseudocódigo y la explicación de la integración con Pinecone.

## Estructura

```
.
├── rag_workshop.ipynb      # Notebook con ambas partes
├── data/knowledge_base.txt # Base de conocimiento local (se suma a la documentación de LangChain)
├── requirements.txt
├── .env.example            # Plantilla de claves
└── README.md
```

## Requisitos

- Python 3.10+
- API key de chat: [OpenAI](https://platform.openai.com/api-keys) **o** un proveedor compatible con OpenAI (ver abajo)
- API key de [Pinecone](https://app.pinecone.io) (el plan gratuito basta)

## Instalación

```bash
git clone https://github.com/julian012/M_AYGO_RAG.git
cd M_AYGO_RAG

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # luego edita .env con tus claves
```

## Configuración de modelos

| Variable | Descripción |
|---|---|
| `OPENAI_API_KEY` | Clave de OpenAI (chat y, por defecto, embeddings) |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Opcionales: usar otro proveedor compatible con OpenAI para el chat |
| `EMBEDDINGS_PROVIDER` | `openai` (`text-embedding-3-small`) o `local` (HuggingFace, sin API key) |
| `PINECONE_API_KEY` | Clave de Pinecone |

**Usar OpenCode Go u otro proveedor compatible:** configura `LLM_API_KEY`, `LLM_BASE_URL=https://opencode.ai/zen/go/v1` y `LLM_MODEL`
(un id de `/v1/models`). Estos servicios suelen no ofrecer embeddings, por eso usa `EMBEDDINGS_PROVIDER=local`.
El índice de Pinecone toma su dimensión automáticamente del modelo de embeddings (si cambias de modelo, usa otro `PINECONE_INDEX_NAME`).

## Ejecución

```bash
jupyter notebook rag_workshop.ipynb
```

Ejecuta las celdas en orden (*Run All*). La Parte 2 crea automáticamente un índice serverless
(métrica coseno, AWS `us-east-1`, configurable en `.env`). La última celda (comentada) elimina el índice.

## Notas

- Nunca subas tu `.env` (ya está en `.gitignore`).
- El notebook descarga páginas de docs.langchain.com; sin internet usa solo `data/knowledge_base.txt`.
- Los tiempos de la comparación varían según red y región.
- RAG es vulnerable a *indirect prompt injection*; los prompts lo mitigan pero no lo eliminan.
