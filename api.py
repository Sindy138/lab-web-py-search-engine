from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    ef = DefaultEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = chroma_client.get_collection("articulos", embedding_function=ef)
    except Exception:
        raise RuntimeError("Colección 'articulos' no encontrada. Ejecuta indexar.py primero.")
    yield


app = FastAPI(title="Motor de Búsqueda Semántica", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/buscar")
async def buscar(q: str = Query(..., min_length=1), n: int = Query(3, ge=1, le=8)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="La query no puede estar vacía")

    results = collection.query(
        query_texts=[q],
        n_results=n
    )

    items = [
        {
            "id": results["ids"][0][i],
            "titulo": results["metadatas"][0][i]["titulo"],
            "contenido": results["documents"][0][i],
            "score": round(1 - results["distances"][0][i], 4),
        }
        for i in range(len(results["ids"][0]))
    ]

    return {"query": q, "resultados": items}
