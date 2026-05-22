import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODELO_EMBEDDING = "text-embedding-3-small"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("articulos")


def buscar(query: str, n_resultados: int = 3) -> list:
    response = openai_client.embeddings.create(
        model=MODELO_EMBEDDING,
        input=[query]
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_resultados
    )

    return [
        {
            "id": results["ids"][0][i],
            "titulo": results["metadatas"][0][i]["titulo"],
            "contenido": results["documents"][0][i],
            "score": 1 - results["distances"][0][i],  # similitud coseno
        }
        for i in range(len(results["ids"][0]))
    ]


if __name__ == "__main__":
    queries = [
        "¿cómo hacer una API en Python?",
        "diferencias entre frameworks de frontend",
        "cómo funciona la autenticación en aplicaciones web",
        "herramientas para trabajar con modelos de lenguaje",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        for r in buscar(query):
            print(f"  [{r['score']:.4f}] {r['titulo']}")
            print(f"           {r['contenido'][:80]}...")
