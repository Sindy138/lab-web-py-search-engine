import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ef = DefaultEmbeddingFunction()

articulos = [
    {"id": "1", "titulo": "FastAPI vs Flask", "contenido": "FastAPI ofrece validación automática con Pydantic, documentación Swagger integrada y mejor rendimiento asíncrono que Flask."},
    {"id": "2", "titulo": "React vs Vue", "contenido": "React tiene un ecosistema más grande y es mantenido por Meta. Vue tiene una curva de aprendizaje más suave y una sintaxis más intuitiva."},
    {"id": "3", "titulo": "PostgreSQL para principiantes", "contenido": "PostgreSQL es una base de datos relacional open source con soporte para JSON, búsqueda de texto completo y extensiones como pgvector."},
    {"id": "4", "titulo": "Introducción a los LLMs", "contenido": "Los Large Language Models son redes neuronales entrenadas para predecir la siguiente palabra. GPT-4, Claude y Gemini son ejemplos populares."},
    {"id": "5", "titulo": "Despliegue con Docker", "contenido": "Docker permite empaquetar aplicaciones en contenedores que se ejecutan de forma consistente en cualquier entorno."},
    {"id": "6", "titulo": "Autenticación JWT", "contenido": "JSON Web Tokens permiten transmitir información verificada entre partes. Se usan para autenticación stateless en APIs REST."},
    {"id": "7", "titulo": "LangChain para agentes", "contenido": "LangChain simplifica la construcción de aplicaciones con LLMs, proporcionando abstracciones para cadenas, agentes y memoria."},
    {"id": "8", "titulo": "Python vs JavaScript para IA", "contenido": "Python domina el ecosistema de IA gracias a librerías como NumPy, PyTorch y HuggingFace. JavaScript tiene opciones como TensorFlow.js pero es menos maduro."},
]


def indexar():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    collection = chroma_client.get_or_create_collection(
        name="articulos",
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef
    )

    existentes = set(collection.get()["ids"])
    nuevos = [art for art in articulos if art["id"] not in existentes]

    if not nuevos:
        print("Todos los artículos ya están indexados.")
        print(f"Total en colección: {collection.count()}")
        return

    contenidos = [art["contenido"] for art in nuevos]
    ids = [art["id"] for art in nuevos]
    metadatas = [{"titulo": art["titulo"]} for art in nuevos]

    print(f"Artículos a indexar: {len(nuevos)}")

    collection.add(
        documents=contenidos,
        ids=ids,
        metadatas=metadatas
    )

    print(f"Indexados {len(nuevos)} artículos correctamente.")
    print(f"Total en colección: {collection.count()}")


if __name__ == "__main__":
    indexar()
