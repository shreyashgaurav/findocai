import chromadb

from findocai.embed import embed
from findocai.ingest import CHROMA_PATH, COLLECTION

# From bge's model card I saw this. This will give us slightly improved matching
# instruction prefix
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Connecting to the chroma db
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION)  # gives text_chunk


def retrieve(question, top_k=10):
    query_vector = embed([QUERY_INSTRUCTION + question])[
        0
    ]  # since only one question is emvedded -> [0]
    # Find the top_k stored document embeddings that are closest to the question embedding.
    result = collection.query(query_embeddings=[query_vector], n_results=top_k)

    hits = []
    for text, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
        strict=True,
    ):
        hits.append({**meta, "text": text, "score": round(1 - distance, 3)})
    return hits
