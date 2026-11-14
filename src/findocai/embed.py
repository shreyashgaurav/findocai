# Converting from text to numerical values
from sentence_transformers import SentenceTransformer


from findocai.chunk import MODEL_NAME #MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(MODEL_NAME) #130 MB downloaded
def embed(texts):
    vector = model.encode(
        texts,
        normalize_embeddings=True, #normalizes each embedding vector to have a length (L2 norm) of 1
        batch_size=32, #model processes up to 32 texts per batch
        show_progress_bar=True,
    )
    return vector.tolist() #numpy array to python list
