from pathlib import Path
import chromadb #the storage and retrieval componen
import yaml #Python does not understand yaml by default. pyYaml changes it to py objects


#have built this in other files
from findocai.chunk import chunk_page #splits a page into chunks
from findocai.embed import embed #converts text into vectors
from findocai.parse import parse_filing, parse_pdf #parses HTML filings, parse PDF documents

#config constants
MANIFEST = Path("corpus/manifest.yaml") #doc manifest
CHROMA_PATH = "data/index/chroma" #persistent chromaDB for files
COLLECTION = "text_chunk" #will ho;d the financial doc's chunks

PARSERS = {"pdf": parse_pdf, "page_breaks": parse_filing}


def load_manifest():
    with MANIFEST.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["documents"] #read and parse the yaml to py data structure

def chunks_for(doc): #doc is a dict- contains page_method, local_path, ....
    #parse one document, chunk every page and attach its metadata
    parse = PARSERS[doc["page_method"]] #appropriate parser (pdf or filing)
    chunks = []

    for page in parse(doc["local_path"]):
        for chunk in chunk_page(page["text"], doc["doc_key"], page["page"]):
            chunk.update(
                company = doc["company"],
                form_type=doc["form_type"],
                fiscal_year = doc["fiscal_year"],
                fiscal_quarter=doc["fiscal_quarter"],
                period_end_date = str(doc["period_end_date"]),
            )
            chunks.append(chunk)
    return chunks

def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    #When we run ingestion then previous chunks are deleted.
    #TODO: Rebuilding shoukd be an explicit choice rather than automatically
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"}, #compare vectirs by cosine similarity
        # Cosine similarity: higher means more similar.
        # Cosine distance: lower means more similar => Used by Chroma
    )

    for doc in load_manifest():
        chunks = chunks_for(doc)
        texts = [c.pop("text") for c in chunks]
        collection.add(
            #unique ID per chunk
            ids=[f"{c['doc_key']}-p{c['page']}-c{c['chunk_index']}" for c in chunks],
            documents=texts,  # the text itself, returned with search results
            embeddings=embed(texts),  # our own vectors (choice 1A)
            metadatas=chunks,  # document, page, company, period
        )
        print(f"{doc['doc_key']}: {len(chunks)} chunks")

    print("total in index:", collection.count())


# Run main() only when this file is executed directly, not when imported.
if __name__ == "__main__":
    main()