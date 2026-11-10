from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 350
OVERLAP_TOKENS = 50

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def chunk_page(text, doc_key, page):
    # Tokenize, and ask for each token's (start, end) character position in text
    # Allows us to cut the original text instead of decoding tokens
    # WHY? => Decoding would lowercase it and mangle numbers like "96,221"
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    #Offset_mapping connects token positions to character positions in the original text
    offsets = encoding["offset_mapping"] #<<- A list of tuple, each tuple represents one token's char boundary in the original text
    # len(offset) == Number of Tokens
    chunks = [] #<<- contains: {"doc_key", "page", "chunk_index", "text"}
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    for start in range(0, len(offsets), step):
        end = min(start + CHUNK_TOKENS, len(offsets)) #Ending pos of each chunk
        #Number of tokens in a chunk = end - start
        chunk_text = text[offsets[start][0] : offsets[end-1][1]]
        chunks.append( #Storing chunks with metadata
            {
                "doc_key": doc_key,
                "page": page,
                "chunk_index": len(chunks),
                "text": chunk_text,
            }
        )
        if end == len(offsets):
            break

    return chunks