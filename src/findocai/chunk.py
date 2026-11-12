from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 350
OVERLAP_TOKENS = 50

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def count_token(text):
    # returns the number of BGE tokens in the text
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_long_line(line):
    # For exceptional cases where number of tokens in a sentence > CHUNK_TOKEN then this fun is used
    offsets = tokenizer(line, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    pieces = []
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    for start in range(0, len(offsets), len(offsets)):
        end = min(start + CHUNK_TOKENS, len(offsets))
        pieces.append(line[offsets[start][0] : offsets[end - 1][1]])
        if end == len(offsets):
            break
    return pieces  # return pieces of a high token count sentence


def chunk_page(text, doc_key, page):
    # Processes the page line by line
    texts = []  # Stores completed chunks
    current = []  # Stores lines of the chunk currently being built
    current_token = 0  # Tracks how many tokens in the current chunk

    for line in text.split("\n"):  # iterate through each line
        if not line.strip():  # Skips empty lines
            continue
        n = count_token(line)  # Count no. of  tokens in the current line

        # case1: line too big
        if n > CHUNK_TOKENS:
            if current:  # Close the current chunk in progress
                texts.append("\n".join(current))

            texts.extend(split_long_line(line))  # cut the line by token
            current, current_token = [], 0  # start fresh after it
            continue

        # case2: the line doesnt fit in the current chunk -> close it
        if current and current_token + n > CHUNK_TOKENS:
            texts.append("\n".join(current))

            carry, carry_tokens = (
                [],
                0,
            )  # preserving some of the previous lines for context
            for prev in reversed(
                current
            ):  # reverse because we want to overlap with the most recent lines
                t = count_token(prev)
                if (
                    carry_tokens + t > OVERLAP_TOKENS
                    or carry_tokens + t + n > CHUNK_TOKENS
                ):
                    break
                carry.insert(0, prev)  # insert at the front to keep line order
                carry_tokens += t
            current, current_token = carry, carry_tokens

        # case3: Always, unless Case1: Add the line to the current chubnk
        current.append(line)
        current_token += n

    if current:
        texts.append("\n".join(current))

    return [
        {"doc_key": doc_key, "page": page, "chunk_index": i, "text": t}
        for i, t in enumerate(texts)
    ]


# def chunk_page(text, doc_key, page):
#     # Tokenize, and ask for each token's (start, end) character position in text
#     # Allows us to cut the original text instead of decoding tokens
#     # WHY? => Decoding would lowercase it and mangle numbers like "96,221"
#     encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
#     #Offset_mapping connects token positions to character positions in the original text
#     offsets = encoding["offset_mapping"] #<<- A list of tuple, each tuple represents one token's char boundary in the original text
#     # len(offset) == Number of Tokens
#     chunks = [] #<<- contains: {"doc_key", "page", "chunk_index", "text"}
#     step = CHUNK_TOKENS - OVERLAP_TOKENS
#     for start in range(0, len(offsets), step):
#         end = min(start + CHUNK_TOKENS, len(offsets)) #Ending pos of each chunk
#         #Number of tokens in a chunk = end - start
#         chunk_text = text[offsets[start][0] : offsets[end-1][1]]
#         chunks.append( #Storing chunks with metadata
#             {
#                 "doc_key": doc_key,
#                 "page": page,
#                 "chunk_index": len(chunks),
#                 "text": chunk_text,
#             }
#         )
#         if end == len(offsets):
#             break

#     return chunks
