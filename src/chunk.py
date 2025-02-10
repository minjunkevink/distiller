import re

def chunk_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in sentences:
        current_length += len(sentence.split())
        current_chunk.append(sentence)
        if current_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

if __name__ == "__main__":
    with open("outputs/extracted.txt", "r") as f:
        text = f.read()

    chunks = chunk_text(text)
    with open("outputs/chunks.txt", "w") as f:
        f.write("\n\n".join(chunks))

    print(f"Chunked into {len(chunks)} sections!")