import openai
import faiss
import numpy as np

openai.api_key = 

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response["data"][0]["embedding"])

if __name__ == "__main__":
    with open("outputs/summaries.txt", "r") as f:
        summaries = f.read().split("\n\n")

    embeddings = np.array([get_embedding(summary) for summary in summaries])

    # Store embeddings in FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "outputs/index.faiss")

    print("Embeddings created and stored!")