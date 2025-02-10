import faiss
import openai
import numpy as np

openai.api_key =

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response["data"][0]["embedding"])

def retrieve(query, k=3):
    index = faiss.read_index("outputs/index.faiss")

    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)

    with open("outputs/summaries.txt", "r") as f:
        summaries = f.read().split("\n\n")

    results = [summaries[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    query = input("Enter your query: ")
    results = retrieve(query)
    print("\n".join(results))