import openai
import os

openai.api_key = 

def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message["content"]

if __name__ == "__main__":
    with open("outputs/chunks.txt", "r") as f:
        chunks = f.read().split("\n\n")

    summaries = [summarize_text(chunk) for chunk in chunks]

    with open("outputs/summaries.txt", "w") as f:
        f.write("\n\n".join(summaries))

    print("Summarization complete!")