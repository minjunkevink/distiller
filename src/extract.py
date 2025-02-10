import fitz  # PyMuPDF
import os

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

if __name__ == "__main__":
    pdf_file = "data/combinatorics-bona.pdf"  # Replace with your PDF
    text = extract_text(pdf_file)
    with open("outputs/extracted.txt", "w") as f:
        f.write(text)
    print("Text extracted successfully!")