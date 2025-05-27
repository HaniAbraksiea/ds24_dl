import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader

# === Setup ===
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))
embedding_model = "models/embedding-001"
model = genai.GenerativeModel("gemini-1.5-flash")

# === Läs PDF och skapa chunks ===
reader = PdfReader("Magisteruppsats.pdf")
text = "".join([page.extract_text() for page in reader.pages])
chunk_size = 1000
overlap = 200
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

# === Ladda eller skapa embeddings ===
if os.path.exists("embeddings.npy"):
    print("🔁 Laddar tidigare sparade embeddings...")
    chunk_embeddings = np.load("embeddings.npy", allow_pickle=True)
else:
    print("⚙️ Skapar nya embeddings... detta kan ta några minuter...")
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        print(f" → Embedding {i+1}/{len(chunks)}")
        emb = genai.embed_content(
            model=embedding_model,
            content=chunk,
            task_type="retrieval_document"
        )["embedding"]
        chunk_embeddings.append(emb)
    np.save("embeddings.npy", chunk_embeddings)
    print("✅ Embeddings sparade till embeddings.npy")

# === Semantisk sökning ===
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(query, chunks, embeddings, k=3):
    query_embedding = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="retrieval_query"
    )["embedding"]
    
    scores = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(embeddings)]
    top_chunks = [chunks[i] for i, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]
    return "\n".join(top_chunks)

# === Generera svar ===
def generate_answer(query):
    context = semantic_search(query, chunks, chunk_embeddings)
    prompt = f"""
Du är en expert på att analysera akademiska texter. Du ska svara tydligt och korrekt på frågan nedan, endast med hjälp av den medföljande kontexten.

KONTEKST:
{context}

FRÅGA:
{query}

INSTRUKTION:
- Använd så mycket relevant information som möjligt från kontexten.
- Om svaret inte tydligt finns i kontexten, svara exakt med: "Det vet jag inte."
- Gissa inte.
"""
    response = model.generate_content(prompt)
    return response.text

# === Terminalchatt ===
print("🤖 Chattbot för Magisteruppsats. Skriv 'q' för att avsluta.")
while True:
    frågan = input("Du: ")
    if frågan.lower() in ["q", "quit", "exit"]:
        break
    svar = generate_answer(frågan)
    print("Bot:", svar)


    # python main.py