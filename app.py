from flask import Flask, render_template, request
from docx import Document
import markdown2
import os
from typing import List
import openai
import tiktoken
import numpy as np
from config import OPENAI_API_KEY
import re
from googletrans import Translator
import fitz

openai.api_key = OPENAI_API_KEY
app = Flask(__name__)
translator = Translator()

def load_docs_from_folder(folder_path="doc") -> List[dict]:
    docs = []
    if not os.path.exists(folder_path):
        return docs
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.endswith(".docx"):
                doc = Document(file_path)
                full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                docs.append({"name": filename, "content": full_text})
            elif filename.endswith(".pdf"):
                with fitz.open(file_path) as pdf:
                    text = ""
                    for page in pdf:
                        text += page.get_text()
                docs.append({"name": filename, "content": text})
        except Exception as e:
            continue  # skip file if error
    return docs

def chunk_text(text: str, max_tokens=400) -> List[str]:
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    sentences = text.split('. ')
    chunks, chunk = [], ""
    for s in sentences:
        if len(tokenizer.encode(chunk + s)) < max_tokens:
            chunk += s + ". "
        else:
            chunks.append(chunk.strip())
            chunk = s + ". "
    chunks.append(chunk.strip())
    return chunks

def get_relevant_chunks(question, all_docs):
    chunks = []
    for doc in all_docs:
        for c in chunk_text(doc["content"]):
            chunks.append({"text": c, "source": doc["name"]})

    # Validate and clean text
    texts = [c['text'] for c in chunks if isinstance(c['text'], str) and c['text'].strip() != ""]

    if not texts:
        return [], "No valid text found for embedding."

    try:
        chunk_embeddings = openai.Embedding.create(input=texts, model="text-embedding-ada-002")["data"]
        question_embedding = openai.Embedding.create(input=[question], model="text-embedding-ada-002")["data"][0]["embedding"]
    except Exception as e:
        return [], f"Embedding error: {e}"

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scored_chunks = []
    valid_chunks = [c for c in chunks if isinstance(c['text'], str) and c['text'].strip() != ""]
    for chunk, emb in zip(valid_chunks, chunk_embeddings):
        score = cosine_similarity(question_embedding, emb["embedding"])
        scored_chunks.append((score, chunk))

    top_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)[:5]
    return [c[1] for c in top_chunks], None

def generate_answer(question: str, context_chunks: List[dict]):
    context_text = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in context_chunks])
    system_prompt = (
        "You are a legal contract assistant for family private trusts. "
        "Answer the user's question strictly using the provided context. "
        "IMPORTANT: Always provide your answer in English language only, regardless of the language of the question or context. "
        "If the user requests a tabular format, return the answer as a markdown table. "
        "Cite filenames where applicable. Be precise and helpful."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
            ],
            temperature=0.3
        )
        english = response['choices'][0]['message']['content'].strip()
        english_cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', english)
        
        # If the response is still in Hindi, translate it to English
        try:
            detected_lang = translator.detect(english_cleaned).lang
            if detected_lang == 'hi':
                english_cleaned = translator.translate(english_cleaned, src='hi', dest='en').text
        except:
            pass  # If detection fails, proceed with original text
        
        # Add document references
        sources = list(set([chunk['source'] for chunk in context_chunks]))
        english_source_text = f"\n\n(Reference: {', '.join(sources)})"
        hindi_source_text = f"\n\n(संदर्भ: {', '.join(sources)})"
        
        hindi = translator.translate(english_cleaned, dest='hi').text
        return english_cleaned + english_source_text, hindi + hindi_source_text
    except Exception as e:
        return f"Answer generation failed: {e}", ""

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    html_answer = ""
    question = ""
    error = ""
    docs = load_docs_from_folder("doc")
    num_docs = len(docs)

    if request.method == "POST":
        question = request.form.get("question")
        translate_option = request.form.get("translate_option", "both")  # NEW: Get translation option
        
        if question:
            top_chunks, error = get_relevant_chunks(question, docs)
            if not error:
                english_answer, hindi_answer = generate_answer(question, top_chunks)
                html_answer_en = markdown2.markdown(english_answer)
                html_answer_hi = markdown2.markdown(hindi_answer)
                
                # NEW: Handle different translation options
                if translate_option == "english_only":
                    html_answer = f"<h3>English:</h3>{html_answer_en}"
                elif translate_option == "hindi_to_english":
                    # Translate Hindi back to English for comparison
                    hindi_to_english = translator.translate(hindi_answer, src='hi', dest='en').text
                    html_answer_hte = markdown2.markdown(hindi_to_english)
                    html_answer = f"<h3>हिंदी:</h3>{html_answer_hi}<hr><h3>Hindi to English Translation:</h3>{html_answer_hte}"
                else:  # both (default)
                    html_answer = f"<h3>English:</h3>{html_answer_en}<hr><h3>हिंदी:</h3>{html_answer_hi}"
        else:
            error = "Please enter a question."

    return render_template("index.html", answer=html_answer, question=question, num_docs=num_docs, error=error)

if __name__ == "__main__":
    app.run(debug=True)