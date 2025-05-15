import os
import chromadb
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
import google.generativeai as genai


os.environ["USE_TF"] = "0"
CHROMA_DIR = "./chroma_store"
CHROMA_COLLECTION_NAME = "legal_sections"
GEMINI_API_KEY = "" #api key

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
driver = GraphDatabase.driver("bolt://localhost:7689", auth=("neo4j", "neo4j123"))

tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def rewrite_query_with_gemini(original_query):
    prompt = f"""You are an expert legal assistant trained in Indian court cases.
Your task is to refine the following legal query to use appropriate terminology, without exceeding 40 words.
TOTALLY Avoid expanding or changing the query unnecessarily. DO NOT ADD EXTRA INFORMATION ON YOUR OWN. Only improve legal specificity and language of the query, IF NEEDED.

Query: "{original_query}"

Refined Query:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[WARN] Gemini failed to rewrite query: {e}")
        return original_query

def summarize_with_gemini(text):
    prompt = f"""
You are a legal summarizer specializing in Indian court rulings. 
You are given a section of legal case content and a legal query.
Summarize the section in max 20 words at max, ensuring the summary is never longer than the original content. In case of extremely small content, avoid summarizing it. For long content, summarise it to 1-2 sentences.
Be to the point and mention why this content is or is not relevant to the query.
Use appropriate legal language and avoid hallucination.

Case Content:
""
{text}
""
Summary:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if len(response.text.strip()) <= len(text) else text
    except Exception as e:
        print(f"[WARN] Gemini summarization failed: {e}")
        return text

def retrieve_semantic_cases(query, section_type=None, top_k=10):
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    if not results["documents"] or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    candidates = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        label = meta.get("label", "").lower()
        if label in section_type or section_type in ["none", "None", "all", "All", "any", "Any"]:
            candidates.append({
                "case_id": meta.get("case_id", ""),
                "score": 1 / (1 + dist),
                "title": meta.get("title", ""),
                "label": meta.get("label", ""),
                "section_id": meta.get("section_id", ""),
                "court": meta.get("court", ""),
                "type": meta.get("type", ""),
                "content": meta.get("content", "")
            })
    return candidates

def get_graph_distance(case_id_1, case_id_2, driver, section_type=None):
    with driver.session() as session:
        query = (
            """
            MATCH (c1:Case {id: $case1})-[:HAS_SECTION]->(s:Section)<-[:HAS_SECTION]-(c2:Case {id: $case2})
            RETURN COUNT(s) AS shared_sections
            """ if not section_type else
            """
            MATCH (c1:Case {id: $case1})-[:HAS_SECTION]->(s:Section {title: $section_type})<-[:HAS_SECTION]-(c2:Case {id: $case2})
            RETURN COUNT(s) AS shared_sections
            """
        )
        result = session.run(query, case1=case_id_1, case2=case_id_2, section_type=section_type)
        record = result.single()
        shared_sections = record["shared_sections"] if record else 0
        return 1 / (shared_sections + 1) if shared_sections else float("inf")

def rerank_cases(candidates, query_text, section_type=None, alpha=0.7, beta=0.3):
    reranked_cases = []
    for case in candidates:
        cosine_score = case['score']
        try:
            graph_score = get_graph_distance(case['case_id'], query_text, driver, section_type)
            final_score = alpha * cosine_score + beta * (1 / (1 + graph_score)) if graph_score != float("inf") else cosine_score
        except:
            final_score = cosine_score
        reranked_cases.append({**case, "score": final_score})
    return sorted(reranked_cases, key=lambda x: x['score'], reverse=True)

def main(query_text, section_type=None):
    print(f"\n[INFO] Original query: '{query_text}'")
    rewritten_query = rewrite_query_with_gemini(query_text)
    print(f"[INFO] Rewritten query: '{rewritten_query}'")

    candidates = retrieve_semantic_cases(rewritten_query, section_type=section_type)
    if not candidates:
        print("[WARN] No candidates found.")
        return []

    for candidate in candidates:
        candidate["query_case_id"] = rewritten_query

    reranked = rerank_cases(candidates, rewritten_query, section_type=section_type)

    print(f"[INFO] Generating summaries using Gemini...\n")
    for r in reranked:
        r["summary"] = summarize_with_gemini(r["content"])

    return reranked

if __name__ == "__main__":
    query = "Doctrine of economic duress in Indian contract law"
    section_type = "any"
    results = main(query_text=query, section_type=section_type)

    for r in results:
        print(f"[{r['score']:.4f}] Case ID: {r['case_id']} | {r['label']} | {r['title']}")
        print(f"→ Summary: {r['summary']}\n")