# eval/gen_cnndm_preds_gemini.py
import os, re, json, argparse, pathlib, random
import numpy as np
from tqdm import tqdm
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

random.seed(0); np.random.seed(0)

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def chunk_text(text, max_chars=900, overlap=120):
    parts = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i+max_chars, n)
        # try to break on sentence boundary near j
        k = text.rfind(". ", i, j)
        if k == -1 or j - k > 200:
            k = j
        else:
            k += 2
        parts.append(text[i:k].strip())
        i = max(k - overlap, i + 1)
    return [p for p in parts if len(p) > 40]

def mmr_indices(X, query_vec, k=8, lambda_mult=0.6):
    # X: (m, d) chunk vectors, query_vec: (1, d)
    m = X.shape[0]
    if m == 0: return []
    sim = cosine_similarity(X, query_vec.reshape(1, -1)).reshape(-1)
    selected = []
    cand = list(range(m))
    while len(selected) < min(k, m):
        if not selected:
            i = int(np.argmax(sim))
            selected.append(i); cand.remove(i)
            continue
        # diversity: penalize similarity to selected centroid
        sel_vec = X[selected].mean(axis=0, keepdims=True)
        div = cosine_similarity(X, sel_vec).reshape(-1)
        # MMR score
        mmr = lambda_mult * sim + (1 - lambda_mult) * (1 - div)
        # pick best among candidates
        best = max(cand, key=lambda idx: mmr[idx])
        selected.append(best); cand.remove(best)
    return selected

def select_evidence_chunks(article_text, k=8):
    # sentence/paragraph-like chunks
    chunks = chunk_text(article_text, max_chars=900, overlap=120)
    if not chunks:
        return [], [], 0
    # TF-IDF against the whole doc: relevance to doc centroid
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(chunks)           # (m, d)
    doc_vec = X.mean(axis=0)                # centroid
    mmr_idx = mmr_indices(X.toarray(), np.asarray(doc_vec).ravel(), k=k, lambda_mult=0.65)
    sel = [chunks[i] for i in mmr_idx]
    ids = [f"C{i+1}" for i in range(len(sel))]
    return ids, sel, len(article_text.split())

PROMPT_NONRAG = lambda target_tokens: f"""
You are a precise, neutral news summarizer.

TASK:
- Write a short paragraph (≈{target_tokens} tokens) and 4–7 bullet points.
- Be factual and concise. Avoid hype and opinion.
- Do NOT invent facts.

OUTPUT JSON (and only JSON):
{{
  "summary": "<~{target_tokens} tokens>",
  "bullets": ["...", "...", "..."]
}}
"""

PROMPT_RAG = lambda target_tokens: f"""
You are a precise, neutral news summarizer.

You will receive EVIDENCE CHUNKS from an article. Use ONLY the evidence to write the summary.
Cite which chunks support each bullet via "citations" indices.

TASK:
- Write a short paragraph (≈{target_tokens} tokens) and 4–7 bullet points.
- Every bullet MUST be supported by at least one evidence chunk.

OUTPUT JSON (and only JSON):
{{
  "summary": "<~{target_tokens} tokens>",
  "bullets": ["...", "...", "..."],
  "bullets_citations": [[1],[1,3], ...]  # 1-based indices of supporting chunks
}}

EVIDENCE CHUNKS FORMAT:
[C1] text...
[C2] text...
"""

def to_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    genai.configure(api_key=api_key)
    # Use a fast, capable default; change if you prefer Pro
    return genai.GenerativeModel("gemini-1.5-flash")

def call_gemini(model, prompt, evidence=None, max_output_tokens=1024, temperature=0.2):
    if evidence:
        content = prompt + "\n\n" + evidence
    else:
        content = prompt
    resp = model.generate_content(
        content,
        generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens}
    )
    return resp.text

def extract_json(text):
    # Be robust to stray prose; try to pull the outermost JSON object.
    m = re.search(r"\{.*\}", text, flags=re.S)
    s = text if m is None else m.group(0)
    try:
        obj = json.loads(s)
        # normalize
        obj["bullets"] = [b.strip() for b in obj.get("bullets", []) if b.strip()]
        return obj
    except Exception:
        # fall back: wrap whole text as 'summary'
        return {"summary": text.strip(), "bullets": []}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default="data/gold/articles.jsonl")
    ap.add_argument("--mode", choices=["rag","nonrag"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8, help="evidence chunks for RAG")
    ap.add_argument("--target_tokens", type=int, default=120)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    model = to_model()
    outdir = pathlib.Path(args.out).parent; outdir.mkdir(parents=True, exist_ok=True)
    fp = open(args.out, "w", encoding="utf-8")

    for i, ex in enumerate(tqdm(load_jsonl(args.articles), total=args.limit, desc=f"gen/{args.mode}")):
        if i >= args.limit: break
        aid = ex["article_id"]; article = ex["article"]

        if args.mode == "nonrag":
            prompt = PROMPT_NONRAG(args.target_tokens) + "\n\nARTICLE:\n" + article[:15000]
            raw = call_gemini(model, prompt)
            obj = extract_json(raw)
            rec = {
                "article_id": aid,
                "summary": obj.get("summary","").strip(),
                "bullets": obj.get("bullets", []),
                "source_len": len(article.split()),
                "retrieved_chunks": []  # empty for non-RAG
            }

        else:  # RAG
            chunk_ids, chunk_texts, src_len = select_evidence_chunks(article, k=args.k)
            evidence = "\n".join([f"[{cid}] {txt}" for cid, txt in zip(chunk_ids, chunk_texts)])
            prompt = PROMPT_RAG(args.target_tokens)
            raw = call_gemini(model, prompt, evidence=evidence)
            obj = extract_json(raw)
            # map 1-based indices to real chunk ids
            cites = obj.get("bullets_citations", [])
            rec = {
                "article_id": aid,
                "summary": obj.get("summary","").strip(),
                "bullets": obj.get("bullets", []),
                "bullets_citations": cites,
                "source_len": src_len,
                "retrieved_chunks": [
                    {"chunk_id": cid, "article_id": aid, "text": txt}
                    for cid, txt in zip(chunk_ids, chunk_texts)
                ]
            }

        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fp.flush()
    fp.close()

if __name__ == "__main__":
    main()
