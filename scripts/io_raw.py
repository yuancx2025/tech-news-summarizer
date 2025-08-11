from pathlib import Path
import hashlib, json, time, urllib.parse

def article_id(url: str) -> str:
    return hashlib.sha1(url.strip().encode("utf-8")).hexdigest()

def domain_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()

def to_record(extracted: dict) -> dict:
    text = (extracted.get("text") or "").strip()
    title = (extracted.get("title") or "").strip()
    url = extracted["url"].strip()
    rec = {
        "id": article_id(url),
        "url": url,
        "source": domain_of(url),
        "title": title,
        "text": text,
        "authors": extracted.get("authors") or [],
        "top_image": extracted.get("top_image"),
        "published_at": extracted.get("published_at"),   # keep as string for now
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language": extracted.get("language"),
        "word_count": len(text.split()),
        "char_count": len(text),
    }
    return rec

def append_jsonl(rec: dict, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")