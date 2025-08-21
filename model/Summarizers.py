from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Lead3Summarizer:
    num_sentences: int = 3
    def summarize(self, text: str, sentences: Optional[List[str]] = None) -> str:
        if sentences and isinstance(sentences, list):
            return " ".join(sentences[: self.num_sentences]).strip()
        # fallback simple splitter
        sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
        return " ".join(sents[: self.num_sentences]).strip()

class TextRankSummarizer:
    def __init__(self, num_sentences: int = 3):
        self.num_sentences = num_sentences
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer as _TR
        self._PlaintextParser = PlaintextParser
        self._Tokenizer = Tokenizer
        self._TR = _TR

    def summarize(self, text: str, sentences: Optional[List[str]] = None) -> str:
        parser = self._PlaintextParser.from_string(text or "", self._Tokenizer("english"))
        summarizer = self._TR()
        sents = summarizer(parser.document, self.num_sentences)
        return " ".join(str(s) for s in sents).strip()

class HFSummarizer:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[int] = None,  # None/-1 CPU, 0 GPU index
        min_length: int = 60,
        max_length: int = 200,
        num_beams: int = 4,
    ):
        from transformers import pipeline
        self.pipe = pipeline("summarization", model=model_name, device=device if device is not None else -1)
        self.kw = dict(min_length=min_length, max_length=max_length, num_beams=num_beams)

    def summarize(self, text: str, sentences: Optional[List[str]] = None) -> str:
        if not text or not text.strip():
            return ""
        out = self.pipe(text, **self.kw)
        return out[0]["summary_text"].strip()

def build_summarizers():
    return {
        "lead3": Lead3Summarizer(num_sentences=3),
        "textrank": TextRankSummarizer(num_sentences=3),
        "distilbart": HFSummarizer(model_name="sshleifer/distilbart-cnn-12-6"),
        "bart": HFSummarizer(model_name="facebook/bart-large-cnn"),
    }
