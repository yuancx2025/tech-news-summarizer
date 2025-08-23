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

class CustomSentenceTokenizer:
    """Custom sentence tokenizer that doesn't rely on NLTK punkt."""
    
    def __init__(self, language="english"):
        self.language = language
        # Common sentence ending patterns
        self.sentence_end_patterns = [
            r'(?<=[.!?])\s+',  # Standard sentence endings
            r'(?<=[.!?])\n+',  # Sentence endings with newlines
            r'(?<=[.!?])\t+',  # Sentence endings with tabs
        ]
    
    def tokenize(self, text):
        """Split text into sentences."""
        if not text:
            return []
        
        # Clean the text
        text = text.strip()
        
        # Try different patterns to split sentences
        sentences = []
        for pattern in self.sentence_end_patterns:
            sentences = re.split(pattern, text)
            if len(sentences) > 1:
                break
        
        # If no pattern worked, split on periods
        if len(sentences) <= 1:
            sentences = re.split(r'\.\s+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

class TextRankSummarizer:
    def __init__(self, num_sentences: int = 3):
        self.num_sentences = num_sentences
        self._check_dependencies()
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.summarizers.text_rank import TextRankSummarizer as _TR
        self._PlaintextParser = PlaintextParser
        self._TR = _TR

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import sumy
        except ImportError:
            raise RuntimeError(
                "sumy package not found. Please install it with:\n"
                "pip install sumy"
            )

    def summarize(self, text: str, sentences: Optional[List[str]] = None) -> str:
        try:
            # Use custom sentence tokenizer instead of NLTK
            if sentences and isinstance(sentences, list):
                # Use pre-split sentences if available
                text_for_parser = " ".join(sentences)
            else:
                text_for_parser = text
            
            # Create a custom tokenizer class that mimics sumy's Tokenizer
            class CustomTokenizer:
                def __init__(self, language="english"):
                    self.language = language
                    self._tokenizer = CustomSentenceTokenizer(language)
                
                def tokenize(self, text):
                    return self._tokenizer.tokenize(text)
            
            # Create parser with custom tokenizer
            parser = self._PlaintextParser.from_string(text_for_parser or "", CustomTokenizer("english"))
            summarizer = self._TR()
            sents = summarizer(parser.document, self.num_sentences)
            return " ".join(str(s) for s in sents).strip()
            
        except Exception as e:
            # Fallback to simple sentence extraction
            print(f"Warning: TextRank failed, using fallback: {e}")
            if sentences and isinstance(sentences, list):
                return " ".join(sentences[:self.num_sentences]).strip()
            else:
                # Simple sentence splitting as fallback
                sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
                return " ".join(sents[:self.num_sentences]).strip()

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
