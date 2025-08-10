import nltk

if __name__ == "__main__":
    print("[BOOTSTRAP] Downloading NLTK punkt tokenizer…")
    nltk.download("punkt")
    print("[BOOTSTRAP] Done.")
    # print(nltk.__version__)