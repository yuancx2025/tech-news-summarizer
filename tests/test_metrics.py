from model.metrics import rouge_batch, lengths, compression_ratio, Timer

def test_rouge_and_lengths():
    refs = ["the quick brown fox jumps over the lazy dog"]
    preds = ["quick fox jumps over dog"]
    m = rouge_batch(preds, refs)
    assert "rouge1" in m and "f" in m["rouge1"]
    L = lengths(preds)
    assert L["words_avg"] > 0
    c = compression_ratio(refs, preds)
    assert 0 < c < 1.0

def test_timer():
    t = Timer()
    t.tic(); t.toc()
    s = t.stats()
    assert "ms_avg" in s
