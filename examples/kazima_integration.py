"""
kazima_integration.py
---------------------
Example: how to use mukhtasar inside kazima-ai RAG pipeline.

kazima-ai is a TypeScript/Next.js project, so the actual bridge runs as a
Python preprocessing script that is called from Node via child_process, or
run independently before indexing.

Install:
    pip install mukhtasar

Usage (standalone):
    python examples/kazima_integration.py
"""

from mukhtasar import normalize, tokenize, summarize


# ---------------------------------------------------------------------------
# 1. Normalize a user query before passing it to the embedding / keyword search
# ---------------------------------------------------------------------------

def clean_query(user_query: str) -> str:
    """
    Normalize Arabic query: strip diacritics, unify alef/yeh/teh variants.
    Drop this into kazima-retrieval before splitKeywords().
    """
    return normalize(user_query)


# ---------------------------------------------------------------------------
# 2. Summarize a long historical text before chunking into the vector store
# ---------------------------------------------------------------------------

def prepare_rag_chunk(long_historical_text: str, ratio: float = 0.3) -> str:
    """
    Summarize text and return the compressed version ready for vector store.
    Keeps the most important sentences (TextRank + TF-IDF, extractive).

    Parameters
    ----------
    long_historical_text : str
        Raw Arabic text from kazima.org topic (contentLong field).
    ratio : float
        Compression ratio — 0.3 means keep 30 % of original sentences.

    Returns
    -------
    str
        Summarized text to pass as the RAG chunk.
    """
    summary = summarize(long_historical_text, ratio=ratio)
    return summary.text


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- query normalization ---
    raw_query = "الإمارات العربيَّة المتَّحدة وتاريخها"
    clean = clean_query(raw_query)
    print("Original query :", raw_query)
    print("Normalized     :", clean)
    print()

    # --- tokenization (for keyword matching / BM25) ---
    tokens = tokenize(clean)
    print("Tokens         :", tokens)
    print()

    # --- RAG chunk preparation ---
    sample_text = (
        "الكويت دولة عربية تقع في الركن الشمالي الغربي من الخليج العربي. "
        "تأسست إمارة الكويت في مطلع القرن الثامن عشر الميلادي. "
        "تتميز الكويت بموقعها الاستراتيجي الذي جعلها مركزاً تجارياً مهماً. "
        "اشتهر أهلها بالتجارة والغوص على اللؤلؤ وبناء السفن. "
        "اكتُشف النفط في الكويت عام 1938م مما حوّل مسار تطورها الاقتصادي بشكل جذري. "
        "أصبحت الكويت من أغنى دول العالم وحققت تقدماً ملحوظاً في شتى المجالات. "
        "تضم الكويت إرثاً تاريخياً وحضارياً عريقاً يعكس تنوع روافدها الثقافية. "
        "تحتضن مكتبة الكويت الوطنية آلاف المخطوطات والوثائق النادرة التي توثق تاريخ المنطقة."
    )

    chunk = prepare_rag_chunk(sample_text, ratio=0.4)
    print("Original length:", len(sample_text), "chars")
    print("Chunk length   :", len(chunk), "chars")
    print("Chunk text     :", chunk)
