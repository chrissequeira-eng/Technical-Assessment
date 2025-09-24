import math
from collections import Counter

def bm25_score(query_terms, doc_text, k1=1.5, b=0.75, avg_doc_len=100):
    """
    Simple BM25-like scoring for keyword matches.
    """
    doc_terms = doc_text.lower().split()
    term_freqs = Counter(doc_terms)
    doc_len = len(doc_terms)

    score = 0.0
    for term in query_terms:
        f = term_freqs.get(term, 0)
        if f == 0:
            continue
        idf = 1.0 
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += idf * (numerator / denominator)
    return score

def rerank(results, query, method="hybrid"):
    """
    Hybrid reranker: combines vector similarity + BM25 keyword score.
    `results` is a list of dicts: {"text": ..., "score": ...}
    """
    if method != "hybrid":
        return results

    query_terms = query.lower().split()
    avg_doc_len = sum(len(r["text"].split()) for r in results) / max(len(results), 1)

    reranked = []
    for r in results:
        bm25 = bm25_score(query_terms, r["text"], avg_doc_len=avg_doc_len)
        combined_score = 0.7 * r["score"] + 0.3 * (bm25 / (bm25 + 1)) 
        r_copy = r.copy()
        r_copy["reranked_score"] = combined_score
        reranked.append(r_copy)

    reranked.sort(key=lambda x: x["reranked_score"], reverse=True)
    return reranked
