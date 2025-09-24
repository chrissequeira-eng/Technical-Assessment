THRESHOLD = 0.7

def search_with_threshold(vectorstore, sources_dict, query, top_k=5, threshold=THRESHOLD):
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    filtered_results = [(doc, score) for doc, score in results if score >= threshold]

    contexts = []
    answer_texts = []
    for doc, score in filtered_results:
        source_info = sources_dict.get(doc.metadata['source'], {"title": doc.metadata['source'], "url": ""})
        contexts.append({
            "chunk_id": doc.metadata['id'],
            "text": doc.page_content,
            "score": score,
            "title": source_info['title'],
            "url": source_info['url']
        })
        answer_texts.append(doc.page_content)

    answer = "\n".join(answer_texts[:3]) if answer_texts else None
    return {
        "answer": answer,
        "contexts": contexts,
        "reranker_used": None
    }
