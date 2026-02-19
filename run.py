import asyncio
from operator import index
from crawl4ai import AsyncWebCrawler
from fetch_urls import fetch_top_n_links
from docs_parse import parse_document
from chunk import get_token_nodes
from store_db import get_or_create_index
from fetch_llm import output_llm, REFINEMENT_PROMPT
from llama_index.core import Document
from rank_bm25 import BM25Okapi
# from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import get_response_synthesizer
from collections import defaultdict

def RRF(vector_nodes, bm25_nodes, k=60):
    scores = defaultdict(float)
    for rank, node_with_score in enumerate(vector_nodes):
        node_id = node_with_score.node.node_id
        scores[node_id] += 1 / (k + rank + 1)
    for rank, node_with_score in enumerate(bm25_nodes):
        node_id = node_with_score.node.node_id
        scores[node_id] += 1 / (k + rank + 1)
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused_nodes = []
    for node_id, score in sorted_nodes:
        node = next(n.node for n in vector_nodes + bm25_nodes if n.node.node_id == node_id)
        fused_nodes.append(NodeWithScore(node=node, score=score))
    return fused_nodes

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        return RRF(vector_nodes, bm25_nodes)
    
class SimpleBM25Retriever(BaseRetriever):
    def __init__(self, nodes, similarity_top_k=5):
        super().__init__()
        self.nodes = nodes
        self.similarity_top_k = similarity_top_k
        corpus = [node.text.split() for node in nodes]
        self.bm25 = BM25Okapi(corpus)

    def _retrieve(self, query_bundle: QueryBundle):
        query_tokens = query_bundle.query_str.split()
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.nodes, scores), key=lambda x: x[1], reverse=True)[:self.similarity_top_k]
        return [NodeWithScore(node=node, score=float(score)) for node, score in ranked]
    
async def main():
    llm = output_llm()
    index, embed_model = get_or_create_index()

    user_query = input("Enter your query: ")

    search_internet = input("Do you want to search the internet? (y/n): ").lower() == 'y'
    web_docs = []
    if search_internet:
        search_query = input("Enter search query (or press enter to use the query above): ") or user_query
        num_results = int(input("Number of results: ") or 5)
        top_links = fetch_top_n_links(search_query, num_results)
        async with AsyncWebCrawler(verbose=True) as crawler:
            results = await crawler.arun_many(top_links)
        for result in results:
            if result.success:
                web_docs.append(Document(text=result.markdown, metadata={"url": result.url, "title": result.metadata.get('title', 'N/A')}))
    
    attach_doc = input("Do you want to attach a document? (y/n): ").lower() == 'y'
    doc_docs = []
    if attach_doc:
        file_path = input("Enter the file path: ")
        doc_docs = parse_document(file_path)
    
    all_new_docs = web_docs + doc_docs
    nodes = None
    if all_new_docs:
        new_nodes = get_token_nodes(all_new_docs)
        nodes = new_nodes
        index.insert_nodes(new_nodes)
        index.storage_context.persist()
    
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    # bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    bm25_retriever = SimpleBM25Retriever(nodes=nodes, similarity_top_k=5)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    rerank = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3)
    response_synthesizer = get_response_synthesizer(llm=llm)
    # query_engine = index.as_query_engine(llm=llm, embed_model=embed_model)
    query_engine = RetrieverQueryEngine(retriever=hybrid_retriever, 
                                        node_postprocessors=[rerank], 
                                        response_synthesizer=response_synthesizer)
    response = query_engine.query(user_query)
    raw_text = response.response

    prompt = REFINEMENT_PROMPT.format(raw_response=raw_text)
    refined_response = llm.complete(prompt).text

    print("\nFinal Refined Response:\n", refined_response)
    
if __name__ == "__main__":
    asyncio.run(main())
