import asyncio
from crawl4ai import AsyncWebCrawler
from fetch_urls import fetch_top_n_links
from docs_parse import parse_document
from chunk import get_token_nodes
from store_db import get_or_create_index
from fetch_llm import output_llm, REFINEMENT_PROMPT
from llama_index.core import Document

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
    if all_new_docs:
        new_nodes = get_token_nodes(all_new_docs)
        index.insert_nodes(new_nodes)
        index.storage_context.persist()

    query_engine = index.as_query_engine(llm=llm, embed_model=embed_model)
    response = query_engine.query(user_query)
    raw_text = response.response

    prompt = REFINEMENT_PROMPT.format(raw_response=raw_text)
    refined_response = llm.complete(prompt).text

    print("\nFinal Refined Response:\n", refined_response)
    
if __name__ == "__main__":
    asyncio.run(main())