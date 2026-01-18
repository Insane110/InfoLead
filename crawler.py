from crawl4ai import AsyncWebCrawler
from llama_index.core import Document
import asyncio

async def crawl_urls_async(urls):
    docs = []
    async with AsyncWebCrawler(verbose=False) as crawler:
        results = await crawler.arun_many(urls)

    for r in results:
        if r.success:
            docs.append(
                Document(
                    text=r.markdown,
                    metadata={
                        "url": r.url,
                        "title": r.metadata.get("title", "N/A")
                    }
                )
            )
    return docs

def crawl_urls(urls):
    return asyncio.run(crawl_urls_async(urls))
