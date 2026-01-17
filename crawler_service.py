from fastapi import FastAPI
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler
from llama_index.core import Document

app = FastAPI()

class CrawlRequest(BaseModel):
    urls: list[str]

@app.post("/crawl")
async def crawl(req: CrawlRequest):
    docs = []
    async with AsyncWebCrawler(verbose=False) as crawler:
        results = await crawler.arun_many(req.urls)

    for r in results:
        if r.success:
            docs.append({
                "text": r.markdown,
                "url": r.url,
                "title": r.metadata.get("title", "N/A")
            })
    return {"docs": docs}
