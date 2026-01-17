import requests
from llama_index.core import Document

def crawl_via_fastapi(urls):
    resp = requests.post(
        "http://localhost:8000/crawl",
        json={"urls": urls},
        timeout=120
    )
    resp.raise_for_status()

    docs = []
    for d in resp.json()["docs"]:
        docs.append(
            Document(
                text=d["text"],
                metadata={
                    "url": d["url"],
                    "title": d.get("title", "N/A")
                }
            )
        )
    return docs