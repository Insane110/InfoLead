import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core.schema import Document

load_dotenv()

assert os.getenv("LLAMA_CLOUD_API_KEY"), "LLAMA_CLOUD_API_KEY not set"

parser = LlamaParse(
    result_type="markdown",
    verbose=True,
    language="en"
)

def parse_document(file_path: str):
    documents = parser.load_data(file_path)

    parsed_docs = []
    for i, doc in enumerate(documents):
        parsed_docs.append(
            Document(
                text=doc.text,
                metadata={
                    "source": "document",
                    "file_name": os.path.basename(file_path),
                    "page": i
                }
            )
        )

    return parsed_docs