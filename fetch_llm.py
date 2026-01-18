from llama_index.llms.ollama import Ollama

REFINEMENT_PROMPT = """
You are an expert summarizer and refiner. Take the following raw response and refine it to be:
- Precise: Remove any redundant or irrelevant information.
- Concise: Shorten while keeping key details.
- Meaningful: Structure it logically (e.g., use bullet points if helpful, start with a clear answer).
- Accurate: Do not add or change facts.

Raw response: {raw_response}

Refined output:
"""

def output_llm():
    llm = Ollama(
        model="qwen2.5:1.5b",       
        request_timeout=120.0,
        temperature=0.2,
        max_tokens=512,
        context_window=4096,          
    )

    return llm

