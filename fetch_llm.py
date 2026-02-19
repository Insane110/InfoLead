from llama_index.llms.ollama import Ollama

REFINEMENT_PROMPT = """
You are an expert assistant.

Use the provided context to answer the question.

Make the answer:
- Precise: Remove redundant information.
- Concise: Keep only key details.
- Meaningful: Structure logically (use bullet points if helpful).
- Accurate: Do not add or change facts.
- Real answer: If answer is not present in context, say:
"I don't know and this was not in my knowledge base."

Context:
{context_str}

Question:
{query_str}

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

