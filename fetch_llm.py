from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
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
    # Use 4-bit quantization for efficiency (runs on CPU/GPU)
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype="float16",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4"
    # )
    # MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    # llm = HuggingFaceLLM(
    #     model_name=MODEL_NAME,
    #     tokenizer_name=MODEL_NAME,
    #     # device_map="cpu",
    #     # model_kwargs={"quantization_config": quantization_config, "trust_remote_code": True},/
    #     model_kwargs={
    #         "torch_dtype": "auto",            # Usually float32 or bfloat16
    #         "trust_remote_code": True,
    #         "low_cpu_mem_usage": True,
    #     },
    #     generate_kwargs={"temperature": 0.2, "max_new_tokens": 512, "do_sample": True},
    #     device_map="auto" 
    # )
    llm = Ollama(
        model="qwen2.5:1.5b",       
        request_timeout=120.0,
        temperature=0.2,
        max_tokens=512,
        context_window=4096,          
    )

    return llm

