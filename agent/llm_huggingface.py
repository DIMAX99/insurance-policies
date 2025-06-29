# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
# from langchain_huggingface import HuggingFacePipeline

# # Use 4-bit quantization to fit on RTX 4050 (8GB)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16 if you get issues
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     llm_int8_enable_fp32_cpu_offload=True  # ✅ Correct place
# )

# model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     device_map="auto",
#     local_files_only=True
# )

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=384,
#     do_sample=True,
#     temperature=0.1,
#     top_k=50,
#     top_p=0.95
# )

# llm = HuggingFacePipeline(pipeline=pipe)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

# Use 4-bit quantization (safe for Gemma and even smoother on RTX 4050)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16 if you get issues
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

# ✅ Change to Gemma
model_id = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True  # keep if you already have model locally
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=384,
    do_sample=True,
    temperature=0.1,
    top_k=50,
    top_p=0.95
)

llm = HuggingFacePipeline(pipeline=pipe)

