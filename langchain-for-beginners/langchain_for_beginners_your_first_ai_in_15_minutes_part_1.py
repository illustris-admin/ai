### 📘 Your First AI in 15 Minutes (LangChain for Beginners, Part 1)

**Goal**: Install LangChain, load a local model, and run your first prompt — all for free in under 15 minutes.

✅ No API keys

✅ Uses Hugging Face + LangChain

✅ Runs on free Google Colab GPU

🚀 Let's go!
"""

# Install required libraries
!pip install -q langchain-huggingface transformers torch accelerate bitsandbytes sentence-transformers

"""### 🤖 Why Use a Local Model?

- 🔐 **Private**: Your data stays in Colab
- 💸 **Free**: No API costs
- 🛠️ **Control**: You own the model
- 🧠 **Educational**: Learn how LLMs really work
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# Step 1: Choose a lightweight, fast model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    do_sample=True,
)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

print("✅ Model loaded and ready!")

"""### 🧪 Run Your First Prompt"""

# Improved: Use chat template and clean output
def ask(question):
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    response = llm.invoke(prompt)
    # Clean up response
    answer = response.strip().replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "").strip()
    print(f"Q: {question}\n")
    print(f"A: {answer}")

# Try it!
ask("Explain quantum computing in simple terms.")

"""### 🔄 Try More Prompts!"""

# Define the clean ask() function
def ask(question):
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    response = llm.invoke(prompt)
    answer = response.replace(prompt, "").replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "").strip()
    print(f"Q: {question}")
    print(f"A: {answer}\n")

# Try fun prompts
ask("Write a haiku about rain.")

ask("Write a Python function to reverse a string.")

ask("Tell me a joke about AI.")

"""### 🧩 What Is LangChain?

> **LangChain** is a framework for building applications powered by large language models.

🔧 It helps you:
- Connect models to data
- Add memory and context
- Chain prompts, tools, and actions
- Build chatbots, agents, and workflows

Think of it as **Lego blocks for AI apps**.

### 🧹 Optional: Free GPU Memory
"""

import torch
torch.cuda.empty_cache()
print("GPU memory cleared.")

"""### 🎉 Summary

In this notebook, you:

✅ Installed LangChain and Hugging Face libraries

✅ Loaded **TinyLlama** — a free, local, private model

✅ Wrapped it in LangChain

✅ Ran real prompts — no API key needed

💡 You now have a working AI assistant in Colab — in under 15 minutes!

➡️ **Next: Add memory so your AI remembers the conversation!**

### 🔗 Resources

- [TinyLlama on Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [LangChain Docs](https://python.langchain.com)
- [Hugging Face Models](https://huggingface.co/models)
"""
