import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

clear_screen()

# Initialize LLM
llm = OllamaLLM(model="mistral")

# First Chain: Text Processing Chain
process_prompt = PromptTemplate.from_template("Process this input: {input}")
enhance_prompt = PromptTemplate.from_template("Enhance this result by adding more details: {input}")

process_chain = (
    process_prompt | 
    llm | 
    StrOutputParser()
)

enhance_chain = (
    enhance_prompt | 
    llm | 
    StrOutputParser()
)

text_processing_chain = (
    process_chain | 
    (lambda x: {"input": x}) |
    enhance_chain
)

# Second Chain: Analysis Chain
analyze_prompt = PromptTemplate.from_template("Analyze the key points in: {input}")
summarize_prompt = PromptTemplate.from_template("Create a concise summary of this analysis: {input}")

analyze_chain = (
    analyze_prompt | 
    llm | 
    StrOutputParser()
)

summarize_chain = (
    summarize_prompt | 
    llm | 
    StrOutputParser()
)

analysis_chain = (
    analyze_chain | 
    (lambda x: {"input": x}) |
    summarize_chain
)

# Test both chains with different inputs

# Testing 'Text Processing Chain'
print("\n=== Text Processing Chain ===")
text_input = "The quick brown fox jumps over the lazy dog"
text_result = text_processing_chain.invoke({"input": text_input})
print(f"Input: {text_input}")
print(f"Output: {text_result}")

# Testing 'Analysis Processing Chain'
print("\n=== Analysis Chain ===")
analysis_input = "Climate change is causing rising sea levels and extreme weather patterns globally"
analysis_result = analysis_chain.invoke({"input": analysis_input})
print(f"Input: {analysis_input}")
print(f"Output: {analysis_result}")
