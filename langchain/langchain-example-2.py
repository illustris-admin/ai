import os
from transformers import pipeline
from time import sleep

def clear_screen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')

# Initialize the text generation pipeline with a free model
generator = pipeline('text-generation', model='gpt2')

def generate_company_name(product):
    prompt = f"Suggest a company name for a business that makes {product}:"
    result = generator(prompt, max_length=50, num_return_sequences=1)
    return result[0]['generated_text']

# Clear the screen before running
clear_screen()

# Example usage
product = "eco-friendly water bottles"
company_name = generate_company_name(product)
print(f"\nProduct: \n{product}")
print(f"\nCompany Name: \n{company_name}")
