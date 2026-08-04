import os
from dotenv import load_dotenv
from openai import OpenAI

# Carrega as variáveis do arquivo .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

modelo = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

response = client.chat.completions.create(
    model=modelo,
    messages=[
        {"role": "user", "content": "Qual a capital do Brasil?"}
    ],
)

print(response.choices[0].message.content)