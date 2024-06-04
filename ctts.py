import ChatTTS
from IPython.display import Audio
import os
import time
import requests
import openai
import torchaudio
import torch
import numpy as np

ollama_api_key = 'llama3'
ollama_client = openai.OpenAI(api_key=ollama_api_key, base_url='http://localhost:11434/v1')

def ollama_chat(user_query):
    response = ollama_client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": """You are a good chatbot"""},
            {"role": "user", "content": user_query}
        ],
        temperature=0.5,
        max_tokens=2000,
    )
    return response.choices[0].message.content

chat = ChatTTS.Chat()
chat.load_models(compile=False) # Set to True for better performance

say = ollama_chat("Just ask me a question ffs")

print(say)

texts = [say,]

wavs = chat.infer(texts, )

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)