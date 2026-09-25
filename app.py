import os
from google import genai

client = genai.Client()

chat = client.chats.create(model = "gemini-3.8-flash")

response = chat.send_message("Explain an AI Service")

print("\n----AI response----\n")
print(response.text)