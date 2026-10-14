from google import genai
client = genai.Client()
# print("models that we can use: ")
# for model in client.models.list():
#     if (model.name and "flash" in model.name) and "generateContent" in (model.supported_actions or []):
#         print(" ", model.name)

MODEL = "gemini-3.6-flash"

response = client.models.generate_content(
    model=MODEL,
    contents="In one sentence, what is an SEC 10-Q filing?",
)
print(response.text)
print(response.usage_metadata)