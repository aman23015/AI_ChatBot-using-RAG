from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content":"Explain me the GenAI"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=100,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

