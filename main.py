from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct", device_map="auto")

messages = [
    {
        "role": "system", 
        "content": "You are a professional UFC analyst and combat sports expert. Provide a sharp, well-structured, and insightful breakdown."
    },
    {
        "role": "user", 
        "content": "Analyze Ilia Topuria's fighting style. Explain why he is such a dangerous threat in the Octagon."
    }
]

output = pipe(messages, max_new_tokens=1000, do_sample=True, temperature=0.7)
print(output[0]['generated_text'][-1]['content'])
