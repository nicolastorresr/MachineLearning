from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

def GPT(prompt, model="gpt-4o-mini", temperature=0.5, max_length=100, num_return_sequences=1):
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_length,
        n=num_return_sequences,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    for k in range(num_return_sequences):
        print(completion.choices[k].message.content,"\n")