from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Generator: GPT-2
gen_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gen_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Reward model
reward_tokenizer = BertTokenizer.from_pretrained("checkpoints/reward")
reward_model = BertForSequenceClassification.from_pretrained("checkpoints/reward")

context = """The Great Wall of China is a series of fortifications 
built along the historical northern borders of China. 
Its purpose was to protect against nomadic groups and invasions."""
question = "Why was the Great Wall of China built?"

# Generator input = context + question
gen_input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
inputs = gen_tokenizer(gen_input_text, return_tensors="pt")

# Generate multiple candidate answers
outputs = gen_model.generate(**inputs, max_length=80, num_return_sequences=3, do_sample=True)
candidates = [gen_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

# Score each candidate
for ans in candidates:
    pair = f"Context: {context}\nQuestion: {question}\nAnswer: {ans}"
    tokens = reward_tokenizer(pair, return_tensors="pt", padding=True, truncation=True)
    score = reward_model(**tokens).logits.softmax(dim=-1)[0][1].item()
    print(f"Answer: {ans}\n -> Reward Score: {score}\n")
