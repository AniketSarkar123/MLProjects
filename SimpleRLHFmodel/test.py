from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model & tokenizer
model = BertForSequenceClassification.from_pretrained("checkpoints/reward")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Long paragraph context
context = """
The Great Wall of China is a historic fortification built across northern China.
It stretches over 13,000 miles and was constructed over several dynasties, mainly
to protect against invasions. Today, it is considered one of the most iconic landmarks
in the world and is even visible from space under certain conditions.
"""

# Example Q/A pairs
examples = [
    {
        "question": "Why was the Great Wall of China built?",
        "answer": "It was built to protect China from invasions and raids."
    },
    {
        "question": "How tall is Mount Everest?",
        "answer": "Mount Everest is about 8,849 meters tall."  # Wrong context
    },
    {
        "question": "Is the Great Wall of China visible from space?",
        "answer": "Yes, under certain conditions it can be seen from space."
    },
    {
        "question": "Where is the Eiffel Tower located?",
        "answer": "The Eiffel Tower is in Paris, France."  # Wrong context
    }
]

# Run reward model
for ex in examples:
    combined_input = context + " " + ex["question"] + " " + ex["answer"]
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    print(f"Q: {ex['question']}")
    print(f"A: {ex['answer']}")
    print(" -> Probability of being a good answer:", probs[0][1].item())
    print("-" * 80)
