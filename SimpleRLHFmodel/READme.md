RLHF Reward Model Project

This project implements a Reward Model training pipeline for Reinforcement Learning with Human Feedback (RLHF).
It allows you to train a BERT-based reward model using human preference data and evaluate answer quality for Q/A pairs.

📂 Project Structure
RLProject/
│
├── prefs.jsonl              # Human preference dataset (chosen vs rejected answers)
├── rlhf_pipeline.py         # Training pipeline (train reward model)
├── test.py                  # Script to test trained reward model
├── checkpoints/             # Saved models after training
│   └── reward/              # Reward model checkpoint
└── README.md                # Project documentation

⚙️ Installation

Clone the repo and install dependencies:

git clone <your-repo-url>
cd RLProject
pip install torch transformers datasets accelerate


(Optional) For GPU training, ensure you have CUDA + PyTorch installed.

📊 Dataset Format

The prefs.jsonl file stores human preferences in JSONL format:

{"prompt": "What is the capital of France?", "chosen": "Paris is the capital of France.", "rejected": "France is in Europe."}
{"prompt": "Who wrote Hamlet?", "chosen": "Hamlet was written by William Shakespeare.", "rejected": "It is a movie directed by Christopher Nolan."}


Each entry contains:

prompt → The question/task.

chosen → The preferred (good) answer.

rejected → The less preferred (bad) answer.

🏋️ Training the Reward Model

Run the following command to train:

python rlhf_pipeline.py --mode train_reward --feedback_file prefs.jsonl --model_name bert-base-uncased --out_dir checkpoints/reward


--mode train_reward → Trains reward model.

--feedback_file → Path to preference dataset.

--model_name → Backbone model (default: bert-base-uncased).

--out_dir → Where to save trained model.

✅ Testing the Model

After training, test the reward model with test.py:

python test.py



Correct, context-related answers score higher, while irrelevant ones score lower.

📌 Next Steps

Add more human feedback data in prefs.jsonl to improve accuracy.

Fine-tune on larger language models (e.g., RoBERTa, DeBERTa).

Extend pipeline for policy optimization with PPO (next RLHF step).
