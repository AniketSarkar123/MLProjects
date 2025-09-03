"""
RLHF PIPELINE (toy implementation)

What this script provides (toy / educational):
- supervised fine-tuning (SFT) on (paragraph, question -> answer) pairs using HuggingFace Transformers
- an interactive loop to ask questions about loaded paragraphs; the model answers
- a CLI-based human feedback collector (pairwise preference): shows two model answers and asks the human to pick the preferred
- trains a simple reward model (transformer classification/regression) on human-labelled preferences
- performs a PPO-style policy update using `trl` (tiny example) where the reward comes from the trained reward model

NOTES / WARNINGS
- This is a simplified, educational pipeline. Real RLHF used by large models contains many additional components (safety filtering, large compute, distributed training, sophisticated reward-model losses, baselines, KL penalties, etc.).
- Use very small models (e.g., `gpt2`/`distilgpt2`) for experimentation on a single GPU/CPU.
- Installing dependencies: `pip install transformers datasets accelerate trl torch sentencepiece` (versions change often).
- You should run this on a machine with a GPU for PPO training. The code is written to be readable and modular, not production-ready.

Usage examples (after installing deps):
  python rlhf_pipeline.py --mode sft --train_file paragraphs_qa.jsonl --model_name gpt2 --out_dir checkpoints/sft
  python rlhf_pipeline.py --mode interact --model_dir checkpoints/sft
  python rlhf_pipeline.py --mode collect_feedback --model_dir checkpoints/sft --n_rounds 20 --out pref_journal.jsonl
  python rlhf_pipeline.py --mode train_reward --feedback_file pref_journal.jsonl --model_name "bert-base-uncased" --out_dir checkpoints/reward
  python rlhf_pipeline.py --mode ppo --model_dir checkpoints/sft --reward_model checkpoints/reward

File formats:
- paragraphs_qa.jsonl: each line is JSON with {"paragraph": "...", "question": "...", "answer": "..."}
- pref_journal.jsonl: each line is JSON with {"prompt": "Paragraph:\n...\nQuestion:\n...", "a": "answer text A", "b": "answer text B", "choice": "a" or "b"}

"""

import os
import json
import random
import argparse
from typing import List, Dict, Tuple

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# NOTE: `trl` (by huggingface/trl) provides PPOTrainer abstractions. Try to import but fall back gracefully.
try:
    from trl import PPOTrainer, PPOConfig
    TRL_AVAILABLE = True
except Exception:
    TRL_AVAILABLE = False


# ----------------------------- Utilities -----------------------------

def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------- Prompt helpers ----------------------------

def build_prompt(paragraph: str, question: str) -> str:
    return f"Paragraph:\n{paragraph}\n\nQuestion:\n{question}\n\nAnswer:\n"


# ----------------------- Supervised Fine-tune -----------------------

def fine_tune_sft(train_file: str, model_name: str, out_dir: str, epochs: int = 3, per_device_train_batch_size: int = 2):
    """Simple SFT using causal language modeling (LM) objective. Expects train_file jsonl with paragraph/question/answer."""
    data = read_jsonl(train_file)
    prompts = [build_prompt(d["paragraph"], d["question"]) + d["answer"] for d in data]
    hf_ds = Dataset.from_dict({"text": prompts})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ensure a pad token exists (some causal models don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # load model first so we can resize embeddings after potentially adding pad token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # set model pad_token_id so generation/loss handling is consistent
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # tokenization function: produce input_ids, attention_mask and labels
    def tokenize_fn(batch):
        # we pad to fixed max_length here (keeps labels aligned). You can change to dynamic padding if desired.
        toks = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
        input_ids = toks["input_ids"]
        # convert pad token ids in labels to -100 so they are ignored by loss
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
        labels = []
        for seq in input_ids:
            if pad_id != -1:
                labels.append([tok if tok != pad_id else -100 for tok in seq])
            else:
                labels.append(list(seq))  # fallback (unlikely)
        toks["labels"] = labels
        return toks

    tokenized = hf_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # DataCollatorWithPadding will handle batching; labels already use -100 for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)


# -------------------- Interactive inference & collection ------------

def load_causal_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def sample_answer(tokenizer, model, prompt: str, max_new_tokens: int = 64,
                  do_sample: bool = True, temperature: float = 0.8) -> str:
    """
    Generate a short, constrained answer for `prompt`.

    Post-processes the decoded text to avoid the LM continuing with extra
    questions or unrelated text. We trim after the first appearance of
    likely "question" markers or blank-line separators.
    """
    # Tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            # early_stopping True often helps stop generation at eos
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # extract generated portion after the prompt
    gen = text[len(prompt):].strip()

    # post-process: cut off if the model started generating new questions or Q/A blocks
    # common markers to stop at:
    stop_markers = ["\nQuestion:", "\nQuestion", "\n\nQuestion", "\n\nQ:", "\nQ:", "\nAnswer:", "\n\n", "\n- ", "\n\n-"]
    cut_pos = None
    for m in stop_markers:
        idx = gen.find(m)
        if idx != -1:
            if cut_pos is None or idx < cut_pos:
                cut_pos = idx
    if cut_pos is not None:
        gen = gen[:cut_pos].strip()

    # further trim: stop at first occurrence of double newlines or multiple short Qs
    # keep only the first 3 lines to avoid long tangents
    lines = [l for l in gen.splitlines() if l.strip() != ""]
    if len(lines) > 3:
        gen = " ".join(lines[:3]).strip()
    else:
        gen = " ".join(lines).strip()

    # Remove trailing "Answer" or "Question" fragments if present
    for suffix in ["Answer", "Answer:", "Question", "Question:"]:
        if gen.endswith(suffix):
            gen = gen[:-len(suffix)].strip()

    return gen

# ---------------------- Feedback collection -------------------------

def collect_pairwise_feedback(model_dir: str, n_rounds: int = 2, out_file: str = "pref_journal.jsonl"):
    """
    Interactive pairwise feedback collector.

    For each round:
     - Ask user for paragraph and question (you type).
     - Ask whether to auto-generate answers or to manually enter A/B.
     - If auto: generate two short answers (using different sampling params).
     - If manual: user types answers A and B directly.
     - Save each round as {"prompt": prompt, "a": a, "b": b, "choice": choice}
       into out_file (JSONL).
    """
    tokenizer, model = load_causal_model(model_dir)
    if torch.cuda.is_available():
        model.to("cuda")

    # load existing journal if present (so we append)
    journal = []
    if os.path.exists(out_file):
        try:
            journal = read_jsonl(out_file)
        except Exception:
            journal = []

    print("Pairwise preference collection.")
    print("You'll be asked to paste a paragraph and a question.")
    print("For answers: choose auto-generation or manual entry. Type 'q' to quit at any prompt.")

    for i in range(n_rounds):
        print(f"\n=== Round {i+1}/{n_rounds} ===")
        paragraph = input("Paste paragraph (or 'q' to quit):\n")
        if paragraph.strip().lower() == 'q':
            break
        question = input("Question (or 'q' to quit):\n")
        if question.strip().lower() == 'q':
            break

        prompt = build_prompt(paragraph, question)

        # Ask whether to auto-generate answers or input manually
        gen_or_manual = None
        while gen_or_manual not in ("g", "m", "q"):
            gen_or_manual = input("Generate answers with model or Manual entry? (g = generate, m = manual, q = quit): ").strip().lower()
            if gen_or_manual == 'q':
                break
        if gen_or_manual == 'q':
            break

        if gen_or_manual == "g":
            # produce two different answers by varying sampling strategy
            a = sample_answer(tokenizer, model, prompt, do_sample=True, temperature=0.9, max_new_tokens=64)
            b = sample_answer(tokenizer, model, prompt, do_sample=True, temperature=0.6, max_new_tokens=48)
            # if generation returned empty, prompt user to enter manually
            if not a:
                print("Model produced empty Answer A — please type Answer A manually:")
                a = input("Answer A:\n")
            if not b:
                print("Model produced empty Answer B — please type Answer B manually:")
                b = input("Answer B:\n")
        else:
            # manual input mode
            print("Enter Answer A (manual):")
            a = input().strip()
            print("Enter Answer B (manual):")
            b = input().strip()

        print("\n----- Answer A -----\n")
        print(a)
        print("\n----- Answer B -----\n")
        print(b)

        choice = None
        while choice not in ("a", "b", "q"):
            choice = input("Which is better? (a/b) or 'q' to quit: ").strip().lower()
            if choice == 'q':
                break
        if choice == 'q':
            break

        entry = {"prompt": prompt, "a": a, "b": b, "choice": choice}
        journal.append(entry)

        # append to file incrementally (so you don't lose data if script stops)
        write_jsonl(out_file, journal)
        print(f"Saved {len(journal)} preference(s) to {out_file}")

    print("Feedback collection finished.")


# ------------------------- PPO training -----------------------------

def ppo_train(policy_model_dir: str, reward_model_dir: str, out_dir: str, ppo_steps: int = 200, batch_size: int = 4):
    """Minimal PPO loop using trl. If `trl` is not installed, this function will raise an informative error.

    This code uses `trl`'s PPOTrainer abstraction. See TRL docs for production usage.
    """
    if not TRL_AVAILABLE:
        raise RuntimeError("`trl` not available. Install with `pip install trl` to run PPO training.")

    # Load tokenizer & policy
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(policy_model_dir)
    policy = AutoModelForCausalLM.from_pretrained(policy_model_dir)

    # load reward model
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_dir)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_dir, num_labels=1)
    reward_model.eval()

    # make sure models are on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    reward_model.to(device)

    # PPO config (very small / toy)
    ppo_config = PPOConfig(
        model_name=policy_model_dir,
        batch_size=batch_size,
        ppo_epochs=4,
        learning_rate=1.41e-5,
        log_with=None,
    )

    ppo_trainer = PPOTrainer(ppo_config, model=policy, tokenizer=tokenizer)

    # toy training loop: sample prompts from user paragraphs or a small dataset.
    print("Starting toy PPO loop. You will be prompted for paragraphs/questions; rewards are computed by the reward model.")

    for step in range(ppo_steps):
        # get prompt from user (or sample from file). We prompt user every iteration for simplicity.
        paragraph = input(f"[PPO step {step+1}/{ppo_steps}] Paste paragraph (or 'q' to quit):\n")
        if paragraph.strip().lower() == 'q':
            break
        question = input("Question:\n")
        prompt = build_prompt(paragraph, question)

        # tokenization
        query_tensors = tokenizer(prompt, return_tensors="pt")
        input_ids = query_tensors["input_ids"].to(device)

        # generate with the current policy (sampled)
        response_ids = policy.generate(input_ids, do_sample=True, max_new_tokens=128)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)[len(prompt):]

        # compute reward via reward model
        with torch.no_grad():
            r_input = reward_tokenizer(prompt + response, return_tensors="pt", truncation=True, max_length=512).to(device)
            score = reward_model(**r_input).logits.squeeze().item()

        # prepare ppo batch - trl's PPOTrainer expects queries, response tensors and rewards
        # convert to python lists as required by PPOTrainer
        query_texts = [prompt]
        response_texts = [response]
        rewards = [score]

        # run ppo step
        loss_dict = ppo_trainer.step(query_texts, response_texts, rewards)
        print(f"PPO step done. reward={score:.4f}, loss_info={loss_dict}")

    # save final policy
    ppo_trainer.save_pretrained(out_dir)
    print(f"Saved PPO-finetuned policy to {out_dir}")


# ------------------------------ Main --------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sft", "interact", "collect_feedback", "train_reward", "ppo"], required=True)
    parser.add_argument("--train_file", type=str, help="SFT train jsonl")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--out_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--model_dir", type=str, help="Directory with pretrained or SFT model")
    parser.add_argument("--feedback_file", type=str, help="Feedback jsonl file")
    parser.add_argument("--reward_model", type=str, help="reward model dir for ppo")
    parser.add_argument("--n_rounds", type=int, default=20)
    parser.add_argument("--ppo_steps", type=int, default=200)

    args = parser.parse_args()

    if args.mode == "sft":
        assert args.train_file, "--train_file is required for sft"
        fine_tune_sft(args.train_file, args.model_name, args.out_dir)

    elif args.mode == "interact":
        assert args.model_dir, "--model_dir required for interact"
        interactive_loop(args.model_dir)

    elif args.mode == "collect_feedback":
        assert args.model_dir, "--model_dir required for collect_feedback"
        collect_pairwise_feedback(args.model_dir, n_rounds=args.n_rounds, out_file=(args.feedback_file or "pref_journal.jsonl"))

    elif args.mode == "train_reward":
        assert args.feedback_file, "--feedback_file required for train_reward"
        train_reward_model(args.feedback_file, args.model_name or "bert-base-uncased", args.out_dir)

    elif args.mode == "ppo":
        assert args.model_dir and args.reward_model, "--model_dir and --reward_model required for ppo"
        ppo_train(args.model_dir, args.reward_model, args.out_dir or "checkpoints/ppo", ppo_steps=args.ppo_steps)


if __name__ == "__main__":
    main()


# -------------------- Test + Improve Mode -------------------------

def load_reward_model(reward_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(reward_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(reward_model_dir, num_labels=1)
    model.eval()
    return tokenizer, model


def generate_candidates(tokenizer, model, prompt: str, n: int = 4, max_new_tokens: int = 128):
    """Generate `n` candidate answers by varying temperature and sampling seeds."""
    temps = [0.9, 0.7, 0.5, 1.0]
    temps = temps[:n]
    answers = []
    for t in temps:
        ans = sample_answer(tokenizer, model, prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=t)
        answers.append(ans)
    # ensure uniqueness while preserving order
    seen = set()
    uniq = []
    for a in answers:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq


def score_candidates(reward_tokenizer, reward_model, prompt: str, candidates: List[str], device: str):
    inputs = [prompt + c for c in candidates]
    enc = reward_tokenizer(inputs, return_tensors="pt", truncation=True, padding=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = reward_model(**enc).logits.squeeze(-1)
        scores = logits.detach().cpu().tolist()
    # in case there's only one candidate, make scores a list
    if isinstance(scores, float):
        scores = [scores]
    return scores


def test_and_improve(model_dir: str, reward_model_dir: str = None, test_file: str = None, out_feedback: str = "pref_journal_test.jsonl", n_candidates: int = 4, do_ppo: bool = False):
    """Run a test dataset (or interactive prompts), show model answers, collect feedback, and optionally improve via reward-model reranking or an on-the-fly PPO step.

    Behavior:
      - If `test_file` is provided, iterate over its entries (jsonl with paragraph/question fields). Otherwise run interactively.
      - For each prompt: generate N candidates, score with reward model (if provided), show top answer to user.
      - Ask the user for feedback (y/n). If 'n', show top-K candidates and let them pick preferred one or type preferred text.
      - Save preference records to `out_feedback`.
      - If do_ppo is True and `trl` is available, perform a single-step PPO update on that example.
    """
    tokenizer, model = load_causal_model(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    reward_tokenizer = None
    reward_model = None
    if reward_model_dir:
        reward_tokenizer, reward_model = load_reward_model(reward_model_dir)
        reward_model.to(device)

    ppo_trainer = None
    if do_ppo:
        if not TRL_AVAILABLE:
            raise RuntimeError("trl not available; install `trl` to use do_ppo=True")
        # small ppo config for single-step updates
        ppo_config = PPOConfig(model_name=model_dir, batch_size=1, ppo_epochs=1, learning_rate=1e-5, log_with=None)
        ppo_trainer = PPOTrainer(ppo_config, model=model, tokenizer=tokenizer)

    records = []

    # load tests
    tests = None
    if test_file:
        tests = read_jsonl(test_file)
    
    idx = 0
    while True:
        if tests is not None:
            if idx >= len(tests):
                break
            entry = tests[idx]
            paragraph = entry.get("paragraph", "")
            question = entry.get("question", "")
            idx += 1
        else:
            paragraph = input("Paste paragraph (or 'q' to quit):")
            if paragraph.strip().lower() == 'q':
                break
            question = input("Question:")

        prompt = build_prompt(paragraph, question)
        candidates = generate_candidates(tokenizer, model, prompt, n=n_candidates)

        # if reward model exists, score and pick top
        if reward_model is not None and len(candidates) > 0:
            scores = score_candidates(reward_tokenizer, reward_model, prompt, candidates, device)
            ranked = sorted(list(zip(candidates, scores)), key=lambda x: x[1], reverse=True)
            best, best_score = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else None
        else:
            best = candidates[0] if candidates else ""
            best_score = None
            second = None

        print("=== Model answer (top-ranked) ===")
        print(best)
        if best_score is not None:
            print(f"(reward score = {best_score:.4f})")

        fb = None
        while fb not in ("y", "n", "s"):
            fb = input("Do you approve this answer? (y = yes, n = no, s = show candidates) ").strip().lower()
        picked = best
        choice = "auto"

        if fb == 's' or fb == 'n':
            print("Candidates:")
            for i, c in enumerate(candidates):
                sc = None
                if reward_model is not None:
                    sc = score_candidates(reward_tokenizer, reward_model, prompt, [c], device)[0]
                print(f"[{i}] {c} {f'(score={sc:.4f})' if sc is not None else ''}")
            pick = input("Pick index of preferred answer or type a better answer (or 'skip'):")
            if pick.strip().lower() == 'skip':
                # user skipped; keep auto
                choice = "skip"
            else:
                if pick.isdigit() and 0 <= int(pick) < len(candidates):
                    picked = candidates[int(pick)]
                    choice = f"cand_{pick}"
                else:
                    # user typed a custom answer
                    picked = pick
                    choice = "human_edit"

        # save record
        rec = {"prompt": prompt, "best": best, "best_score": best_score, "picked": picked, "choice": choice}
        records.append(rec)
        write_jsonl(out_feedback, records)
        print(f"Saved feedback so far to {out_feedback}")

        # optional on-the-fly PPO step using the selected answer as positive feedback
        if do_ppo and ppo_trainer is not None:
            if choice != 'skip':
                query_texts = [prompt]
                response_texts = [picked]
                # compute reward for the picked answer
                r_input = reward_tokenizer(prompt + picked, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    score = reward_model(**r_input).logits.squeeze().item()
                rewards = [score]
                loss_info = ppo_trainer.step(query_texts, response_texts, rewards)
                print(f"Performed on-the-fly PPO step. reward={score:.4f}, loss_info={loss_info}")

    # end while
    if do_ppo and ppo_trainer is not None:
        ppo_trainer.save_pretrained("checkpoints/ppo_interactive")
        print("Saved interactive PPO-updated policy to checkpoints/ppo_interactive")

    print("Test-and-improve session complete.")

# add to argparse options
# insert before main's parser.parse_args() call logic by replacing the main function section

# We will append support in the main() dispatch via a small patch below. Note: if you already ran main(), re-run the script to pick up this change.

def _dispatch_test_and_improve(args):
    test_and_improve(
        model_dir=args.model_dir,
        reward_model_dir=args.reward_model,
        test_file=args.test_file,
        out_feedback=(args.feedback_file or "pref_journal_test.jsonl"),
        n_candidates=args.candidates,
        do_ppo=args.do_ppo,
    )

# Extend main's argparse with new options safely by parsing sys.argv again if requested
if __name__ == "__main__":
    import sys
    # hack: if user passed --mode test_and_improve, handle it here to avoid changing original main too much
    if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "test_and_improve":
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["sft", "interact", "collect_feedback", "train_reward", "ppo", "test_and_improve"], required=True)
        parser.add_argument("--model_dir", type=str, help="Directory with pretrained or SFT model")
        parser.add_argument("--reward_model", type=str, help="reward model dir for scoring")
        parser.add_argument("--test_file", type=str, help="Optional test jsonl with paragraph/question fields")
        parser.add_argument("--feedback_file", type=str, help="output feedback jsonl")
        parser.add_argument("--candidates", type=int, default=4)
        parser.add_argument("--do_ppo", action='store_true')
        args = parser.parse_args()
        if args.mode == "test_and_improve":
            if not args.model_dir:
                raise ValueError("--model_dir required for test_and_improve")
            _dispatch_test_and_improve(args)
            sys.exit(0)

# End of file appended
