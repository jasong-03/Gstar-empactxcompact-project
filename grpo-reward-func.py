"""
COMPACT Training: SFT + GRPO for Empathetic Response Generation
Using EmpatheticDialogues dataset (loaded from local CSV)
"""

#%% Cell 1: Setup and Load Model
import os
from dotenv import load_dotenv
import comet_ml
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Login to Hugging Face
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("Logging in to Hugging Face Hub...")
    login(token=hf_token)
    print("✓ Logged in.")

# Initialize Comet ML
comet_api_key = os.getenv("COMET_API_KEY")
if comet_api_key:
    print("Initializing Comet ML...")
    os.environ["COMET_PROJECT_NAME"] = "compact-empathetic-response-reward"
    comet_ml.login(api_key=comet_api_key)
    print("✓ Comet ML initialized.")

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from unsloth import FastLanguageModel
import torch

max_seq_length = 1024
lora_rank = 32

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.95,  # ⬆️ Tăng từ 0.9 → 0.95
    dtype = torch.float16,  # ✨ Explicitly set FP16
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = lora_rank*2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

#%% Cell 2: Define Custom Tags
emotion_context_start = "<EMOTION_CONTEXT>"
emotion_context_end = "</EMOTION_CONTEXT>"
response_start = "<RESPONSE>"
response_end = "</RESPONSE>"

system_prompt = \
f"""You are an empathetic AI companion.
When given a user's statement and emotion context, provide a supportive, dignified, and empathetic response.
Place the emotion context between {emotion_context_start} and {emotion_context_end}.
Place your empathetic response between {response_start} and {response_end}."""

print(system_prompt)

#%% Cell 3: Create Chat Template (FIXED with json.dumps)
import json

# Escape for Jinja2 (handles newlines and quotes)
system_prompt_escaped = json.dumps(system_prompt)
response_start_escaped = json.dumps(response_start)

chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ " + system_prompt_escaped + " + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ " + response_start_escaped + " }}"
    "{% endif %}"
)

tokenizer.chat_template = chat_template
print("✓ Chat template applied")

#%% Cell 4: Test Chat Template
example = tokenizer.apply_chat_template([
    {"role": "user", "content": f"I just lost my job.\n{emotion_context_start}sadness{emotion_context_end}"},
    {"role": "assistant", "content": f"{response_start}I'm so sorry to hear that.{response_end}"},
], tokenize=False, add_generation_prompt=True)
print("\n=== Template Test ===")
print(example[:200] + "...")

#%% Cell 5: Load EmpatheticDialogues from Local CSV
import pandas as pd
import numpy as np

print("\n=== Loading EmpatheticDialogues from Local CSV ===")
# Update this path for your server
LOCAL_DATA_PATH = "/home/cursor-remote/empathetic_data/empatheticdialogues"

# Load train data with robust error handling
print("Loading train.csv...")
try:
    # Try with comprehensive options for EmpatheticDialogues format
    train_df = pd.read_csv(
        f"{LOCAL_DATA_PATH}/train.csv",
        on_bad_lines='skip',  # Skip malformed lines
        engine='python',  # More flexible parser
        encoding='utf-8',  # Explicit encoding
        keep_default_na=True,  # Handle empty fields
        na_values=['', 'NA', 'null'],  # Empty tags → NaN
    )
    print(f"✅ Loaded train.csv with Python engine")
except Exception as e:
    print(f"Python engine failed: {e}")
    print("Trying C engine with skip...")
    # Fallback: C engine (faster but less flexible)
    try:
        train_df = pd.read_csv(
            f"{LOCAL_DATA_PATH}/train.csv",
            on_bad_lines='skip',
            encoding='utf-8',
        )
        print(f"✅ Loaded train.csv with C engine")
    except Exception as e2:
        print(f"C engine also failed: {e2}")
        print("Last resort: warn mode...")
        # Last resort: show warnings but continue
        train_df = pd.read_csv(
            f"{LOCAL_DATA_PATH}/train.csv",
            on_bad_lines='warn',
            engine='python',
        )
        print(f"✅ Loaded train.csv with warnings")

# Load validation data and merge into training
print("\nLoading valid.csv to merge into training...")
try:
    valid_df = pd.read_csv(
        f"{LOCAL_DATA_PATH}/valid.csv",
        on_bad_lines='skip',
        engine='python',
        encoding='utf-8',
        keep_default_na=True,
        na_values=['', 'NA', 'null'],
    )
    print(f"✅ Loaded valid.csv with Python engine")
    # Merge validation into training
    train_df = pd.concat([train_df, valid_df], ignore_index=True)
    print(f"✅ Merged validation data into training")
except Exception as e:
    print(f"Warning: Could not load valid.csv: {e}")
    print("Continuing with train.csv only...")

print(f"✓ Total training samples after merge: {len(train_df)}")
print(f"Columns: {train_df.columns.tolist()}")
print(f"\nFirst row:")
print(train_df.head(1))

# Check unique emotions
print(f"\nUnique emotions: {train_df['context'].nunique()}")
print(f"Sample emotions: {train_df['context'].unique()[:10]}")

#%% Cell 6: Format Dataset for SFT
def format_empathetic_dataset(x):
    """Format EmpatheticDialogues row for training"""
    user_statement = str(x["prompt"])
    emotion = str(x["context"])
    empathetic_response = str(x["utterance"])
    
    user_input = f"{user_statement}\n{emotion_context_start}{emotion}{emotion_context_end}"
    assistant_response = f"{response_start}{empathetic_response}{response_end}"
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_response},
    ]

print("\n=== Formatting Dataset ===")
train_df["Messages"] = train_df.apply(format_empathetic_dataset, axis=1)
print(f"✓ Formatted {len(train_df)} samples")

#%% Cell 7: Filter by Length for SFT
train_df["N"] = train_df["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
sft_max_length = max_seq_length // 2

sft_df = train_df.loc[train_df["N"] <= sft_max_length].copy()
print(f"\n=== SFT Dataset Filtering ===")
print(f"Max length: {sft_max_length} tokens")
print(f"After filtering: {len(sft_df)} samples")

# Sample for faster training
sft_sample_size = 500
if len(sft_df) > sft_sample_size:
    sft_df = sft_df.sample(n=sft_sample_size, random_state=3407)
print(f"SFT training samples: {len(sft_df)}")

#%% Cell 8: Convert to HuggingFace Dataset
from datasets import Dataset

sft_df["text"] = tokenizer.apply_chat_template(
    sft_df["Messages"].values.tolist(), 
    tokenize=False
)
sft_dataset = Dataset.from_pandas(sft_df[["text", "Messages"]])
print(f"\n=== SFT Dataset Ready ===")
print(sft_dataset)

#%% Cell 9: SFT Training
from trl import SFTTrainer, SFTConfig

print("\n=== Starting SFT Phase ===")
sft_trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = sft_dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 8,  # ⬆️ Tăng từ 2 → 8
        gradient_accumulation_steps = 4,   # ⬆️ Tăng từ 2 → 4
        # Effective batch size = 8 × 4 = 32
        warmup_steps = 10,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        logging_steps = 5,  # ⬇️ Giảm để log thường xuyên hơn
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs/sft",
        report_to = "none",
        fp16 = True,  # ✨ BẬT mixed precision
        dataloader_num_workers = 4,  # ✨ Parallel data loading
    ),
)

print("Starting SFT training...")
sft_trainer.train()

#%% Cell 10: Test After SFT
print("\n=== Testing Model After SFT ===")
test_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"I just got accepted!\n{emotion_context_start}excited{emotion_context_end}"},
]

text = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)

from transformers import TextStreamer
print("\n=== Generated Response ===")
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    temperature = 0.7,
    max_new_tokens = 256,
    streamer = TextStreamer(tokenizer, skip_prompt=True),
)

#%% Cell 11: Clean Up Before GRPO
del sft_dataset, sft_df
torch.cuda.empty_cache()
import gc
gc.collect()

#%% Cell 12: Load Full Dataset for GRPO
print("\n=== Loading Full Dataset for GRPO ===")
grpo_df = train_df.copy()

# Map for GRPO format
grpo_data = []
for idx, row in grpo_df.iterrows():
    grpo_data.append({
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{row['prompt']}\n{emotion_context_start}{row['context']}{emotion_context_end}"},
        ],
        "expected_response": row["utterance"],
        "emotion": row["context"],
    })

grpo_dataset = Dataset.from_list(grpo_data)
print(f"GRPO dataset size: {len(grpo_dataset)}")

#%% Cell 13: Define Regex for Reward Functions
import re

response_end_regex = r"</RESPONSE>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
match_format = re.compile(
    rf"{response_start}(.+?){response_end_regex}[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

# Test
test_text = f"{response_start}I'm sorry to hear that.{response_end}"
print(f"\n=== Testing Regex ===")
print(f"Match: {match_format.findall(test_text)}")

#%% Cell 14: Reward Functions (Hybrid - Heuristic + BERT-GoEmotions)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

print("Loading BERT-GoEmotions reward model...")
reward_tokenizer = AutoTokenizer.from_pretrained("codewithdark/bert-Gomotions")
reward_model = AutoModelForSequenceClassification.from_pretrained("codewithdark/bert-Gomotions").cuda()
reward_model.eval()
print("✓ Loaded reward model: codewithdark/bert-Gomotions")

# List of emotion labels from GoEmotions
goemotion_labels = reward_model.config.id2label
print(f"Loaded {len(goemotion_labels)} GoEmotion labels:", list(goemotion_labels.values())[:10])

#--------------------------------------------
# Heuristic reward functions
#--------------------------------------------
EMPATHY_KEYWORDS = [
    "sorry", "understand", "feel", "must be", "sounds like", "can imagine",
    "here for you", "support", "care", "glad", "happy for you", "that's tough",
    "how are you", "i'm here", "it's okay", "tell me more", "that sounds",
    "i hear you", "makes sense", "understandable"
]
NEGATIVE_KEYWORDS = ["stupid", "dumb", "whatever", "don't care", "not my problem"]

def match_format_exactly(completions, **kwargs):
    """Reward for having <RESPONSE> tags (softer check)"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        if response_start in response and response_end in response:
            score = 2.0
        elif response_start in response:
            score = 1.0
        else:
            score = 0.0
        scores.append(score)
    return scores

def check_response_quality(completions, **kwargs):
    """Check length and engagement"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0
        text = response.replace(response_start, "").replace(response_end, "").strip()
        if len(text) > 0:
            words = len(text.split())
            if 5 <= words <= 100:
                score += 1.0
            if "?" in text:
                score += 0.5
            if any(word in text.lower() for word in ["you", "your", "how"]):
                score += 0.5
        scores.append(max(0, min(2.0, score)))
    return scores

def check_empathy_keywords(completions, **kwargs):
    """Heuristic reward for empathy language"""
    scores = []
    for completion in completions:
        response = completion[0]["content"].lower()
        score = 0
        empathy_count = sum(1 for kw in EMPATHY_KEYWORDS if kw in response)
        score += min(empathy_count * 0.3, 2.0)
        negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in response)
        score -= negative_count * 1.0
        scores.append(max(-1.0, min(2.0, score)))
    return scores

#--------------------------------------------
# BERT-GoEmotions reward
#--------------------------------------------
@torch.no_grad()
def bert_emotion_reward(prompts, completions, emotion, **kwargs):
    """
    Use BERT-GoEmotions to score how well response emotion matches target context emotion.
    Returns reward in [-1, +2]
    """
    scores = []
    for completion, target_emotion in zip(completions, emotion):
        text = completion[0]["content"]
        inputs = reward_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
        logits = reward_model(**inputs).logits
        probs = torch.sigmoid(logits)[0]  # multi-label sigmoid
        # Map target emotion (context) to GoEmotions label if available
        target_emotion = target_emotion.lower()
        match_idx = None
        for i, label in goemotion_labels.items():
            if target_emotion in label.lower():
                match_idx = i
                break

        if match_idx is not None:
            emotion_score = probs[match_idx].item()
        else:
            # fallback: mean of top-5
            emotion_score = probs.topk(5).values.mean().item()

        # scale reward roughly to [-1, +2]
        reward = emotion_score * 3 - 1
        scores.append(float(max(-1.0, min(2.0, reward))))
    return scores

#%% Cell 15: Filter Dataset by Length
tokenized = grpo_dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched = True,
)
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

maximum_length = int(np.quantile(tokenized["L"], 0.9))
print(f"\n=== Filtering GRPO Dataset ===")
print(f"Max prompt length (90th percentile): {maximum_length}")

grpo_dataset = grpo_dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized
print(f"GRPO dataset after filtering: {len(grpo_dataset)}")

#%% Cell 16: GRPO Training Configuration
max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 0.9,
    top_k = 50,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = 0.8,
    learning_rate = 5e-6,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 4,  # ⬆️ Tăng từ 1 → 4
    gradient_accumulation_steps = 8,   # ⬆️ Tăng từ 4 → 8
    num_generations = 4,
    # Effective batch size = 4 × 8 × 4 = 128 generations
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 1000,
    save_steps = 200,
    report_to = "none",
    output_dir = "outputs/grpo",
    fp16 = True,  # ✨ BẬT mixed precision
    dataloader_num_workers = 2,  # ✨ Parallel loading
)

#%% Cell 17: Initialize GRPO Trainer
print("\n=== Initializing GRPO Trainer ===")
grpo_trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        check_response_quality,
        check_empathy_keywords,   # Heuristic
        bert_emotion_reward,      # 🔥 BERT-GoEmotions
    ],
    args = training_args,
    train_dataset = grpo_dataset,
)

print("\n=== Starting GRPO Training ===")
grpo_trainer.train()

#%% Cell 18: Save LoRA and Upload to Hub
print("\n=== Saving LoRA ===")
model.save_lora("compact_grpo_lora")
print("✓ LoRA saved to: compact_grpo_lora/")

if hf_token:
    print("\n=== Uploading to Hugging Face Hub ===")
    try:
        model.push_to_hub_lora("jasong03/compact-grpo-lora-qwen3-4b-reward", private=True)
        print("✓ LoRA adapter pushed to Hugging Face Hub")
    except Exception as e:
        print(f"Could not push to hub. Error: {e}")

#%% Cell 19: Test Final Model
print("\n=== Testing Final Model ===")
test_cases = [
    ("My dog passed away.", "devastated"),
    ("I got promoted!", "proud"),
    ("I'm worried about tomorrow.", "anxious"),
]

from vllm import SamplingParams
sampling_params = SamplingParams(temperature=0.7, top_k=50, max_tokens=256)

for statement, emotion in test_cases:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{statement}\n{emotion_context_start}{emotion}{emotion_context_end}"},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("compact_grpo_lora"),
    )[0].outputs[0].text
    
    print(f"\n{'='*60}")
    print(f"Input: {statement} ({emotion})")
    print(f"Output: {output}")

print("\n" + "="*60)
print("🎉 Training Complete!")
print("="*60)
