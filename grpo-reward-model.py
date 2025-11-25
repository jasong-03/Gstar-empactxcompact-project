"""
COMPACT Training: SFT + GRPO for Empathetic Response Generation
Using EmpatheticDialogues dataset (loaded from local CSV)
"""

#%% Cell 1: Setup and Load Model
import os
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
    gpu_memory_utilization = 0.9,
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
LOCAL_DATA_PATH = "empathetic_data/empatheticdialogues"

# Load train data with error handling for malformed lines
print("Loading train.csv...")
try:
    # Try with quoting and error handling
    train_df = pd.read_csv(
        f"{LOCAL_DATA_PATH}/train.csv",
        on_bad_lines='skip',  # Skip malformed lines
        quoting=1,  # QUOTE_ALL
        engine='python'  # More flexible parser
    )
except Exception as e:
    print(f"First attempt failed: {e}")
    print("Trying with minimal options...")
    # Fallback: minimal options
    train_df = pd.read_csv(
        f"{LOCAL_DATA_PATH}/train.csv",
        on_bad_lines='skip',
        engine='python'
    )

# Load validation data and merge with training data
print("Loading valid.csv and merging...")
try:
    valid_df = pd.read_csv(
        f"{LOCAL_DATA_PATH}/valid.csv",
        on_bad_lines='skip',
        quoting=1,
        engine='python'
    )
except Exception as e:
    print(f"Validation load failed: {e}")
    print("Trying with minimal options...")
    valid_df = pd.read_csv(
        f"{LOCAL_DATA_PATH}/valid.csv",
        on_bad_lines='skip',
        engine='python'
    )

# Combine train and validation sets
train_df = pd.concat([train_df, valid_df], ignore_index=True)


print(f"✓ Loaded {len(train_df)} combined training samples")
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
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        warmup_steps = 10,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs/sft",
        report_to = "none",
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

#%% Cell 14: Reward Functions
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 3.0 if match_format.search(completion[0]["content"]) else 0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 1.0 if response.count(response_end) == 1 else -1.0
        scores.append(score)
    return scores

EMPATHY_KEYWORDS = [
    "sorry", "understand", "feel", "must be", "sounds like",
    "here for you", "support", "care", "glad", "happy for you"
]

def check_empathy_keywords(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"].lower()
        match = match_format.search(completion[0]["content"])
        if match:
            response_text = match.group(1).lower()
            count = sum(1 for kw in EMPATHY_KEYWORDS if kw in response_text)
            score = min(count * 0.5, 2.5)
        else:
            score = -1.0
        scores.append(score)
    return scores

def check_emotion_appropriateness(prompts, completions, emotion, **kwargs):
    # Expanded lists based on the 32 emotions in the dataset for more accurate reward
    NEGATIVE_EMOTIONS = [
        "sad", "anxious", "afraid", "angry", "disappointed", "annoyed", 
        "lonely", "terrified", "guilty", "disgusted", "furious", "jealous", 
        "devastated", "embarrassed", "ashamed", "apprehensive"
    ]
    POSITIVE_EMOTIONS = [
        "joyful", "excited", "proud", "grateful", "confident", "hopeful", 
        "impressed", "content", "caring", "trusting", "faithful", "prepared"
    ]
    
    # Slightly expanded keywords for better scoring
    SUPPORTIVE = ["sorry", "understand", "tough", "here for you", "that must be", "I can imagine"]
    CELEBRATORY = ["congratulations", "happy for you", "that's wonderful", "amazing", "fantastic"]
    
    scores = []
    for completion, user_emotion in zip(completions, emotion):
        match = match_format.search(completion[0]["content"])
        if match:
            text = match.group(1).lower()
            # Check if the user's emotion is in our categorized lists
            if user_emotion.lower() in NEGATIVE_EMOTIONS:
                # Reward supportive words for negative emotions
                score = sum(1.0 for w in SUPPORTIVE if w in text)
            elif user_emotion.lower() in POSITIVE_EMOTIONS:
                # Reward celebratory words for positive emotions
                score = sum(1.0 for w in CELEBRATORY if w in text)
            else:
                # Neutral score for ambiguous emotions like 'surprised' or 'nostalgic'
                score = 0.5
            score = min(score, 3.0)  # Cap the score
        else:
            # Penalize if the format is incorrect
            score = -1.0
        scores.append(score)
    return scores

def check_response_quality(completions, **kwargs):
    scores = []
    for completion in completions:
        match = match_format.search(completion[0]["content"])
        if match:
            text = match.group(1).strip()
            words = len(text.split())
            score = 2.0 if 10 <= words <= 50 else (1.0 if 5 <= words < 10 else -1.0)
            if "?" in text:
                score += 1.0
            score = min(score, 2.0)
        else:
            score = -1.0
        scores.append(score)
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
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 8,
    num_generations = 4,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 1000,
    save_steps = 200,
    report_to = "none",
    output_dir = "outputs/grpo",
)

#%% Cell 17: Initialize GRPO Trainer
print("\n=== Initializing GRPO Trainer ===")
grpo_trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_empathy_keywords,
        check_emotion_appropriateness,
        check_response_quality,
    ],
    args = training_args,
    train_dataset = grpo_dataset,
)

print("\n=== Starting GRPO Training ===")
grpo_trainer.train()

#%% Cell 18: Save LoRA
print("\n=== Saving LoRA ===")
model.save_lora("compact_grpo_lora")
print("✓ LoRA saved to: compact_grpo_lora/")

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
