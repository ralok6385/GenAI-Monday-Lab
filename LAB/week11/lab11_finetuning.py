import torch, math
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from transformers import set_seed

set_seed(42)

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"


# -----------------------------
# LOAD MODEL
# -----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id


# -----------------------------
# TEXT GENERATION FUNCTION
# -----------------------------
def generate_text(model, tokenizer, prompt):
    model.eval()
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=80,
            temperature=0.8,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)


# -----------------------------
# BASELINE OUTPUT
# -----------------------------
print("\n=== BASELINE OUTPUT ===")

review_prompts = [
    "This product is",
    "I bought this phone and",
    "The quality of this item"
]

baseline = {}

for p in review_prompts:
    baseline[p] = generate_text(model, tokenizer, p)
    print(f"\nPrompt: {p}")
    print(f"Output: {baseline[p]}")


# -----------------------------
# DATASET
# -----------------------------
corpus = [
    "this phone has an amazing battery life and the camera quality is outstanding for the price.",
    "i bought this laptop for college and it handles all my assignments perfectly.",
    "the sound quality of these headphones is incredible with deep bass.",
    "this smartwatch tracks my steps accurately and is very reliable.",
    "great wireless earbuds with noise cancellation.",
    "the keyboard is comfortable for long typing sessions.",
    "this charger is very useful during travel.",
    "the tablet screen is bright and colorful.",
    "i love this fitness tracker for daily goals.",
    "this bluetooth speaker gives clear audio."
]

dataset = Dataset.from_dict({"text": corpus})

tokenized = dataset.map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=128),
    batched=True
)

split = tokenized.train_test_split(test_size=0.2)


# -----------------------------
# TRAINING
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./review_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=5,
    save_strategy="epoch"   # ✅ FIXED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    data_collator=data_collator
)

trainer.train()

# ✅ SAVE MODEL
trainer.save_model("./review_model")


# -----------------------------
# AFTER TRAINING OUTPUT
# -----------------------------
print("\n=== AFTER FINE-TUNING ===")

for p in review_prompts:
    ft_output = generate_text(model, tokenizer, p)
    
    print(f"\nPrompt: {p}")
    print(f"Baseline: {baseline[p][:80]}")
    print(f"Fine-Tuned: {ft_output[:80]}")


# =============================
# RECIPE MODEL
# =============================

tokenizer2 = GPT2Tokenizer.from_pretrained("gpt2")
model2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

tokenizer2.pad_token = tokenizer2.eos_token
model2.config.pad_token_id = tokenizer2.eos_token_id


print("\n=== BASELINE RECIPES ===")

recipe_prompts = [
    "To make butter chicken",
    "For pasta carbonara",
    "To prepare chocolate cake"
]

baseline2 = {}

for p in recipe_prompts:
    baseline2[p] = generate_text(model2, tokenizer2, p)
    print(f"\nPrompt: {p}")
    print(f"Output: {baseline2[p]}")


recipes = [
    "to make butter chicken marinate chicken with spices.",
    "heat butter and cook onions.",
    "add tomato and cook well.",
    "add chicken and cook fully.",
    "serve with naan.",
    
    "boil pasta until soft.",
    "fry pancetta.",
    "mix eggs and cheese.",
    "combine everything.",
    "serve hot."
]

dataset2 = Dataset.from_dict({"text": recipes})

tokenized2 = dataset2.map(
    lambda x: tokenizer2(x["text"], padding="max_length", truncation=True, max_length=128),
    batched=True
)

split2 = tokenized2.train_test_split(test_size=0.2)


collator2 = DataCollatorForLanguageModeling(tokenizer=tokenizer2, mlm=False)

args2 = TrainingArguments(
    output_dir="./recipe_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=5,
    save_strategy="epoch"   # ✅ FIXED
)

trainer2 = Trainer(
    model=model2,
    args=args2,
    train_dataset=split2["train"],
    eval_dataset=split2["test"],
    data_collator=collator2
)

trainer2.train()

# ✅ SAVE MODEL
trainer2.save_model("./recipe_model")


print("\n=== AFTER FINE-TUNING RECIPES ===")

for p in recipe_prompts:
    ft = generate_text(model2, tokenizer2, p)
    
    print(f"\nPrompt: {p}")
    print(f"Baseline: {baseline2[p][:80]}")
    print(f"Fine-Tuned: {ft[:80]}")