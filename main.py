import polars as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
import datasets
import re
# from transformers import Trainer, TrainingArguments  # Not needed for inference
# from sklearn.model_selection import train_test_split  # Not needed for inference
# from sklearn.feature_extraction.text import TfidfVectorizer  # Not needed
# from sklearn.preprocessing import LabelEncoder  # Not needed
# from sklearn.metrics import accuracy_score, classification_report  # Not needed
# import tensorflow as tf  # Not needed

path = 'comparisons_validation.csv'

df = pl.read_csv(path)
# print(df.head(5))
# print(df.null_count())
df = df.drop_nulls()
# print(df['split'].unique())
# print(df['batch'].unique())
df = df.with_columns(df['info'].map_elements(eval).alias('infodict'))
df = df.with_columns(df['infodict'].map_elements(lambda x: x['post']).alias('post'))
df = df.with_columns(df['summaries'].map_elements(lambda x: eval(re.sub(r"\}\s+\{","},{", x))).alias('summarieslist'))
df = df.head(20000)

# Prepare dataset for inference
dataset = df.to_dicts()

# # No longer needed for inference-only
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenize your data
# def preprocess_function(examples):
#     inputs = ["summarize: " + (item if item is not None else "") for item in examples["post"]]
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)
#
#     labels = tokenizer(
#         [(item if item is not None else "") for item in examples["summaries"]],
#         max_length=150,
#         truncation=True,
#         padding=True
#     )
#
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
# # Convert to HuggingFace Dataset
# dataset_hf = datasets.Dataset.from_list(dataset)
#
# # Train/test split (HF style)
# dataset_split = dataset_hf.train_test_split(test_size=0.2, seed=42)
#
# # Tokenize datasets
# tokenized_train = dataset_split["train"].map(preprocess_function, batched=True)
# tokenized_val = dataset_split["test"].map(preprocess_function, batched=True)
#
# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     eval_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )
#
# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_val,
# )
#
# # Train
# trainer.train()


# ============ INFERENCE ============
print("\n" + "="*50)
print("GENERATING SUMMARIES")
print("="*50 + "\n")

# Load the trained model
try:
    trained_model = T5ForConditionalGeneration.from_pretrained("./results/checkpoint-12000")
    trained_tokenizer = T5Tokenizer.from_pretrained("./results/checkpoint-12000")
    # print("✓ Loaded checkpoint-12000")
except Exception as e:
    # print(f"Warning: Could not load checkpoint-12000 ({e})")
    # print("Loading final model from ./results...")
    trained_model = T5ForConditionalGeneration.from_pretrained("./results")
    trained_tokenizer = T5Tokenizer.from_pretrained("./results")
    # print("✓ Loaded model from ./results")

# Function to generate summary
def generate_summary(text, max_length=150):
    """Generate a summary for the given text."""
    if not text or len(text.strip()) == 0:
        return "[Empty input]"

    input_ids = trained_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = trained_model.generate(
        input_ids,
        max_length=max_length,
        min_length=10,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    summary = trained_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Handle empty summaries
    if not summary or len(summary.strip()) == 0:
        return "[No summary generated]"

    return summary

# Generate summaries for test samples
print("Sample summaries from test set:\n")

# Create dataset split for inference (only needed for test data)
dataset_hf = datasets.Dataset.from_list(dataset)
dataset_split = dataset_hf.train_test_split(test_size=0.2, seed=42)
test_data = dataset_split["test"]

for i in range(min(5, len(test_data))):
    post = test_data[i]["post"]
    actual_summary = test_data[i]["summaries"]
    predicted_summary = generate_summary(post)

    print(f"--- Sample {i+1} ---")
    print(f"Post (first 200 chars): {post[:200]}...")
    print(f"Actual Summary: {actual_summary}")
    print(f"Generated Summary: {predicted_summary}")
    print()
