import re

import datasets
import polars as pl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


MODEL_PATH = "./results/checkpoint-12000"
FALLBACK_MODEL_PATH = "./results"
BASE_TOKENIZER = "t5-small"
DEFAULT_SUMMARY_TOKENS = 120
MIN_SUMMARY_TOKENS = 20
MAX_SUMMARY_TOKENS = 1000
SINGLE_PASS_MAX_TOKENS = 220
INPUT_CHUNK_TOKENS = 430
LONG_DOCUMENT_MIN_TOKENS = 300
MAX_LONG_DOCUMENT_CHUNKS = 30


try:
    trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load checkpoint-12000 ({e})")
    print("Loading final model from ./results...")
    trained_model = T5ForConditionalGeneration.from_pretrained(FALLBACK_MODEL_PATH)

# Training checkpoints usually contain model weights/config, not tokenizer files.
# The model was trained from t5-small, so use the matching base tokenizer.
trained_tokenizer = T5Tokenizer.from_pretrained(BASE_TOKENIZER)
trained_model.eval()


def _clamp_summary_length(max_length):
    try:
        max_length = int(max_length)
    except (TypeError, ValueError):
        max_length = DEFAULT_SUMMARY_TOKENS

    return max(MIN_SUMMARY_TOKENS, min(max_length, MAX_SUMMARY_TOKENS))


def _sanitize_generated_summary(summary):
    summary = re.sub(r"\s+", " ", str(summary)).strip()
    summary = summary.strip("[]{} \t\r\n\"'")

    quote_chars = "'\"\u2018\u2019\u201c\u201d"
    text_label = re.match(rf"(?i)^text\s*[{re.escape(quote_chars)}]?\s*[:=]\s*[{re.escape(quote_chars)}]?", summary)
    if text_label:
        summary = summary[text_label.end():].strip()

    metadata_pattern = (
        rf"(?i)\s*(?:[,.;:!?-]+\s*)"
        rf"(?:[{re.escape(quote_chars)}]?\s*)"
        r"(?:policy|note|purpose|failured|failure|ref)\b"
        r".*$"
    )
    summary = re.sub(metadata_pattern, "", summary).strip()

    drift_markers = (
        r"\bn-+",
        r"\*{3,}",
        r"_{2,}",
        r"&[#a-z0-9]+;",
        r"={2,}",
        r">{2,}",
    )
    for marker in drift_markers:
        summary = re.split(marker, summary, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    sentence_end = max(summary.rfind("."), summary.rfind("?"), summary.rfind("!"))
    if sentence_end >= 40:
        summary = summary[:sentence_end + 1].strip()

    return summary.strip(" \t\r\n\"'\u2018\u2019\u201c\u201d")


def _chunk_text(text, chunk_size=INPUT_CHUNK_TOKENS):
    token_ids = trained_tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for start in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[start:start + chunk_size]
        chunk_text = trained_tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _generate_single_pass_summary(text, max_length):
    max_length = min(_clamp_summary_length(max_length), SINGLE_PASS_MAX_TOKENS)
    input_ids = trained_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        summary_ids = trained_model.generate(
            input_ids,
            max_length=max_length,
            min_length=max(5, min(40, int(max_length * 0.25))),
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            length_penalty=1.0,
        )

    return _sanitize_generated_summary(trained_tokenizer.decode(summary_ids[0], skip_special_tokens=True))


def _generate_long_document_summary(text, max_length):
    chunks = _chunk_text(text)
    if len(chunks) > MAX_LONG_DOCUMENT_CHUNKS:
        step = len(chunks) / MAX_LONG_DOCUMENT_CHUNKS
        chunks = [chunks[int(i * step)] for i in range(MAX_LONG_DOCUMENT_CHUNKS)]

    target_length = max(max_length, LONG_DOCUMENT_MIN_TOKENS)
    per_chunk_length = max(45, min(120, target_length // max(3, min(len(chunks), 10))))
    part_summaries = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_summary = _generate_single_pass_summary(chunk, per_chunk_length)
        if chunk_summary and chunk_summary != "[No summary generated]":
            part_summaries.append(f"Part {index}: {chunk_summary}")

    if not part_summaries:
        return "[No summary generated]"

    combined = "\n\n".join(part_summaries)
    combined_token_count = len(trained_tokenizer.encode(combined, add_special_tokens=False))
    if combined_token_count <= target_length:
        return combined

    keep_count = max(3, min(len(part_summaries), target_length // 60))
    if keep_count >= len(part_summaries):
        return combined

    step = (len(part_summaries) - 1) / max(1, keep_count - 1)
    selected = [part_summaries[round(i * step)] for i in range(keep_count)]
    return "\n\n".join(selected)


def generate_summary(text, max_length=DEFAULT_SUMMARY_TOKENS):
    if not text or len(text.strip()) == 0:
        return "[Empty input]"

    max_length = _clamp_summary_length(max_length)
    input_token_count = len(trained_tokenizer.encode(text, add_special_tokens=False))

    if input_token_count > INPUT_CHUNK_TOKENS:
        summary = _generate_long_document_summary(text, max_length)
    else:
        summary = _generate_single_pass_summary(text, max_length)

    if not summary or len(summary.strip()) == 0:
        return "[No summary generated]"

    return summary


if __name__ == "__main__":
    path = "comparisons_validation.csv"

    df = pl.read_csv(path)
    df = df.drop_nulls()
    df = df.with_columns(df["info"].map_elements(eval).alias("infodict"))
    df = df.with_columns(df["infodict"].map_elements(lambda x: x["post"]).alias("post"))
    df = df.with_columns(df["summaries"].map_elements(lambda x: eval(re.sub(r"\}\s+\{", "},{", x))).alias("summarieslist"))
    df = df.head(20000)

    dataset = df.to_dicts()
    dataset_hf = datasets.Dataset.from_list(dataset)
    dataset_split = dataset_hf.train_test_split(test_size=0.2, seed=42)
    test_data = dataset_split["test"]

    print("\n" + "=" * 50)
    print("GENERATING SUMMARIES")
    print("=" * 50 + "\n")
    print("Sample summaries from test set:\n")

    for i in range(min(5, len(test_data))):
        post = test_data[i]["post"]
        actual_summary = test_data[i]["summaries"]
        predicted_summary = generate_summary(post)

        print(f"--- Sample {i + 1} ---")
        print(f"Post (first 200 chars): {post[:200]}...")
        print(f"Actual Summary: {actual_summary}")
        print(f"Generated Summary: {predicted_summary}")
        print()
