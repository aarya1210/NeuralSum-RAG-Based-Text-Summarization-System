"""
Fine-tune T5-small on CNN/DailyMail summarization dataset.
Optimized for Colab T4 GPU — trains in ~8-12 minutes.
Run this ONCE before starting the web app.
"""

# ───────────────────────── IMPORTS ─────────────────────────

import os                  # for file paths and directory handling
import json                # to save metadata in JSON format
import torch               # PyTorch (deep learning framework)

from datasets import load_dataset   # to load dataset from HuggingFace

from transformers import (
    T5ForConditionalGeneration,     # T5 model for text generation (summarization)
    T5Tokenizer,                   # tokenizer to convert text ↔ tokens
    Seq2SeqTrainer,                # handles training loop automatically
    Seq2SeqTrainingArguments,      # training configuration
    DataCollatorForSeq2Seq,        # prepares batches (padding etc.)
    EarlyStoppingCallback,         # stops training if no improvement
)

import evaluate                    # for evaluation metrics (ROUGE)

# ───────────────────────── CONFIGURATION ─────────────────────────

MODEL_NAME    = "t5-small"     # pretrained model name
OUTPUT_DIR    = "./model_output"  # folder to save trained model

MAX_INPUT     = 256   # max length of input text (reduced for speed)
MAX_TARGET    = 64    # max length of output summary

BATCH_SIZE    = 32    # number of samples processed at once
GRAD_ACCUM    = 1     # gradient accumulation (not needed here)

LR            = 5e-4  # learning rate (controls learning speed)
NUM_EPOCHS    = 3     # number of training cycles

TRAIN_SAMPLES = 2000  # number of training samples (subset for faster training)
VAL_SAMPLES   = 200   # validation samples

# ───────────────────────── PREPROCESS FUNCTION ─────────────────────────

def preprocess(examples, tokenizer):
    """
    This function prepares input and output for the model.
    It converts raw text into tokenized format.
    """

    prefix = "summarize: "   # T5 requires task prefix

    # Add prefix to each article
    inputs  = [prefix + doc for doc in examples["article"]]

    # Actual summaries (target output)
    targets = examples["highlights"]

    # Tokenize input text (convert to numbers)
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT,
        truncation=True,   # cut long text
        padding=False
    )

    # Tokenize target summaries
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET,
        truncation=True,
        padding=False
    )

    # Attach labels to model input
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# ───────────────────────── METRICS FUNCTION ─────────────────────────

def compute_metrics(eval_pred, tokenizer, rouge):
    """
    This function evaluates model performance using ROUGE score.
    """

    preds, labels = eval_pred   # predictions and actual outputs

    # Sometimes predictions come as tuple → fix that
    if isinstance(preds, tuple):
        preds = preds[0]

    # Convert predictions from tokens → text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 (ignored tokens) with padding token
    labels = [
        [l if l != -100 else tokenizer.pad_token_id for l in lab]
        for lab in labels
    ]

    # Convert labels from tokens → text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # Compute ROUGE score
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Convert to percentage
    return {k: round(v * 100, 2) for k, v in result.items()}

# ───────────────────────── MAIN FUNCTION ─────────────────────────

def main():

    print("=" * 60)
    print("  Fine-tuning T5-small on CNN/DailyMail Summarization")
    print("=" * 60)

    # ─── Check GPU ───
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device.upper()}")

    # Warning if no GPU
    if device == "cpu":
        print("⚠️  WARNING: No GPU detected! Training will be very slow.")
        print("   In Colab: Runtime → Change runtime type → T4 GPU\n")

    # ─── Load Model & Tokenizer ───
    print(f"\n[1/5] Loading {MODEL_NAME} ...")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)  # load tokenizer
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)  # load model

    # ─── Load Dataset ───
    print("[2/5] Loading CNN/DailyMail dataset ...")

    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Use smaller subset for faster training
    dataset["train"]      = dataset["train"].select(range(TRAIN_SAMPLES))
    dataset["validation"] = dataset["validation"].select(range(VAL_SAMPLES))

    print(f"   Train: {len(dataset['train'])} samples | Val: {len(dataset['validation'])} samples")

    # ─── Tokenization ───
    print("[3/5] Tokenizing ...")

    tokenized = dataset.map(
        lambda ex: preprocess(ex, tokenizer),  # apply preprocess function
        batched=True,                          # process in batches
        num_proc=2,                            # parallel processing (faster)
        remove_columns=dataset["train"].column_names  # remove raw text columns
    )

    tokenized.set_format("torch")  # convert dataset to PyTorch format

    # ─── Training Configuration ───
    print("[4/5] Setting up trainer ...")

    use_fp16 = device == "cuda"   # use mixed precision if GPU available

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,                 # where to save model
        eval_strategy="epoch",                 # evaluate every epoch
        save_strategy="epoch",                 # save model every epoch
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,            # generate summaries during eval
        generation_max_length=MAX_TARGET,
        fp16=use_fp16,                        # faster training on GPU
        dataloader_num_workers=2,             # background data loading
        dataloader_pin_memory=use_fp16,       # faster CPU→GPU transfer
        load_best_model_at_end=True,          # keep best model
        metric_for_best_model="rouge2",       # choose best based on ROUGE-2
        logging_steps=20,
        save_total_limit=1,                   # keep only latest model
        report_to="none",
    )

    # Load ROUGE metric
    rouge = evaluate.load("rouge")

    # Data collator (handles padding)
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8
    )

    # ─── Trainer Setup ───
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=lambda ep: compute_metrics(ep, tokenizer, rouge),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ─── Training ───
    steps_per_epoch = len(dataset["train"]) // BATCH_SIZE
    print(f"[5/5] Training ... ({steps_per_epoch} steps/epoch × {NUM_EPOCHS} epochs)\n")

    trainer.train()   # model learns here

    # ─── Save Model ───
    trainer.save_model(OUTPUT_DIR)       # save trained model
    tokenizer.save_pretrained(OUTPUT_DIR)  # save tokenizer

    # ─── Save Metadata ───
    meta = {
        "base_model": MODEL_NAME,
        "dataset": "cnn_dailymail 3.0.0",
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES,
        "epochs": NUM_EPOCHS,
        "max_input": MAX_INPUT,
        "max_target": MAX_TARGET,
        "best_metric": "rouge2",
    }

    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Fine-tuning complete! Model saved to:", OUTPUT_DIR)
    print("   Now download the model_output/ folder and run:  python app.py")

# ───────────────────────── ENTRY POINT ─────────────────────────

if __name__ == "__main__":
    main()   # start execution