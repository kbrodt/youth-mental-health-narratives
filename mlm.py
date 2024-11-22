# https://huggingface.co/docs/transformers/en/training
import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import (
    KFold,
)
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    Trainer,
    TrainingArguments,
)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to data",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./submission.csv",
        help="Path to save",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        #default="vibhorag101/roberta-base-suicide-prediction-phr-v2",
        default="google/bigbird-roberta-base",
        help="Model name",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Local files only",
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default="./mlm",
        help="Model save dr",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",  # adamw_torch
        help="Optimizator",
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        help="bf16",
    )
    parser.add_argument(
        "--f16-eval",
        action="store_true",
        help="f16 full eval",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="tf32",
    )
    parser.add_argument(
        "--peft",
        action="store_true",
        help="peft",
    )
    parser.add_argument(
        "--grad-check",
        action="store_true",
        help="gradieng checkpointing",
    )
    parser.add_argument(
        "--find-unused",
        action="store_true",
        help="Find unused",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch compile",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--trust-code",
        action="store_true",
        help="trust remote code",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Chunk size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,  # 3e-5,
        help="Learning rate",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--grad-acc",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=0.01,
        help="Weight decay",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Folds",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=314159,
        help="Random seed",
    )

    return parser.parse_args(args)


def train_dev_split(df, fold=0, n_folds=5, seed=None):
    df["fold"] = None

    n_col = len(df.columns) - 1
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (_, dev_index) in enumerate(skf.split(df)):
        df.iloc[dev_index, n_col] = i

    train, dev = (
        df[df.fold != fold].copy(),
        df[df.fold == fold].copy(),
    )

    return train, dev


def to_text(data, out_fpath):
    text_LE = "\n".join(data.NarrativeLE.tolist())
    text_CME = "\n".join(data.NarrativeCME.tolist())
    text = "\n".join((text_LE, text_CME))
    with open(out_fpath, "w") as f:
        f.write(text)

    del data["NarrativeLE"]
    del data["NarrativeCME"]


def train(args):
    random_seed = args.seed

    data_path = Path(args.data_path)

    train = pd.read_csv(data_path / "train_features_X4juyT6.csv", index_col="uid")

    pattern = re.compile(r"\.([a-zA-Z])")
    train.NarrativeLE = train.NarrativeLE.apply(lambda text: re.sub(pattern, r". \1", text))
    train.NarrativeCME = train.NarrativeCME.apply(lambda text: re.sub(pattern, r". \1", text))
    train, test = train_dev_split(train, fold=args.fold, n_folds=args.n_folds, seed=args.seed)
    to_text(train, data_path / "train.txt")
    to_text(test, data_path / "test.txt")

    model_name = args.model_name
    is_llm = "mistral" in model_name.lower() or "lama" in model_name.lower() or "llm" in model_name.lower()
    if is_llm:
        # https://huggingface.co/docs/transformers/v4.45.1/quantization/bitsandbytes?bnb=4-bit#4-bit-qlora-algorithm
        # https://huggingface.co/mistralai/Mistral-7B-v0.1/discussions/116
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16

            #load_in_8bit=True,
        )
    else:
        quantization_config = None

    # https://github.com/huggingface/blog/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md#resources
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        #ignore_mismatched_sizes=True,
        local_files_only=args.local,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_code,
        #torch_dtype="auto",  # if model.half().save...
    )
    if "bigbird" in model_name.lower():
        for param in model.parameters():
            param.data = param.data.contiguous()

    if args.grad_check:
        model.gradient_checkpointing_enable(dict(use_reentrant=False) if args.grad_check else None)

    if is_llm:
        model.config.use_cache = False
        model.config.pad_token_id = model.config.eos_token_id
        print(model.config.pad_token_id)

        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        #model_max_length=2048,
        local_files_only=args.local,
        #add_prefix_space=True,
        trust_remote_code=args.trust_code,
    )
    if is_llm:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.pad_token_id)
        print(tokenizer.pad_token)

    print(tokenizer.model_max_length)
    print(model.config.max_position_embeddings)

    # https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=ga-fCOB4LI4W
    batch_size = args.batch_size
    train = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=str(data_path / "train.txt"),
        block_size=args.block_size,
    )

    test = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=str(data_path / "test.txt"),
        block_size=args.block_size,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,  # 0.3 for bigbird
    )

    if args.peft:
        # https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing#scrollTo=Ybeyl20n3dYH
        peft_config = LoraConfig(
            #target_modules="all-linear",
            #target_modules=["query_key_value"],
            #target_modules=[
            #    "q_proj",
            #    "v_proj",
            #],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,  # 16,
            lora_alpha=16,  # 8,
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model_output_dir = Path(args.model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_name).stem
    output_dir = model_output_dir / f"{model_name}_f{args.fold}_e{args.epochs}_b{batch_size}_ga{args.grad_acc}_o{args.optim}_lr{int(args.lr * 1e5)}_bs{args.block_size}"
    output_dir = str(output_dir)
    n = len(train) // batch_size
    print(n)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,  # 2e-5,
        warmup_steps=n,
        #lr_scheduler_type="cosine",
        #lr_scheduler_kwargs=dict(
        #    t_initial=args.epochs * n,
        #    warmup_t=n,
        #    cycle_limit=1,
        #    t_in_epochs=False,
        #    #lr_min=args.lr * 1e-2,
        #),
        #lr_scheduler_type="cosine_with_min_lr",
        #lr_scheduler_kwargs=dict(
        #    min_lr=args.lr * 1e-2,
        #),
        per_device_train_batch_size=batch_size,  # 4,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,  # 100,
        weight_decay=args.wd,  # 0.01,
        eval_strategy="steps",
        #save_strategy="epoch",
        eval_steps=500,
        logging_steps=n//2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        save_total_limit=2,
        #fp16=not args.bf16,
        #bf16=args.bf16,
        # https://discuss.pytorch.org/t/why-might-ddp-perform-worse-than-dp/112903/14
        ddp_find_unused_parameters=args.find_unused,
        seed=random_seed,
        # https://huggingface.co/docs/bitsandbytes/main/en/optimizers
        # https://huggingface.co/docs/transformers/en/perf_train_gpu_one#optimizer-choice
        optim=args.optim if not is_llm else "paged_adamw_8bit",
        torch_compile=args.compile,
        #eval_on_start=True,
        fp16_full_eval=args.f16_eval,
        tf32=args.tf32,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    try:
        trainer.train(
            resume_from_checkpoint=args.resume,
            #ignore_keys_for_eval=[f"eval_thresh_{i}" for i in range(len(BIN_COLS))],
        )
    except KeyboardInterrupt:
        pass

    if is_llm:
        model.dequantize()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    args = parse_args()
    print(args)
    train(args)


if __name__ == "__main__":
    main()
