# https://huggingface.co/docs/transformers/en/training
import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from scipy.special import expit
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    KFold,
)
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cross_entropy,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from lovasz_losses import symmetric_lovasz


LABEL_COLS = [
    "DepressedMood",
    "MentalIllnessTreatmentCurrnt",
    "HistoryMentalIllnessTreatmnt",
    "SuicideAttemptHistory",
    "SuicideThoughtHistory",
    "SubstanceAbuseProblem",
    "MentalHealthProblem",

    "DiagnosisAnxiety",
    "DiagnosisDepressionDysthymia",
    "DiagnosisBipolar",
    "DiagnosisAdhd",

    "IntimatePartnerProblem",
    "FamilyRelationship",
    "Argument",
    "SchoolProblem",
    "RecentCriminalLegalProblem",

    "SuicideNote",
    "SuicideIntentDisclosed",
    "DisclosedToIntimatePartner",
    "DisclosedToOtherFamilyMember",
    "DisclosedToFriend",

    "InjuryLocationType",
    "WeaponType1",
]
BIN_COLS = LABEL_COLS[:-2]
CAT_COLS = LABEL_COLS[-2:]
N_CAT_INJ = 6
N_CAT_WEAP = 12

ID2LABEL = {
    i: l for i, l in enumerate(BIN_COLS)
}
ID2LABEL = ID2LABEL | {
    len(BIN_COLS) + i: f"{CAT_COLS[0]}_{i}" for i in range(N_CAT_INJ)
}
ID2LABEL = ID2LABEL | {
    len(BIN_COLS) + N_CAT_INJ + i: f"{CAT_COLS[1]}_{i:0>2}" for i in range(N_CAT_WEAP)
}
LABEL2ID = {
    l:i for i, l in ID2LABEL.items()
}


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
        default="allenai/longformer-base-4096",
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
        default="./model",
        help="Model save dr",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",  # adamw_torch
        help="Optimizator",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        help="scheduler",
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
        default=4,
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


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    p = torch.sigmoid(inputs)
    ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    loss = loss.mean()

    return loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        logits_bin, logits_1, logits_2 = logits.split([len(BIN_COLS), N_CAT_INJ, N_CAT_WEAP], dim=1)
        labels_bin, labels_1, labels_2 = labels.split([len(BIN_COLS), 1, 1], dim=1)
        labels_bin = labels_bin.float()
        loss_bin = (
            #binary_cross_entropy_with_logits(logits_bin, labels_bin)
            #sigmoid_focal_loss(logits_bin, labels_bin)
            symmetric_lovasz(logits_bin, labels_bin)
        )
        loss_1 = cross_entropy(logits_1, labels_1.squeeze(1))
        loss_2 = cross_entropy(logits_2, labels_2.squeeze(1))
        loss = loss_bin + loss_1 + loss_2

        return (loss, outputs) if return_outputs else loss


class ThreshCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not state.is_local_process_zero:
            return

        model = kwargs["model"]
        if hasattr(model, "module"):
            model = model.module

        if model.config.task_specific_params is None:
            model.config.task_specific_params = {}

        to_save = []
        for i in range(len(BIN_COLS)):
            thresh_name = f"eval_zthresh_{i:0>2}"
            score_name = f"eval_zf1_{i:0>2}"
            thresh = metrics.pop(thresh_name)
            score = metrics.pop(score_name)
            try:
                _, _, best_thresh, best_score, _ = model.config.task_specific_params[thresh_name]
            except KeyError:
                best_thresh, best_score = 0.5, 0.0

            if score >= best_score:
                best_thresh = thresh
                best_score = score
                to_save.append(score_name)

            model.config.task_specific_params[thresh_name] = [thresh, score, best_thresh, best_score, state.global_step]

        for j in [0, 1]:
            i = len(BIN_COLS) + j
            thresh_name = f"eval_zthresh_{i:0>2}"
            score_name = f"eval_zf1_{i:0>2}"
            score = metrics.pop(score_name)
            try:
                _, best_score, _ = model.config.task_specific_params[thresh_name]
            except KeyError:
                best_score = 0.0

            if score >= best_score:
                best_score = score
                to_save.append(score_name)

            model.config.task_specific_params[thresh_name] = [score, best_score, state.global_step]


def get_score(y_true, y_pred, threshold):
    if threshold is None:
        return f1_score(y_true, y_pred, average="micro")

    if len(y_pred.shape) > 1:
        return f1_score(y_true, (y_pred > threshold).astype("int"), average="macro")

    return f1_score(y_true, (y_pred > threshold).astype("int"), average="binary")


def threshold_search(y_train, y_train_hat):
    thresholds = np.linspace(0.0, 1.0, 100 + 1)
    scores = []
    for thresh in thresholds:
        score = get_score(y_train, y_train_hat, thresh)
        scores.append(score)

    thresh_ind = np.argmax(scores)
    thresh = thresholds[thresh_ind].item()
    score = scores[thresh_ind]

    return thresh, score


def compute_metrics(p):
    predictions = p.predictions
    labels = p.label_ids

    predictions_bin, predictions_1, predictions_2 = np.split(
        predictions,
        [len(BIN_COLS), len(BIN_COLS) + N_CAT_INJ],
        axis=1,
    )
    predictions_bin = expit(predictions_bin)
    labels_bin, labels_1, labels_2 = np.split(
        labels,
        [len(BIN_COLS), len(BIN_COLS) + 1],
        axis=1,
    )

    f1s = []
    scores = {}
    threshes = {}
    for c in range(labels_bin.shape[1]):
        thresh, score = threshold_search(labels_bin[:, c], predictions_bin[:, c])
        threshes[f"zthresh_{c:0>2}"] = thresh
        scores[f"zf1_{c:0>2}"] = score
        f1s.append(score)

    #binary_f1 = f1_score(
    #    labels_bin,
    #    (predictions_bin > 0.5).astype("int32"),
    #    average="macro",
    #)
    #f1s = [binary_f1]

    predictions_1 = np.argmax(predictions_1, axis=1)
    score = f1_score(labels_1, predictions_1, average="micro")
    scores[f"zf1_{len(BIN_COLS):0>2}"] = score
    f1s.append(score)

    predictions_2 = np.argmax(predictions_2, axis=1)
    score = f1_score(labels_2, predictions_2, average="micro")
    scores[f"zf1_{len(BIN_COLS) + 1:0>2}"] = score
    f1s.append(score)
    #score = (np.average(f1s, weights=[len(BIN_COLS), 1, 1]))
    score = np.average(f1s)

    return {
        "f1": score,
    } | scores | threshes


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


def train(args):
    random_seed = args.seed

    data_path = Path(args.data_path)

    train = pd.read_csv(data_path / "train_features_X4juyT6.csv", index_col="uid")
    labels = pd.read_csv(data_path / "train_labels_JxtENGl.csv", index_col="uid")
    labels[CAT_COLS] -= 1
    train = pd.concat(
        [
            train,
            labels,
        ],
        axis=1,
    )
    assert len(CAT_COLS) == 2
    assert sum(labels[c].nunique() for c in CAT_COLS) == N_CAT_INJ + N_CAT_WEAP
    num_labels = len(BIN_COLS) + sum(labels[c].nunique() for c in CAT_COLS)

    pattern = re.compile(r"\.([a-zA-Z])")
    train["text"] = train["NarrativeLE"] + " " + train["NarrativeCME"]
    train.text = train.text.apply(lambda text: re.sub(pattern, r". \1", text))

    train["labels"] = train[LABEL_COLS].values.tolist()
    for c in LABEL_COLS:
        del train[c]
    del train["NarrativeLE"]
    del train["NarrativeCME"]

    train, test = train_dev_split(train, fold=args.fold, n_folds=args.n_folds, seed=args.seed)
    print(train)
    print(test)

    train = Dataset.from_pandas(train)
    test = Dataset.from_pandas(test)

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
    assert num_labels == len(ID2LABEL) == len(LABEL2ID), (num_labels, len(ID2LABEL))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        #problem_type="multi_label_classification",
        local_files_only=args.local,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_code,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
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

    def preprocess_function(example):
        labels = example["labels"]
        example = tokenizer(example["text"], truncation=True)
        example["labels"] = labels

        return example

    # https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=ga-fCOB4LI4W
    batch_size = args.batch_size
    remove_columns = ["text", "uid", "fold"]
    train = train.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=remove_columns,
    )
    test = test.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=remove_columns,
    )
    max_len = []
    def __ml(x):
        nonlocal max_len
        max_len.append(len(x["input_ids"]))

        return x

    train.map(__ml)
    test.map(__ml)
    max_len = np.array(max_len)
    print(f"{np.min(max_len, axis=0)=}")
    print(f"{np.mean(max_len, axis=0)=}")
    print(f"{np.median(max_len, axis=0)=}")
    print(f"{np.percentile(max_len, q=50, axis=0)=}")
    print(f"{np.percentile(max_len, q=95, axis=0)=}")
    print(f"{np.max(max_len, axis=0)=}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
    model_name = Path(model_name).stem.split("_")[0]
    output_dir = model_output_dir / f"{model_name}_f{args.fold}_e{args.epochs}_b{batch_size}_ga{args.grad_acc}_o{args.optim}_lr{int(args.lr * 1e6)}_1e6"
    output_dir = str(output_dir)
    n = len(train) // batch_size
    print(n)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,  # 2e-5,
        warmup_steps=n,
        lr_scheduler_type=args.lr_scheduler_type,
        #lr_scheduler_type="cosine",
        #lr_scheduler_kwargs=dict(
        #    #num_warmup_steps=n,
        #    #num_training_steps=args.epochs,
        #    #num_cycles=0.5,
        #    #t_initial=args.epochs * n,
        #    #warmup_t=n,
        #    #cycle_limit=1,
        #    #t_in_epochs=False,
        #    ##lr_min=args.lr * 1e-2,

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
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=n//2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=not args.bf16,
        bf16=args.bf16,
        # https://discuss.pytorch.org/t/why-might-ddp-perform-worse-than-dp/112903/14
        ddp_find_unused_parameters=args.find_unused,
        seed=random_seed,
        # https://huggingface.co/docs/bitsandbytes/main/en/optimizers
        # https://huggingface.co/docs/transformers/en/perf_train_gpu_one#optimizer-choice
        optim=args.optim if not is_llm else "paged_adamw_8bit",
        #optim_args=None,
        torch_compile=args.compile,
        fp16_full_eval=args.f16_eval,
        tf32=args.tf32,
        #eval_on_start=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(ThreshCallback)

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
