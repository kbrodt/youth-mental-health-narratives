import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


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
        "--model-dir",
        type=str,
        default="./assets",
        help="Model dir",
    )

    return parser.parse_args(args)


def inference(args):
    data_path = Path(args.data_path)

    pattern = re.compile(r"\.([a-zA-Z])")
    test = pd.read_csv(data_path / "test_features.csv", index_col="uid")
    test["text"] = test["NarrativeLE"] + " " + test["NarrativeCME"]
    test.text = test.text.apply(lambda text: re.sub(pattern, r". \1", text))
    submission = pd.read_csv(data_path / "submission_format.csv", index_col="uid")

    test["str_len"] = test.text
    test = test.sort_values(by="str_len", ascending=False)
    test = test.drop(columns=["str_len"])

    test = Dataset.from_pandas(test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    all_threshes = []
    predictions_b = []
    predictions_c1 = []
    predictions_c2 = []
    for model_name in sorted(model_dir.iterdir()):
        print(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            #torch_dtype=torch.float16,
            use_safetensors=True,
            device_map=device,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()

        def predict_fn(example):
            inputs = tokenizer(example["text"], padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)

            logits = outputs.logits
            logits_bin, logits_1, logits_2 = logits.split([len(BIN_COLS), N_CAT_INJ, N_CAT_WEAP], dim=1)
            logits_bin = logits_bin.sigmoid()
            logits_1 = logits_1.softmax(dim=1)
            logits_2 = logits_2.softmax(dim=1)

            logits = torch.cat([logits_bin, logits_1, logits_2], dim=1)

            example["output"] = logits

            return example

        preds = test.map(
            predict_fn,
            batched=True,
            batch_size=16,
        )
        predictions_bin, predictions_1, predictions_2 = np.split(
            np.array(preds["output"]),
            [len(BIN_COLS), len(BIN_COLS) + N_CAT_INJ],
            axis=1,
        )
        threshes = np.array(
            [
                model.config.task_specific_params[f"eval_zthresh_{i:0>2}"][0]  # [t, s, bt, bs, gs]
                for i, _ in enumerate(BIN_COLS)
            ]
        )
        all_threshes.append(threshes)
        predictions_b.append(predictions_bin)
        predictions_c1.append(predictions_1)
        predictions_c2.append(predictions_2)

    all_threshes = np.mean(all_threshes, axis=0)
    predictions_b = np.mean(predictions_b, axis=0)
    predictions_b = (predictions_b > all_threshes).astype("int64")
    predictions_c1 = np.mean(predictions_c1, axis=0).argmax(axis=1) + 1
    predictions_c2 = np.mean(predictions_c2, axis=0).argmax(axis=1) + 1

    submission.loc[test["uid"], BIN_COLS] = predictions_b.tolist()
    submission.loc[test["uid"], CAT_COLS[0]] = predictions_c1.tolist()
    submission.loc[test["uid"], CAT_COLS[1]] = predictions_c2.tolist()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(save_path, index=True)


def main():
    args = parse_args()
    inference(args)


if __name__ == "__main__":
    main()
