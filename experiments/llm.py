import json
import logging
import re
import subprocess
import time
from pathlib import Path

root_dir = Path('.').cwd()
pkg_dir = str(root_dir / "assets")
subprocess.run(
    f"pip install --break-system-packages --no-index --find-links=file://{pkg_dir} protobuf".split()
)

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

APPLICATION_NAME = __name__
logger = logging.getLogger(APPLICATION_NAME)
logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    level=logging.INFO,
)
SUBMISSION_PATH = Path("submission.csv")
FEATURES_PATH = Path("data/test_features.csv")
SUBMISSION_FORMAT_PATH = Path("data/submission_format.csv")
#FEATURES_PATH = Path("data/train_features_X4juyT6.csv")
#SUBMISSION_FORMAT_PATH = Path("data/train_labels_JxtENGl.csv")
MODEL_DIR = Path("assets")
MAX_NEW_TOKENS = 650
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
HARD_EXAMPLES = [
    'ctko',
    'bihc',
    'aseb',
    'aajt',
    'cors',
    'fari',
    'bonz',
    'cezf',
    'dqgf',
    'bjhe',
]
QS = [
    "The person was perceived to be depressed at the time",
    "Currently in treatment for a mental health or substance abuse problem",
    "History of ever being treated for a mental health or substance abuse problem",
    "History of attempting suicide previously",
    "History of suicidal thoughts or plans",
    "The person struggled with a substance abuse problem",
    "The person had a mental health condition at the time",
    "Diagnosis Anxiety",
    "Diagnosis Depression or dysthymia",
    "Diagnosis Bipolar",
    "Diagnosis ADHD",
    "Problems with a current or former intimate partner appear to have contributed",
    "Relationship problems with a family member (other than an intimate partner) appear to have contributed",
    "An argument or conflict appears to have contributed",
    "Problems at or related to school appear to have contributed",
    "Criminal legal problem(s) appear to have contributed",
    "The person left a suicide note",
    "The person disclosed their thoughts and/or plans to die by suicide to someone else within the last month",
    "Intent was disclosed to a previous or current intimate partner",
    "Intent was disclosed to another family member",
    "Intent was disclosed to a friend",
    #"The type of place where the suicide took place (semicolon-separated list of possible values: House, apartment; Motor vehicle (excluding school bus and public transportation); Natural area (e.g., field, river, beaches, woods); Street/road, sidewalk, alley; Park, playground, public use area; Other)",
    #"Type of weapon used (semicolon-separated list of possible values: Firearm; Hanging, strangulation, suffocation; Poisoning; Fall; Other transport vehicle, eg, trains, planes, boats; Motor vehicle including buses, motorcycles; Drowning; Sharp instrument; Fire or burns; Blunt instrument; Unknown; Other (e.g. taser, electrocution, nail gun))",
    'The type of place where the suicide took place (choices: ["House, apartment", "Motor vehicle (excluding school bus and public transportation)", "Natural area (e.g., field, river, beaches, woods)", "Street/road, sidewalk, alley", "Park, playground, public use area", "Other"])"',
    'Type of weapon used (choices: ["Firearm", "Hanging, strangulation, suffocation", "Poisoning", "Fall", "Other transport vehicle, eg, trains, planes, boats", "Motor vehicle including buses, motorcycles", "Drowning", "Sharp instrument", "Fire or burns", "Blunt instrument", "Unknown", "Other (e.g. taser, electrocution, nail gun)"])',
]

def list_of_tuple_to_str(lst_of_tpl, sep=":"):
    return "\n".join(
        f"{a}{sep} {b}"
        for a, b in lst_of_tpl
    )

PROMPT_TEMPLATE = """You are an expert abstractor who reads law enforcement (LE) and coroner/medical examiner (CME) narratives about youth suicide and answers the questions. All questions are in the closed form. To most questions you answer either yes or no. Use example input questions and output answers.

{}

Answer yes or no to questions Q1-Q21. For questions Q22 and Q23, answer only from the list of possible values.

You must answer all 23 questions correctly in the same order. Do NOT output anything other than the answers. Do not include any explanation or summaries. Do not include any other questions and answers that aren't specified in the list.
""".format(
    #json.dumps(QS, indent=0).strip("}{").replace('"', "")
    list_of_tuple_to_str(
        [
            (f"Q{i}", q)
            for i, q in enumerate(QS, start=1)
        ],
        sep="."
    )
)

TRAIN_PROMT = """
-------------
EXAMPLE INPUT:
Narrative LE (a summary of the information in the law enforcement report):
{}

Narrative CME (a summary of the information in the coroner/medical examiner report):
{}

EXAMPLE OUTPUT:
{}
"""
TEST_PROMT = """
-------------
INPUT:
Narrative LE (a summary of the information in the law enforcement report):
{}

Narrative CME (a summary of the information in the coroner/medical examiner report):
{}

OUTPUT:
"""
LOCATIONS = {
    "house, apartment": 1,
    "motor vehicle (excluding school bus and public transportation)": 2,
    "natural area (e.g., field, river, beaches, woods)": 3,
    "park, playground, public use area": 4,
    "street/road, sidewalk, alley": 5,
    "other": 6,

    "residence": 1,
    "apartment": 1,
    "house": 1,
    "house, apartment (v": 1,
    'house, apartment (for': 1,

    "natural area": 3,
    "field": 3,
    "river": 3,
    "beach": 3,
    "beaches": 3,
    "woods": 3,

    "park": 4,
    "playground": 4,
    "public use area": 4,

    "street/road": 5,
    "sidewalk": 5,
    "alley": 5,
    "driveway": 5,
} | {
    'alleyway': 5,
    'barn': 6,
    'beach': 3,
    'bridge': 6,
    'church': 6,
    'dorm': 1,
    'hospital': 6,
    'house, apartment (for': 1,
    'jail cell': 6,
    'jail': 6,
    'prison cell': 6,
    'prison': 6,
    'school': 6,
    'treatment facility': 6,
    'unspecified': 6,
    'no information': 6,
    'no': 6,
} | {
    "dorm room": 6,
    "driveway": 5,
    "garage": 1,
    "house, apartment (v": 1,
    "indoor shooting range": 1,
    "mall": 6,
    "other (in-patient addiction facility)": 6,
    "parking lot": 6,
    "unknown": 6,
}

WEAPONS = {
    "blunt instrument": 1,
    "drowning": 2,
    "fall": 3,
    "fire or burns": 4,
    "firearm": 5,
    "hanging, strangulation, suffocation": 6,
    "motor vehicle including buses, motorcycles": 7,
    "other transport vehicle, eg, trains, planes, boats": 8,
    "poisoning": 9,
    "sharp instrument": 10,
    "other (e.g. taser, electrocution, nail gun)": 11,
    "unknown": 12,

    'carbon monoxide': 4,
    'carbon monoxide, fentanyl, despropionyl f': 4,
    'carbon monoxide inhalation': 4,
    'carbon monoxide intoxication': 4,
    'carbon monoxide poisoning': 9,
    'carbon monoxide toxicity and hyperthermia': 4,
    'carbon monoxide toxicity': 4,

    'firearm (for': 11,

    'other transport vehicle, eg, trains, planes, boats (not specified)': 8,
    'other transport vehicle, eg, trains, planes, boats, motor vehicle including bus': 8,
    'other transport vehicle, eg, trains, planes, boats (not applicable)': 8,
    'other transport vehicle, eg, trains, planes, boats, motor vehicle including bus': 8,
    'other transport vehicle, eg, trains, planes, boats, motorcycles (': 8,
    'other transport vehicle, eg, trains, planes, boats (crossbow)': 8,
} | {
    "hanging- brown leather belt - around v's beck - suicide": 6,
    '.380 caliber handgun': 5,
    '.410 break over single shot shotgun': 5,
    '.410 combination rifle': 5,
    '.45 caliber semi-automatic handgun': 5,
    '12 gauge shotgun (incorrect, but the answer provided is': 5,
    '12 gauge shotgun': 5,
    '40-caliber handgun': 5,
    'airplane': 8,
    'alcohol': 9,
    'asphyxia': 6,
    'blunt force injuries': 11,
    'blunt trauma of head, torso and extremities': 11,
    'bupropion and buspirone': 9,
    'bupropion': 9,
    'carbon monoxide': 9,
    'carbon monoxide, fentanyl, despropionyl f': 9,
    'carbon monoxide inhalation': 9,
    'carbon monoxide intoxication': 9,
    'carbon monoxide poisoning': 9,
    'carbon monoxide toxicity and hyperthermia': 9,
    'carbon monoxide toxicity': 9,
    'carbon monoxide': 9,
    'difluoroethane and diphenhydramine': 9,
    'diphenhydramine': 9,
    'drug toxicity': 9,
    'extreme blunt force traumatic injuries': 3,
    'firearm (for': 5,
    'gsw (gunshot wound)': 5,
    'gsw of the head': 5,
    'gsw to the head': 5,
    'gsw': 5,
    'hanging-strangulation-suffocation': 6,
    'intravenous drug paraphernalia': 9,
    'ligature hanging': 6,
    'ligature': 6,
    'medication overdose': 9,
    'mixed drug (fentanyl, sertraline, and am': 9,
    'mixed drug toxicity': 9,
    'mixed prescription medication': 9,
    'multiple blunt force injuries': 1,
    'multiple blunt impact injury - suicide': 1,
    'multiple blunt force injuries': 1,
    'multiple blunt force traumatic injuries': 1,
    'no weapon used. the person intentionally jumped from a 2nd story window': 3,
    'none (the person walked in front of oncoming traffic)': 7,
    'other (e.g. gasoline, self-immolation)': 4,
    'other transport vehicle, eg, trains, planes, boats (crossbow)': 8,
    'other transport vehicle, eg, trains, planes, boats (not applicable)': 8,
    'other transport vehicle, eg, trains, planes, boats (not specified)': 8,
    'other transport vehicle, eg, trains, planes, boats, motor vehicle including bus': 7,
    'other transport vehicle, eg, trains, planes, boats, motor vehicle including bus': 7,
    'other transport vehicle, eg, trains, planes, boats, motorcycles (': 8,
    'other': 11,
    'overdose of recently prescribed citalopram.': 9,
    'pills': 9,
    'plastic bag over the head': 6,
    'plastic bag': 6,
    'polysubstance drug toxicity': 9,
    'prescription medications': 9,
    'quetiapine': 9,
    'revolver': 5,
    'rifle': 5,
    'running motor vehicles in a closed garage': 6,
    'rx overdose': 9,
    'shotgun': 5,
    'sleep aid': 9,
    'suffocation': 9,
    'train': 8,
    'venlafaxine hcl pills': 9,
    'nan': 11,
    'no weapon specified, but journal indicates contemplation of suicide by shooting and jumping off': 2,
    'no weapon used (death by overdose)': 8,
    'no weapon used': 9,
    'no weapon used, the person fell unconscious and not breathing.': 9,
    'no': 11,
    'unknown': 12,
} | {
    "airplane": 8,
    "asphyxiation": 6,
    "blunt impact injuries of head, torso and extremities": 11,
    "blunt trauma of head, torso and extremities": 11,
    "bupropion": 9,
    "carbon monoxide toxicity": 9,
    "carbon monoxide": 9,
    "carbon monoxide, fentanyl, despropionyl f": 9,
    "carbon monoxide intoxication": 9,
    "carbon monoxide poisoning": 9,
    "carbon monoxide toxicity": 9,
    "carbon monoxide": 9,
    "combined drug toxicity": 9,
    "crossbow": 10,
    "diphenhydramine": 9,
    "drug toxicity": 9,
    "extreme blunt force traumatic injuries": 3,
    "firearm (revolver)": 5,
    "firearm (v": 5,
    "gsw of the head": 5,
    "gsw to the head": 5,
    "gsw": 5,
    "intravenous drug paraphernalia": 9,
    "ligature hanging": 6,
    "ligature": 6,
    "medication": 9,
    "mixed drug toxicity": 9,
    "multiple blunt force injuries": 11,
    "multiple blunt impact injury - suicide": 11,
    "nitrous oxide": 11,
    "no weapon used": 3,
    "no weapon used.": 3,
    "not applicable (drug overdose)": 9,
    "not applicable (vehicle crash)": 7,
    "not applicable (drug intoxication)": 9,
    "not applicable": 12,
    "other (e.g. methamphetamine)": 11,
    "other (e.g. diphenhydramine pills)": 11,
    "other transport vehicle, eg, trains, planes, boats (carbon monox": 8,
    "other transport vehicle, eg, trains, planes, boats, motor vehicle including bus": 8,
    "other transport vehicle, eg, trains, planes, boats, motor vehicle including bus": 8,
    "over the counter sleeping pills": 11,
    "overdose": 9,
    "paperclips": 11,
    "plastic bag over the head": 6,
    "plastic bag": 6,
    "polysubstance drug toxicity": 9,
    "prescription medications": 9,
    "quetiapine (prescription medication)": 11,
    "quetiapine": 11,
    "revolver": 5,
    "rifle": 5,
    "rx overdose": 9,
    "sharp instrument (v": 2,
    "shotgun": 5,
    "train": 11,
    "nan": 11,
    "no weapon found": 11,
    "no weapon specified": 11,
    "no weapon specified, but journal mentions guns and jumping off a cliff.": 2,
    "no weapon used": 11,
    "no weapon used, but the cause of death is tramadol intoxication": 9,
    "no weapon used, but the cause of death was adverse effects of drug combination": 11,
    "no weapon used, but there were baggies with what appears to be crystal": 11,
    "unknown": 12,
}


def load_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {MODEL_DIR}, {MODEL_DIR.exists()}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, device_map=DEVICE, local_files_only=True,
    )
    #model.to_bettertransformer()
    #or
    #model.generation_config.cache_implementation = "static"
    #model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer


def process_features(features):
    """
    Order features by ascending string length
    """
    features["str_len"] = features.NarrativeLE.str.len() + features.NarrativeCME.str.len()
    features = features.sort_values(by="str_len", ascending=False)
    return features.drop(columns=["str_len"])


def batch_features(features, batch_size: int):
    """
    Batch features together
    """
    if len(features) > batch_size:
        return np.array_split(features, int(len(features) / batch_size))
    return [features]


def predict_on_batch(feature_batch, model, tokenizer, tr_prompt):
    """
    Tokenize input batches, generate and decode outputs
    """
    # Tokenize input narratives (NarrativeLE) in batch

    prompts = [
        "\n".join(
            (
                PROMPT_TEMPLATE,
                tr_prompt,
                TEST_PROMT.format(
                    item.NarrativeLE,
                    item.NarrativeCME,
                    #repr(item[LABEL_COLS].to_dict()),
                ),
            )
        )
        for _, item in feature_batch.iterrows()
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs.to("cuda")

    # Generate outputs for variables
    #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Remove prompt from output
    decoded = [resp[len(prompt) :] for resp, prompt in zip(decoded, prompts)]

    return decoded


def predict_on_batch_qa(feature_batch, model, tokenizer, tr_prompt):
    """
    Tokenize input batches, generate and decode outputs
    """
    # Tokenize input narratives (NarrativeLE) in batch

    past_key_values = None
    prompts = [
        "\n".join(
            (
                PROMPT_TEMPLATE,
                tr_prompt,
                TEST_PROMT.format(
                    item.NarrativeLE,
                    item.NarrativeCME,
                    #repr(item[LABEL_COLS].to_dict()),
                ),
            )
        )
        for _, item in feature_batch.iterrows()
    ]
    #from ipdb import set_trace; set_trace()

    all_decoded = [[] for _ in prompts]

    for i, q in enumerate(QS, start=1):
        prompts = [
            f"{prompt}\nQuestion {i}: {q}\nAnswer:"
            for prompt in prompts
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs.to("cuda")
        # Generate outputs for variables
        # TODO: KV-cache https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization#32-the-key-value-cache
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=3 if i <= len(LABEL_COLS[:-2]) else 15+1,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
        )
        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        _prompts = []
        for j, (resp, prompt) in enumerate(zip(decoded, prompts)):
            prefix, a = resp[:len(prompt)], resp[len(prompt):].rstrip()
            a = re.sub(r"Question.*", "", a).rstrip()
            if i <= len(QS) - 2:
                a = " yes" if "yes" in a.lower() else " no"
            else:
                if a.strip().lower() == "nan":
                    a = " unknown"

            all_decoded[j].append(f"Q:{a}")
            _prompts.append(f"{prefix}{a}")
        prompts = _prompts
        #prompts = [
        #    "\n".join(prompt.splitlines()[:-1])
        #    if "question" in prompt.splitlines()[-1].lower()
        #    else prompt
        #    for prompt in prompts
        #]
        #prompts = [
        #    prompt.rstrip()
        #    for prompt in prompts
        #]
        past_key_values = outputs.past_key_values

    # Remove prompt from output
    #decoded = [
    #    resp[len_prompt:]
    #    for len_prompt, resp in zip(len_prompts, prompts)
    #]
    decoded = [
        "\n".join(d)
        for d in all_decoded
    ]

    return decoded


def parse_response(output):
    """
    Transform response into a json object using minimal cleaning
    """
    output = [
        line.split(":")[-1].strip()
        for line in output.splitlines()
        if len(line) > 0
    ]
    output = dict(zip(LABEL_COLS, output))
    return output
    try:
        # Try loading the raw string into 
        resp = json.loads(output)
        return resp
    except json.JSONDecodeError:
        pass
    try:
        # Get rid of extra trailing sections that follow "--"
        split_output = output.split("--")[0]
        resp = json.loads(split_output)
        return resp
    except json.JSONDecodeError:
        pass
    try:
        # Get rid of sections that follow the a closing bracket "}"
        split_output = output.split("}")[0] + "}"
        resp = json.loads(split_output)
        return resp
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse {output} into valid json")
        return None


def process_injury_location(data: pd.Series):
    """
    Transform InjuryLocationType model output to integers, fill in default for invalid outputs
    """
    ilt = data.str.lower().map(LOCATIONS)
    if ilt.isna().any():
        logger.warning(
            f"There are unexpected values in injury location: {data[ilt.isna()].unique()} "
        )
        ilt = ilt.fillna(6)  # Fill with other

    return ilt.astype("int32")


def process_weapon_type(data: pd.Series):
    """
    Transform WeaponType1 model output to integers, fill in default for invalid outputs
    """
    wt = data.str.lower().map(WEAPONS)
    if wt.isna().any():
        logger.warning(
            f"There are unexpected values in weapon type: {data[wt.isna()].unique()} "
        )
        wt = wt.fillna(11)  # Fill with other

    return wt.astype("int32")


def generate_predictions(features: pd.DataFrame, submission_format: pd.DataFrame, train) -> pd.DataFrame:
    chat = [
        {
            "role": "system",
            "content": PROMPT_TEMPLATE,
        },

        {
            "role": "assistant",
            "content": "I'm doing great. How can I help you today?",
        },
        {
            "role": "user",
            "content": "I'd like to show off how chat templating works!",
        },
    ]
    #tokenizer.apply_chat_template(chat, tokenize=False)
    tr_prompt = "\n".join(
        (
            TRAIN_PROMT.format(
                item.NarrativeLE,
                item.NarrativeCME,
                (
                    "".join(
                        (
                            f"Question {i}: {q}\nAnswer: {a}\n"
                            for i, (q, a) in enumerate(
                                zip(
                                    QS,
                                    item[LABEL_COLS].to_dict().values(),
                                ),
                                start=1,
                            )
                        )
                    )
                )
                #"\n".join(item[LABEL_COLS].to_dict().values())
                #list_of_tuple_to_str(
                #    zip(
                #        [a for a, _ in QS],
                #        item[LABEL_COLS].to_dict().values(),
                #    )
                #)
                #list_of_tuple_to_str(
                #    zip(
                #        [f"Q{i}. {q}" for i, q in enumerate(QS, start=1)],
                #        item[LABEL_COLS].to_dict().values(),
                #    )
                #)
                #json.dumps(
                #    dict(
                #        zip(
                #            [q for _, q in QS],
                #            item[LABEL_COLS].to_dict().values(),
                #        ),
                #    ),
                #    indent=0
                #).strip("}{").replace('"', ""),
            )
            for _, item in train.head(3).iterrows()
        )
    )

    # Load model
    model, tokenizer = load_model()
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Batch inputs
    BATCH_SIZE = 2
    df = process_features(features)
    data_batches = batch_features(df, BATCH_SIZE)

    responses = []
    idxs = []
    logger.info(f"Iterating over {len(data_batches)} batches")
    for ix, data_batch in enumerate(data_batches):
        logger.info(f"Generating predictions on batch {ix}, with {len(data_batch)} samples")
        responses += predict_on_batch_qa(data_batch, model, tokenizer, tr_prompt)
        #print(responses[-1])
        idxs += list(data_batch.index)
    logger.info(f"Finished inference")
    interim_preds = pd.DataFrame({"string_output": responses}, index=df.index)

    # Get submission-ready solution
    idxs = []
    parsed_resps = []
    could_not_parse = []

    for row in interim_preds.itertuples():
        parsed = parse_response(row.string_output)
        if type(parsed) == dict:
            idxs.append(row.Index)
            parsed_resps.append(parsed)
        else:
            idxs.append(row.Index)
            could_not_parse.append(row.Index)
            # Fill any we couldn't parse with placeholder values for now
            parsed_resps.append(
                {
                    "DepressedMood": 0,
                    "IntimatePartnerProblem": 0,
                    "FamilyRelationship": 0,
                    "Argument": 0,
                    "MentalIllnessTreatmentCurrnt": 0,
                    "HistoryMentalIllnessTreatmnt": 0,
                    "SuicideAttemptHistory": 0,
                    "SuicideThoughtHistory": 0,
                    "SuicideNote": 0,
                    "SubstanceAbuseProblem": 0,
                    "SchoolProblem": 0,
                    "RecentCriminalLegalProblem": 0,
                    "SuicideIntentDisclosed": 0,
                    "DisclosedToIntimatePartner": 0,
                    "DisclosedToOtherFamilyMember": 0,
                    "DisclosedToFriend": 0,
                    "MentalHealthProblem": 0,
                    "DiagnosisAnxiety": 0,
                    "DiagnosisDepressionDysthymia": 0,
                    "DiagnosisBipolar": 0,
                    "DiagnosisAdhd": 0,
                    "WeaponType1": "Firearm",
                    "InjuryLocationType": "House, apartment",
                }
            )

    if len(could_not_parse) > 0:
        logger.warning(
            f"Could not parse {len(could_not_parse)} rows. Indices: {could_not_parse}"
        )

    parsed_preds = pd.DataFrame(parsed_resps, index=pd.Index(idxs, name="uid")).fillna(0)
    #parsed_preds.to_csv("pre_submit3.csv", index=True)
    for c in LABEL_COLS[:-2]:
        ppc = parsed_preds[c].str.lower().map(
            {
                "no": 0,
                "yes": 1,
                "no information given": 0,
                "not specified": 0,
                "n/a": 0,
            }
        )
        if ppc.isna().any():
            logger.warning(
                f"There are unexpected values in {c}: {parsed_preds[c][ppc.isna()].unique()} "
            )
            ppc = ppc.fillna(0)

        parsed_preds[c] = ppc.astype("int32")

    parsed_preds["InjuryLocationType"] = process_injury_location(parsed_preds.InjuryLocationType)
    parsed_preds["WeaponType1"] = process_weapon_type(parsed_preds.WeaponType1)

    # Make sure column order is the same as in the submission format
    parsed_preds = parsed_preds[submission_format.columns]

    # Make sure the row order is the same as in the submission format
    parsed_preds = parsed_preds.loc[features.index]

    # Make sure all values are int and not NaN
    parsed_preds = parsed_preds.round().astype("int32")

    return parsed_preds


def main():
    train = pd.read_csv(MODEL_DIR / "train_features_X4juyT6.csv", index_col="uid")

    labels = pd.read_csv(MODEL_DIR / "train_labels_JxtENGl.csv", index_col="uid")
    for c in LABEL_COLS[:-2]:
        labels[c] = labels[c].map({0: "no", 1: "yes"})

    labels["InjuryLocationType"] = labels.InjuryLocationType.map(
        {
            1: "House, apartment",
            2: "Motor vehicle (excluding school bus and public transportation)",
            3: "Natural area (e.g., field, river, beaches, woods)",
            4: "Park, playground, public use area",
            5: "Street/road, sidewalk, alley",
            6: "Other",
        }
    )
    labels["WeaponType1"] = labels.WeaponType1.map(
        {
            1: "Blunt instrument",
            2: "Drowning",
            3: "Fall",
            4: "Fire or burns",
            5: "Firearm",
            6: "Hanging, strangulation, suffocation",
            7: "Motor vehicle including buses, motorcycles",
            8: "Other transport vehicle, eg, trains, planes, boats",
            9: "Poisoning",
            10: "Sharp instrument",
            11: "Other (e.g. taser, electrocution, nail gun",
            12: "Unknown",
        }
    )
    train = pd.concat(
        [
            train,
            labels,
        ],
        axis=1,
    )
    #train = train.loc[HARD_EXAMPLES]

    features = pd.read_csv(FEATURES_PATH, index_col=0)
    #train = train[~train.index.isin(features.index)].copy()
    print(f"Loaded train features of shape {train.shape}")
    print(f"Loaded test features of shape {features.shape}")

    #preds = pd.read_csv("preds.csv", index_col=0)
    #preds["InjuryLocationType"] = preds.InjuryLocationType.map(
    #    {
    #        1: "House, apartment",
    #        2: "Motor vehicle (excluding school bus and public transportation)",
    #        3: "Natural area (e.g., field, river, beaches, woods)",
    #        4: "Park, playground, public use area",
    #        5: "Street/road, sidewalk, alley",
    #        6: "Other",
    #    }
    #)
    #preds["WeaponType1"] = preds.WeaponType1.map(
    #    {
    #        1: "Blunt instrument",
    #        2: "Drowning",
    #        3: "Fall",
    #        4: "Fire or burns",
    #        5: "Firearm",
    #        6: "Hanging, strangulation, suffocation",
    #        7: "Motor vehicle including buses, motorcycles",
    #        8: "Other transport vehicle, eg, trains, planes, boats",
    #        9: "Poisoning",
    #        10: "Sharp instrument",
    #        11: "Other (e.g. taser, electrocution, nail gun",
    #        12: "Unknown",
    #    }
    #)
    #features = pd.concat(
    #    [
    #        features,
    #        preds,
    #    ],
    #    axis=1,
    #)

    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col=0)
    print(f"Loaded submission format of shape: {submission_format.shape}")

    # Generate predictions
    predictions = generate_predictions(features, submission_format, train)
    print(f"Saving predictions of shape {predictions.shape} to {SUBMISSION_PATH}")
    predictions.to_csv(SUBMISSION_PATH, index=True)


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Generated in {time.time() - start_time} seconds.")
    print(bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))
