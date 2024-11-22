import argparse
import functools as f
import itertools as it
import logging
import multiprocessing
import random
import re
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
)
from sklearn.metrics import (
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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
#LABEL_COLS = [
#    #"DiagnosisBipolar",
#    #"DiagnosisAnxiety",
#    #"DiagnosisAdhd",
#
#    #"MentalHealthProblem",
#    #"DiagnosisDepressionDysthymia",
#
#    #"Argument",
#
#    #"SuicideNote",
#
#    #"IntimatePartnerProblem",
#
#    #"HistoryMentalIllnessTreatmnt",
#
#    #"SuicideAttemptHistory",
#    #"SuicideThoughtHistory",
#
#    #"MentalIllnessTreatmentCurrnt",
#
#    #"SubstanceAbuseProblem",
#
#    #"DepressedMood",
#
#    #"SchoolProblem",
#
#    #"RecentCriminalLegalProblem",
#
#    #"FamilyRelationship",
#
#    "SuicideIntentDisclosed",
#
#    ##"RecentCriminalLegalProblem",
#
#    ##"FamilyRelationship",
#
#    "DisclosedToIntimatePartner",
#    "DisclosedToFriend",
#    "DisclosedToOtherFamilyMember",
#
#    "InjuryLocationType",
#    "WeaponType1",
#]
BIN_COLS = LABEL_COLS[:-2]
CAT_COLS = LABEL_COLS[-2:]
LGB_MAX_BIN_COLS = {
    "MentalIllnessTreatmentCurrnt": 1023,

    "SuicideIntentDisclosed": 1023,
    "DisclosedToIntimatePartner": 511,
    "DisclosedToOtherFamilyMember": 1023,
    "DisclosedToFriend": 1023,
}
APPLICATION_NAME = __name__
logger = logging.getLogger(APPLICATION_NAME)
logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    level=logging.INFO,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
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
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds",
    )

    # Vectorizers' parameters
    parser.add_argument(
        "--max-features-word",
        type=int,
        default=100_000,  # 250_000,  # 50_000
        help="Number of features for word level TfIdf",
    )
    parser.add_argument(
        "--ngram-word",
        type=str,
        default="1,3",  # "1,2",
        help="Number of n-grams for word level TfIdf",
    )
    parser.add_argument(
        "--max-features-char",
        type=int,
        default=100_000,  # 350_000
        help="Number of features for char level TfIdf",
    )
    parser.add_argument(
        "--ngram-char",
        type=str,
        default="3,6",  # 3,5
        help="Number of n-grams for char level TfIdf",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=314159,
        help="Random seed",
    )

    return parser.parse_args(args)


def get_score(y_true, y_pred, threshold):
    if threshold is None:
        return f1_score(y_true, y_pred, average="micro")

    if len(y_pred.shape) > 1:
        return f1_score(y_true, (y_pred > threshold).astype("int"), average="macro")

    return f1_score(y_true, (y_pred > threshold).astype("int"), average="binary")


def average_f1(predictions: pd.DataFrame, labels: pd.DataFrame):
    """Score a set of predictions using the competition metric. F1 is averaged
    across all target variables. For categorical variables, micro-averaged
    F1 score is used.

    Args:
        predictions (pd.DataFrame): Dataframe of predictions, with one column
            for each target variable. The index should be the uid.
        labels (pd.DataFrame): Dataframe of ground truth values, with one column
            for each target variable. The index should be the uid.
    """
    # Check that there are 23 target variables
    assert predictions.shape[1] == 23

    # Check that column order and row order are the same
    assert (predictions.columns == labels.columns).all()
    assert (predictions.index == labels.index).all()

    # All values should be integers
    assert (predictions.dtypes == int).all()

    CATEGORICAL_VARS = ["InjuryLocationType", "WeaponType1"]
    BINARY_VARS = np.setdiff1d(labels.columns, CATEGORICAL_VARS)

    # Calculate F1 score averaged across binary variables
    binary_f1 = f1_score(
        labels[BINARY_VARS],
        predictions[BINARY_VARS],
        average="macro",
    )
    f1s = [binary_f1]

    # Calculate F1 score for each categorical variable
    for cat_col in CATEGORICAL_VARS:
        f1s.append(f1_score(labels[cat_col], predictions[cat_col], average="micro"))

    return np.average(f1s, weights=[len(BINARY_VARS), 1, 1])


def average_f1(y_cross_val_gt, y_cross_val_hat):
    binary_f1 = f1_score(
        y_cross_val_gt[:, :-2],
        y_cross_val_hat[:, :-2],
        average="macro",
    )
    f1s = [binary_f1]
    f1s.append(f1_score(y_cross_val_gt[:, -2], y_cross_val_hat[:, -2], average="micro"))
    f1s.append(f1_score(y_cross_val_gt[:, -1], y_cross_val_hat[:, -1], average="micro"))
    score = (np.average(f1s, weights=[len(BIN_COLS), 1, 1]))

    return score


def make_folds(df, cols, cv_fn, n_folds=5, seed=None):
    y = df[cols].values

    cv = cv_fn(n_splits=n_folds, random_state=seed, shuffle=True)
    #df["fold"] = -1
    #n_cols = len(df.columns)
    train_dev_inds = []
    for i, (train_index, dev_index) in enumerate(cv.split(range(len(df)), y)):
        #df.iloc[dev_index, n_cols - 1] = i
        train_dev_inds.append((train_index, dev_index))

    return cv, y, train_dev_inds


def plot_scores(scores, score=-1.0, score_random=-1.0, out=None, to_show=True):
    x, y = zip(*scores)
    if score == -1.0:
        score = np.mean(y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"Score: {score:.4} (random {score_random:.4})")
    rects = ax.bar(x, y)
    ax.bar_label(rects, [f"{_y:.4}" for _y in y], padding=0)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(ls="--")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out)
    #fig.show()
    if to_show:
        plt.show()

    plt.close(fig)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def threshold_search(y_train, y_train_hat):
    thresholds = np.linspace(0.0, 1.0, 100 + 1)
    with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as p:
        scores = list(
            p.imap(
                f.partial(get_score, y_train, y_train_hat),
                thresholds
            )
        )

    thresh_ind = np.argmax(scores)
    thresh = thresholds[thresh_ind].item()
    score = scores[thresh_ind]

    return thresh, score


def main():
    args = parse_args()
    set_seed(args.seed)
    logger.info(args)

    data_dir = Path(args.data_dir)
    logger.info(f"Reading train from {data_dir}..")
    train = pd.read_csv(data_dir / "train_features_X4juyT6.csv", index_col="uid")
    labels = pd.read_csv(data_dir / "train_labels_JxtENGl.csv", index_col="uid")
    labels[CAT_COLS] -= 1

    models_dir = Path("./assets")
    models_dir.mkdir(parents=True, exist_ok=True)

    # https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
    # https://stackoverflow.com/questions/27924813/extracting-clusters-from-seaborn-clustermap
    #correlations = labels.corr()
    #correlations_array = np.asarray(correlations)
    #method = "median"
    #row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method=method)
    #col_linkage = hierarchy.linkage(distance.pdist(correlations_array.T), method=method)
    #g = sns.clustermap(
    #    correlations,
    #    row_linkage=row_linkage,
    #    col_linkage=col_linkage,
    #    method=method,
    #    figsize=(12, 12),
    #    #cmap="RdBu",
    #    annot=True,
    #    annot_kws={"size": 8},
    #)
    ##g = sns.clustermap(
    ##    labels.corr(),
    ##    method="complete",
    ##    cmap="RdBu",
    ##    annot=True,
    ##    annot_kws={"size": 8},
    ##)
    ###plt.setp(g.ax_heatmap.get_xticklabels(), rotation=70);
    ##g.figure.show()
    #g.figure.savefig(models_dir / "corr.png")
    #plt.show()
    #plt.close(g.figure)

    train = pd.concat(
        [
            train,
            labels,
        ],
        axis=1,
    )

    train["corpus"] = train["NarrativeLE"] + " " + train["NarrativeCME"]
    pattern = re.compile(r"\.([a-zA-Z])")
    train.corpus = train.corpus.apply(lambda text: re.sub(pattern, r". \1", text))
    #train.corpus = train.corpus.apply(lambda text: re.sub(r"[^a-zA-Z0-9\s]", "", text))

    logger.info(f"Reading test from {data_dir}..")
    test = pd.read_csv(data_dir / "test_features.csv", index_col="uid")
    test["corpus"] = test["NarrativeLE"] + " " + test["NarrativeCME"]
    test.corpus = test.corpus.apply(lambda text: re.sub(pattern, r". \1", text))
    #test.corpus = test.corpus.apply(lambda text: re.sub(r"[^a-zA-Z0-9\s]", "", text))
    submission = pd.read_csv(data_dir / "submission_format.csv", index_col="uid")

    #llm_preds = pd.read_csv("s3_fixed.csv", index_col=0)
    #logger.warning("ONLY FOR LOCAL TESTS")
    #train = train[~train.index.isin(test.index)].copy()
    #llm_preds_test = llm_preds[llm_preds.index.isin(test.index)].copy()
    #llm_preds_test = llm_preds_test.loc[test.index]
    #llm_preds_train = llm_preds[~llm_preds.index.isin(test.index)].copy()
    #llm_preds_train = llm_preds_train.loc[train.index]

    ### feature extraction
    if args.max_features_word > 0:
        logger.info(
            f"Creating word level TfIdf vectorizer with params: ngram_range={args.ngram_word},"
            f" max_features={args.max_features_word}"
        )
        tfidf_word_vec = TfidfVectorizer(
        #tfidf_word_vec = CountVectorizer(
            ngram_range=tuple(map(int, args.ngram_word.split(","))),
            analyzer="word",
            max_features=args.max_features_word,
            min_df=10,
            max_df=0.6,
            stop_words="english",

            sublinear_tf=True,
            #dtype=np.float32,

            #binary=True,
            #use_idf=False,
            #norm=None,
        )

        logger.info("Training word tfidf..")
        tfidf_word_vec.fit(
            it.chain(
                train.corpus,
                #train.NarrativeLE,
                #train.NarrativeCME,
                #test.corpus,
                #test.NarrativeLE,
                #test.NarrativeCME,
            )
        )
        joblib.dump(tfidf_word_vec, models_dir / "tfidf_word_vec.job")
    else:
        tfidf_word_vec = None

    if args.max_features_char > 0:
        logger.info(
            f"Creating char level TfIdf vectorizer with params: ngram_range={args.ngram_char},"
            f" max_features={args.max_features_char}"
        )
        tfidf_char_vec = TfidfVectorizer(
        #tfidf_char_vec = CountVectorizer(
            ngram_range=tuple(map(int, args.ngram_char.split(","))),
            analyzer="char_wb",
            max_features=args.max_features_char,
            min_df=10,
            max_df=0.6,

            sublinear_tf=True,
            #dtype=np.float32,

            #binary=True,
            #use_idf=False,
            #norm=None,
        )

        logger.info("Training char tfidf..")

        tfidf_char_vec.fit(
            it.chain(
                train.corpus,
                #train.NarrativeLE,
                #train.NarrativeCME,
                #test.corpus,
                #test.NarrativeLE,
                #test.NarrativeCME,
            )
        )
        joblib.dump(tfidf_char_vec, models_dir / "tfidf_char_vec.job")
    else:
        tfidf_char_vec = None

    assert not (tfidf_word_vec is None and tfidf_char_vec is None), "One tfidf must be present"

    logger.info(f"Transforming train {len(train)}..")
    x_tr = []
    if tfidf_word_vec is not None:
        x_tr.append(tfidf_word_vec.transform(train["corpus"]))
    if tfidf_char_vec is not None:
        x_tr.append(tfidf_char_vec.transform(train["corpus"]))
    #x_tr.append(llm_preds_train[LABEL_COLS].values)
    X_train = sps.hstack(x_tr, format="csr")
    #print(X_train)
    #raise
    #std = StandardScaler(with_mean=False)
    #X_train = std.fit_transform(X_train)
    #    [
    #        #tfidf_word_vec.transform(train["corpus"]),
    #        #tfidf_char_vec.transform(train["corpus"]),
    #        #llm_preds_train[LABEL_COLS].values,
    #    ]
    #)

    logger.info(f"Transforming test {len(test)}..")
    x_tr = []
    if tfidf_word_vec is not None:
        x_tr.append(tfidf_word_vec.transform(test["corpus"]))
    if tfidf_char_vec is not None:
        x_tr.append(tfidf_char_vec.transform(test["corpus"]))
    #x_tr.append(llm_preds_test[LABEL_COLS].values)
    X_test = sps.hstack(x_tr, format="csr")
    #X_test = std.transform(X_test)
    #    [
    #        #tfidf_word_vec.transform(test["corpus"]),
    #        #tfidf_char_vec.transform(test["corpus"]),
    #        #llm_preds_test[LABEL_COLS].values,
    #    ]
    #)
    ### end feature extraction

    ### model training
    models = []
    y_cross_val_gt = []
    y_cross_val_hat = []
    y_cross_val_hat_raw = []
    y_cross_test_hat_raw = []
    all_scores = []

    # sgd params
    #loss = "hinge"  # 0.6458363670834235
    loss = "modified_huber"  # 0.7076827603433415
    loss = "log_loss"  # 0.7200638516434433
    max_iter = 1000
    alpha = 0.0001  # 0.0001
    penalty = "l1"
    #penalty = "l2"

    n_cpus = 8
    n_jobs = 2

    is_SuicideIntentDisclosed_seen = False

    for cols in BIN_COLS + CAT_COLS:
        bin_label = cols in BIN_COLS
        logger.info(f"Training for column {cols} (is binary: {bin_label})..")

        if bin_label:
            loss = "log_loss"
        else:
            loss = "modified_huber"

        if not bin_label or cols in ["DisclosedToOtherFamilyMember"]:
            final_estimator = SGDClassifier(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                max_iter=max_iter,
                random_state=args.seed,
                n_jobs=n_jobs,
            )
        else:
            final_estimator = lgb.LGBMClassifier(
                #objective="cross_entropy",  # "multiclass",
                #boosting_type="gbdt",

                #num_leaves=31,  # more  #  num_leaves <= 2 ** max_depth
                #min_child_samples=20,
                #max_depth=-1,

                #learning_rate=0.1,
                #n_estimators=100,
                #n_estimators=200,

                #max_bin=255,
                #max_bin=1023 if cols in LGB_MAX_BIN_COLS else 255,
                #max_bin=LGB_MAX_BIN_COLS.get(cols, 255),

                #reg_alpha=0.0,  # l1

                random_state=args.seed,
                n_jobs=n_jobs,
                #device_type="cuda",
            )

        final_estimator = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            max_iter=max_iter,
            random_state=args.seed,
            n_jobs=n_jobs,
        )

        logger.info(f"Making {args.n_folds} folds..")
        cv_fn = StratifiedKFold
        cv, y_train, tr_dev_inds = make_folds(
            train,
            cols=cols,
            cv_fn=cv_fn,
            n_folds=args.n_folds,
            seed=args.seed,
        )

        #bc = np.bincount(y_train)
        #class_prior = bc / bc.sum()

        estimators = []
        #estimators.append(
        #    (
        #        "mnaive_b",
        #        MultinomialNB(),  # use CountVectorizer
        #    ),
        #)
        #estimators.append(
        #    (
        #        "bnaive_b",
        #        BernoulliNB(  # use binary CountVectorizer
        #            class_prior=class_prior,
        #        ),
        #    ),
        #)
        #estimators.append(
        #    (
        #        "comnaive_b",
        #        ComplementNB(),
        #    ),
        #)
        #estimators.append(
        #    (
        #        "knn_51",
        #        KNeighborsClassifier(
        #            n_neighbors=51,
        #            #metric="cosine",
        #            #weights="distance",
        #            n_jobs=1,
        #        ),
        #    ),
        #)
        #estimators.append(
        #    (
        #        "sgd_l1_ll",
        #        SGDClassifier(
        #            loss=loss,
        #            penalty=penalty,
        #            alpha=alpha,
        #            max_iter=max_iter,
        #            random_state=args.seed,
        #            n_jobs=1,
        #        ),
        #    ),
        #)
        #estimators.append(
        #    (
        #        "sgd_l2_ll",
        #        SGDClassifier(
        #            loss=loss,
        #            penalty="l2",
        #            alpha=alpha,
        #            max_iter=max_iter,
        #            random_state=args.seed,
        #            n_jobs=1,
        #        ),
        #    ),
        #)
        #estimators.append(
        #    (
        #        "tree",
        #        DecisionTreeClassifier(
        #            random_state=args.seed,
        #        ),
        #    ),
        #)
        #estimators.append(
        #    (
        #        "rf_200",
        #        RandomForestClassifier(
        #            n_estimators=200,
        #            random_state=args.seed,
        #            n_jobs=1,
        #        ),
        #    ),
        #)
        #estimators.append(
        #    (
        #        "gb",
        #        GradientBoostingClassifier(
        #            random_state=args.seed,
        #        )
        #    ),
        #)
        #estimators.append(
        #    (
        #        "lr_l2_c1",
        #        LogisticRegression(
        #            random_state=args.seed,
        #            n_jobs=1,
        #        ),
        #    ),
        #)
        #estimators.append(
        #    (
        #        "lr_l2_c1_dual",
        #        LogisticRegression(
        #            dual=True,
        #            solver="liblinear",
        #            random_state=args.seed,
        #            n_jobs=1,
        #        ),
        #    ),
        #)
        #estimators.append(
        #    (
        #        "extra",
        #        ExtraTreesClassifier(
        #            random_state=args.seed,
        #            n_jobs=1,
        #        ),
        #    ),
        #)
        #estimators.append(
            #(
            #    "lgb",
            #    lgb.LGBMClassifier(
            #        #objective="cross_entropy",  # "multiclass",
            #        #boosting_type="gbdt",

            #        #num_leaves=31,  # more  #  num_leaves <= 2 ** max_depth
            #        #min_child_samples=20,
            #        #max_depth=-1,

            #        #learning_rate=0.1,
            #        #n_estimators=100,
            #        #n_estimators=200,

            #        #max_bin=255,
            #        #max_bin=1023 if cols in LGB_MAX_BIN_COLS else 255,
            #        #max_bin=LGB_MAX_BIN_COLS.get(cols, 255),

            #        #reg_alpha=0.0,  # l1

            #        random_state=args.seed,
            #        n_jobs=1,
            #        #device_type="cuda",
            #    ),
            #)
        #)
        #if cols != "WeaponType1":
            #estimators.append(
                #(
                #    "lrcv_l2_c1",
                #    LogisticRegressionCV(
                #        random_state=args.seed,
                #        n_jobs=1,
                #    ),
                #),
            #)
        #    estimators.append(
        #        (
        #            "lrcv_l2_c1_dual",
        #            LogisticRegressionCV(
        #                dual=True,
        #                solver="liblinear",
        #                #max_iter=1000,
        #                #scoring=make_scorer() bin_label else ,
        #                random_state=args.seed,
        #                n_jobs=1,
        #            ),
        #        )
        #    )
        #is_SuicideIntentDisclosed_seen = False

        if bin_label:
            method = "predict_proba"
            #method = "predict"
            threshold = 0.5
        else:
            method = "predict"
            threshold = None

        clf = final_estimator
        #clf = StackingClassifier(
        #    estimators=estimators,
        #    #final_estimator=final_estimator,
        #    #passthrough=True,
        #    cv=cv,
        #    #stack_method=method,
        #    n_jobs=n_cpus,
        #)
        #clf.fit(
        #    X_train,
        #    y_train,
        #    #np.random.uniform(0, 1, size=(5 * args.n_folds * (y_train.max() + 1), X_train.shape[1])),
        #    #np.repeat(np.arange(y_train.max() + 1), 5 * args.n_folds),
        #)
        #print(clf.predict(X_test))
        #clf.n_jobs = n_jobs
        #raise
        #_, clf = estimators[0]
        #print(_)
        #if len(estimators) > 1:
        #    clf = VotingClassifier(
        #        estimators=estimators,
        #        voting="soft",
        #        #weights=[0, 1],
        #        n_jobs=n_cpus,
        #    )

        #if cols != "WeaponType1":
        #    clf = CalibratedClassifierCV(
        #        estimator=clf,
        #        method="isotonic",
        #        cv=cv,
        #        n_jobs=n_jobs,
        #        ensemble=True,
        #    )

        _X_train = (
            #X_train,
            #X_train if not bin_label or len(y_cross_val_hat_raw) == 0 else sps.hstack(
            X_train if not is_SuicideIntentDisclosed_seen or not bin_label else sps.hstack(
                [
                    X_train,
                    #np.stack(y_cross_val_hat_raw, axis=1),
                    y_cross_val_hat_raw[LABEL_COLS.index("SuicideIntentDisclosed")][:, None],
                ],
                format="csr",
            )
        )
        logger.info(f"{args.n_folds} cross-validation with {method=}..")
        y_train_hat = cross_val_predict(
            clf,
            _X_train,
            y_train,
            method=method,
            cv=cv,
            n_jobs=args.n_folds,
            #n_jobs=1,
        )
        if len(y_train_hat.shape) == 2:
            y_train_hat = y_train_hat[:, 1]

        score = get_score(y_train, y_train_hat, threshold=threshold)
        logger.info(f"Score without normalization: {score:.4}")

        thresh = None
        if bin_label:
            logger.info("Threshold search..")
            thresh, score = threshold_search(y_train, y_train_hat)
            logger.info(f"Max score: {score} with thresh {thresh}")
            y_cross_val_hat_raw.append(y_train_hat)
            y_train_hat = (y_train_hat > thresh).astype("int32")
        else:
            y_cross_val_hat_raw.append(y_train_hat)

        y_cross_val_gt.append(y_train)
        y_cross_val_hat.append(y_train_hat)
        all_scores.append((cols, score.item()))

        logger.info(f"Training on full train {_X_train.shape}..")
        clf.n_jobs = n_cpus
        clf.fit(_X_train, y_train)
        models.append((cols, clf, thresh))

        _X_test = (
            #X_test,
            #X_test if not bin_label or len(y_cross_val_hat_raw) == 0 else sps.hstack(
            X_test if not is_SuicideIntentDisclosed_seen or not bin_label else sps.hstack(
                [
                    X_test,
                    #np.stack(y_cross_val_hat_raw, axis=1),
                    y_cross_test_hat_raw[LABEL_COLS.index("SuicideIntentDisclosed")][:, None],
                ],
                format="csr",
            )
        )
        ## prediction
        logger.info(f"Predicting on full test {_X_test.shape}..")
        if bin_label:
            if method == "predict_proba":
                y_test_hat = clf.predict_proba(_X_test)
                y_test_hat = y_test_hat[:, 1]
            else:
                y_test_hat = clf.predict(_X_test)

            y_cross_test_hat_raw.append(y_test_hat)
            y_test_hat = (y_test_hat > thresh).astype("int32")
        else:
            y_test_hat = clf.predict(_X_test) + 1

        #y_test_hat = []
        #for train_inds, _ in tr_dev_inds:
        #    clf.fit(X_train[train_inds], y_train[train_inds])
        #    logger.info(f"Predicting on full test {X_test.shape}..")
        #    if bin_label:
        #        _y_test_hat = clf.predict_proba(X_test)

        #    else:
        #        _y_test_hat = clf.predict_proba(X_test)

        #    y_test_hat.append(_y_test_hat)

        #y_test_hat = np.mean(y_test_hat, axis=0)
        #if bin_label:
        #    y_test_hat = (y_test_hat > thresh).astype("int32")
        #else:
        #    y_test_hat = np.argmax(y_test_hat, axis=1) + 1

        ### end model training

        logger.info(f"Making submission for {cols=}..")
        submission[cols] = y_test_hat
        is_SuicideIntentDisclosed_seen = is_SuicideIntentDisclosed_seen or (cols == "SuicideIntentDisclosed")

        joblib.dump(models, models_dir / "models.job")
        joblib.dump(
            {
                "labels": np.array(y_cross_val_gt).T,
                "preds": np.array(y_cross_val_hat).T,
                "preds_raw": np.array(y_cross_val_hat_raw).T,
                "scores": all_scores,
            },
            models_dir / "cv.job",
        )
        plot_scores(
            all_scores,
            out=models_dir / "scores.png",
            to_show=False,
        )

    save_path = Path(args.save_path)
    logger.info(f"Saving results to {save_path}..")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(save_path, index=True)

    y_cross_val_gt = np.array(y_cross_val_gt).T
    y_cross_val_hat = np.array(y_cross_val_hat).T
    y_cross_val_hat_raw = np.array(y_cross_val_hat_raw).T

    score = average_f1(y_cross_val_gt, y_cross_val_hat)
    logger.info(f"cross val score: {score}")
    y_random = np.concatenate(
        [
            np.random.randint(0, 2, size=(len(y_cross_val_hat), len(BIN_COLS))),
            np.random.randint(1, 6 + 1, size=(len(y_cross_val_hat), 1)),
            np.random.randint(1, 12 + 1, size=(len(y_cross_val_hat), 1)),
        ],
        axis=1,
    )
    #axiserror
    #y_random = np.concatenate(
    #    [
    #        np.full(
    #            len(y_cross_val_hat),
    #            fill_value=np.argmax(np.bincount(y_cross_val_hat[:, _y])),
    #        )
    #        for _y in range(len(y_cross_val_hat[0]))
    #    ],
    #    axis=1,
    #)
    score_random = average_f1(y_cross_val_gt, y_random)
    plot_scores(
        all_scores,
        score,
        score_random,
        out=models_dir / "scores.png",
    )


if __name__ == "__main__":
    main()
