import os
import pickle
import subprocess
import sys
import time

import fire
import hypertune
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


# Set environments
REGION = "us-east1"
PROJECT_ID = "qwiklabs-asl-04-5e165f533cac"
ARTIFACT_STORE = f"gs://{PROJECT_ID}-beer-artifact-store"
DATA_ROOT = os.path.join(ARTIFACT_STORE, "data")
JOB_DIR_ROOT = os.path.join(ARTIFACT_STORE, "jobs")
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
JOB_NAME = f"JOB_VERTEX_{TIMESTAMP}"
JOB_DIR = f"{JOB_DIR_ROOT}/{JOB_NAME}"


def _create_rating_matrix(num_users: int, num_items: int) -> np.ndarray:
    r"""Create rating matrix with U x I dimension.
    U is the number of users and I is the number of items.

    Parameters
    ----------
    num_users : int
        The number of users
    num_items : int
        The number of items

    Returns
    -------
    np.ndarray
        Rating matrix with zeros
    """

    return np.zeros((num_users, num_items))


def _fill_rating_matrix(matrix: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    r"""Fill a rating matrix.

    Parameters
    ----------
    matrix : np.ndarray
        A rating matrix
    df : pd.DataFrame
        A data frame with the form of (user_id, item_id, rating)

    Returns
    -------
    np.ndarray
        Rating matrix with implicit feedback
    """
    for _, user, item, rating in df.itertuples():
        matrix[user, item] = 1

    return matrix


def _topk(arr: np.ndarray, k: int) -> np.ndarray:
    r"""Returns indices of k largest element of the given input matrix along
    the horizontal axis.

    Parameters
    ----------
    arr : np.ndarray
        An input array
    k : int

    Returns
    -------
    np.ndarray
        Indices of k largest elements
    """
    return np.argsort(arr)[:, -k:][:, ::-1]


def map_at_k(
    actual: np.ndarray, pred: np.ndarray, top_k: int, is_score=False
) -> float:
    r"""Mean average precision at k.

    Parameters
    ----------
    actual : np.ndarray
        A matrix with actual values.
    pred : np.ndarray
        A matrix with predictions.
    top_k : int

    Returns
    -------
    float
        Mean average precision at k
    """
    if is_score:
        if not _assert_same_dimension(actual, pred):
            raise AssertionError(
                "Two input matrices should have same dimension."
            )
    else:
        if len(actual) != len(pred):
            raise AssertionError("Two input matrices should have same length.")

    map_ = 0

    num_users = len(pred)
    if is_score:
        top_k_items = _topk(arr=pred, k=top_k)
    else:
        top_k_items = pred[:, :top_k]

    for i in range(num_users):
        actual_item = set(actual[i].nonzero()[0])
        pred_item = top_k_items[i]

        map_ += _ap_at_k(actual=actual_item, pred=pred_item, top_k=top_k)

    return map_ / num_users


def _ap_at_k(actual: np.array, pred: np.array, top_k: int) -> float:
    r"""Avearge precision at k

    Parameters
    ----------
    actual : np.array
        A list of item are to be predicted
    pred : np.array
        A list of predicted items
    top_k : int

    Returns
    -------
    float
        Average precision at k
    """

    if len(pred) > top_k:
        pred = pred[:top_k]

    p, cnt = 0, 0

    if not actual:
        return 0.0

    for idx, item in enumerate(pred):
        if item in actual:
            cnt += 1
            p += cnt / (idx + 1)

    return 0.0 if cnt == 0 else p / min(cnt, len(actual))


def _assert_same_dimension(actual: np.ndarray, pred: np.ndarray) -> bool:
    r"""Check the actual matrix and the prediction have same dimension.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    pred : np.ndarray
        Predicted values

    Returns
    -------
    bool
    """
    return actual.shape == pred.shape


def preprocess():
    # Load matrices and files
    print("Load files...")
    train = pd.read_parquet(os.path.join(DATA_ROOT, "train.parquet"))
    valid = pd.read_parquet(os.path.join(DATA_ROOT, "valid.parquet"))
    most_common_beers = pd.read_parquet(
        os.path.join(DATA_ROOT, "most_common_beers.parquet")
    )
    user_mapper = pd.read_parquet(
        os.path.join(DATA_ROOT, "user_mapper.parquet")
    )["idx"].to_dict()
    item_mapper = pd.read_parquet(
        os.path.join(DATA_ROOT, "item_mapper.parquet")
    )["idx"].to_dict()

    # Preprocess
    print("Preprocessing...")
    num_users = len(user_mapper)
    num_items = len(item_mapper)

    train_mat = _fill_rating_matrix(
        _create_rating_matrix(num_users, num_items), train
    )
    valid_mat = _fill_rating_matrix(
        _create_rating_matrix(num_users, num_items), valid
    )

    baseline_topk = most_common_beers[:10]["beer_name"].tolist()
    valid_users = np.where(valid_mat.sum(axis=1) > 0)[0]

    print("Make sparce matrices...")
    train_csr = csr_matrix(train_mat)

    return baseline_topk, train_csr, valid_mat, valid_users


def train(
    factors: int,
    regularization: float,
    iterations: int,
    is_tune: bool = True,
) -> None:
    r"""A helper function to train a matrix factorization model.

    Parameters
    ----------
    factors : int
        The dimension of latent spaces
    regularization : float
        The strength of L2 regularization
    iterations : int
        The number of iterations
    is_tune : bool, by default True
        If True is passed, a job for hyperparameter tuning via Vertex AI is triggered.
        When the job is not triggered, the trained model saved in the desired path.
    """
    print(f"""Current hyperparameter:
        - factors: {factors}
        - regularization: {regularization}
        - iterations: {iterations}"""
    )
    
    baseline_topk, train_csr, valid_mat, valid_users = preprocess()
    
    print("Model is training...")
    model = AlternatingLeastSquares(
        factors=int(factors),
        regularization=regularization,
        iterations=int(iterations),
        random_state=0,
    )
    model.fit(train_csr)

    if is_tune:
        print("Get 10 recommendations for each users...")
        pred = predict(
            user_ids=valid_users,
            train_csr=train_csr,
            model=model,
            popular_items=baseline_topk,
            top_k=10,
        )
        map_at_10 = map_at_k(actual=valid_mat[valid_users], pred=pred, top_k=10)
        print(f"MAP@10: {map_at_10:.6f}")

        # Log it with hypertune
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="map_at_10", metric_value=map_at_10
        )
    else:
        model_filename = "model.pkl"
        with open(model_filename, "wb") as model_file:
            pickle.dump(model, model_file)
        gcs_model_path = os.path.join(JOB_DIR, model_filename)
        subprocess.check_call(
            ["gsutil", "cp", model_filename, gcs_model_path], stderr=sys.stdout
        )


def predict(
    user_ids: np.ndarray,
    train_csr: csr_matrix,
    model: AlternatingLeastSquares,
    popular_items: list,
    top_k: int,
) -> np.ndarray:
    r"""Make recommendations for all users.
    Items which already reviewed by each user are removed from the recommendations.
    Note that the most popular items are provided to users that does not appear in training data.
    Those users have zero scores so they cannot get appropriate recommendations.

    Parameters
    ----------
    user_ids : np.ndarray
        The list of user ids
    train_mat : csr_matrix
        A sparse matrix which used for training
    model : AlternatingLeastSquares
        A trained model
    popular_items : list
        A list of the most common/popular items based on training data.
    top_k : int

    Returns
    -------
    np.ndarray
        Recommendations for each user

    """
    # Make recommendations based on the model
    rec = model.recommend(
        user_ids, train_csr[user_ids], N=top_k, filter_already_liked_items=True
    )

    # Substitutes for cold users with the most popular items
    rec_items = np.array(
        [
            popular_items if np.all(scores == 0) else items
            for items, scores in zip(rec[0], rec[1])
        ]
    )

    return rec_items


if __name__ == "__main__":
    fire.Fire(train)
