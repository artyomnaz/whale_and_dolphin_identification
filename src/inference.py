from typing import Tuple

import faiss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder, normalize
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def get_embeddings(
    module: pl.LightningModule, dataloader: DataLoader, encoder: LabelEncoder, stage: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_image_names = []
    all_embeddings = []
    all_targets = []

    for batch in tqdm(dataloader, desc=f"Creating {stage} embeddings"):
        image_names = batch["image_name"]
        images = batch["image"].to(module.device)
        targets = batch["target"].to(module.device)

        embeddings = module(images)

        all_image_names.append(image_names)
        all_embeddings.append(embeddings.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_image_names = np.concatenate(all_image_names)
    all_embeddings = np.vstack(all_embeddings)
    all_targets = np.concatenate(all_targets)

    all_embeddings = normalize(all_embeddings, axis=1, norm="l2")
    all_targets = encoder.inverse_transform(all_targets)

    return all_image_names, all_embeddings, all_targets


def create_and_search_index(embedding_size: int, train_embeddings: np.ndarray, val_embeddings: np.ndarray, k: int):
    index = faiss.IndexFlatIP(embedding_size)
    index.add(train_embeddings)
    D, I = index.search(val_embeddings, k=k)  # noqa: E741

    return D, I


def create_val_targets_df(
    train_targets: np.ndarray, val_image_names: np.ndarray, val_targets: np.ndarray
) -> pd.DataFrame:

    allowed_targets = np.unique(train_targets)
    val_targets_df = pd.DataFrame(np.stack([val_image_names, val_targets], axis=1), columns=["image", "target"])
    val_targets_df.loc[~val_targets_df.target.isin(allowed_targets), "target"] = "new_individual"

    return val_targets_df


def create_distances_df(
    image_names: np.ndarray, targets: np.ndarray, D: np.ndarray, I: np.ndarray, stage: str  # noqa: E741
) -> pd.DataFrame:

    distances_df = []
    for i, image_name in tqdm(enumerate(image_names), desc=f"Creating {stage}_df"):
        target = targets[I[i]]
        distances = D[i]
        subset_preds = pd.DataFrame(np.stack([target, distances], axis=1), columns=["target", "distances"])
        subset_preds["image"] = image_name
        distances_df.append(subset_preds)

    distances_df = pd.concat(distances_df).reset_index(drop=True)
    distances_df = distances_df.groupby(["image", "target"]).distances.max().reset_index()
    distances_df = distances_df.sort_values("distances", ascending=False).reset_index(drop=True)

    return distances_df


def get_best_threshold(val_targets_df: pd.DataFrame, valid_df: pd.DataFrame) -> Tuple[float, float]:
    best_th = 0
    best_cv = 0
    for th in [0.1 * x for x in range(11)]:
        all_preds = get_predictions(valid_df, threshold=th)

        cv = 0
        for i, row in val_targets_df.iterrows():
            target = row.target
            preds = all_preds[row.image]
            val_targets_df.loc[i, th] = map_per_image(target, preds)

        cv = val_targets_df[th].mean()

        print(f"th={th} cv={cv}")

        if cv > best_cv:
            best_th = th
            best_cv = cv

    print(f"best_th={best_th}")
    print(f"best_cv={best_cv}")

    # Adjustment: Since Public lb has nearly 10% 'new_individual' (Be Careful for private LB)
    val_targets_df["is_new_individual"] = val_targets_df.target == "new_individual"
    val_scores = val_targets_df.groupby("is_new_individual").mean().T
    val_scores["adjusted_cv"] = val_scores[True] * 0.1 + val_scores[False] * 0.9
    best_th = val_scores["adjusted_cv"].idxmax()
    print(f"best_th_adjusted={best_th}")

    return best_th, best_cv


def get_predictions(df: pd.DataFrame, threshold: float = 0.2):
    sample_list = ["938b7e931166", "5bf17305f073", "7593d2aee842", "7362d7a01d00", "956562ff2888"]

    predictions = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Creating predictions for threshold={threshold}"):
        if row.image in predictions:
            if len(predictions[row.image]) == 5:
                continue
            predictions[row.image].append(row.target)
        elif row.distances > threshold:
            predictions[row.image] = [row.target, "new_individual"]
        else:
            predictions[row.image] = ["new_individual", row.target]

    for x in tqdm(predictions):
        if len(predictions[x]) < 5:
            remaining = [y for y in sample_list if y not in predictions]
            predictions[x] = predictions[x] + remaining
            predictions[x] = predictions[x][:5]

    return predictions


# TODO: add types
def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def create_predictions_df(test_df: pd.DataFrame, best_th: float) -> pd.DataFrame:
    predictions = get_predictions(test_df, best_th)

    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ["image", "predictions"]
    predictions["predictions"] = predictions["predictions"].apply(lambda x: " ".join(x))

    return predictions
