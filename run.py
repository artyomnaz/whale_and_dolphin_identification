import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.dataset import HappyWhaleDataset
from src.inference import infer
from src.train import train
from src.utils import get_image_path


def run():
    with open('configs/config.yaml', "r") as yml_file:
        opt = yaml.safe_load(yml_file)

    IMAGE_SIZE = opt['train_params']['image_size']
    BATCH_SIZE = opt['train_params']['batch_size']
    CHECKPOINT_DIR = opt['paths']['CHECKPOINTS_DIR']
    MODEL_NAME = opt['train_params']['model_name']

    train_df = pd.read_csv(opt['paths']['TRAIN_CSV_PATH'])
    train_df["image_path"] = train_df["image"].apply(get_image_path, dir=opt['paths']['TRAIN_DIR'])

    encoder = LabelEncoder()
    train_df["individual_id"] = encoder.fit_transform(train_df["individual_id"])
    np.save(opt['paths']['ENCODER_CLASSES_PATH'], encoder.classes_)

    skf = StratifiedKFold(n_splits=opt['inference_params']['N_SPLITS'])
    for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.individual_id)):
        train_df.loc[val_, "kfold"] = fold
        
    train_df.to_csv(opt['paths']['TRAIN_CSV_ENCODED_FOLDED_PATH'], index=False)

    train(**opt['train_params'])
    infer(checkpoint_path=f"{CHECKPOINT_DIR}/{MODEL_NAME}_{IMAGE_SIZE}.ckpt", 
          image_size=IMAGE_SIZE, 
          batch_size=BATCH_SIZE)


if __name__ == '__main__':
    run()
