import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    train_csv_path = os.path.join('dataset', 'train.csv')
    save_path = os.path.join('dataset', 'modified_train.csv')

    # read train df
    train_df = pd.read_csv(train_csv_path)

    # fix the mistakes
    train_df['species'].replace({'bottlenose_dolpin': 'bottlenose_dolphin',
                                 'kiler_whale': 'killer_whale',
                                 'beluga': 'beluga_whale',
                                 'globis': 'short_finned_pilot_whale',
                                 'pilot_whale': 'short_finned_pilot_whale'}, inplace=True)

    # create class whale or dolphin
    train_df['class'] = train_df['species'].apply(lambda x: x.split("_")[-1])

    # map the individual id to a unique key (integer, not string)
    individual_mapping = train_df['individual_id'].value_counts(
    ).reset_index().drop(columns=['individual_id'])
    individual_mapping.columns = ['individual_id']
    individual_mapping['individual_key'] = np.arange(
        start=0, stop=len(individual_mapping), step=1)
    train_df = pd.merge(train_df, individual_mapping, on='individual_id')

    # add validation fold based on individual key group
    skf = StratifiedKFold(n_splits=5)
    skf_splits = skf.split(X=train_df.drop(
        columns="individual_key"), y=train_df["individual_key"])

    for fold, (train_index, valid_index) in enumerate(skf_splits):
        train_df.loc[valid_index, "kfold"] = np.int(fold)

    train_df["kfold"] = train_df["kfold"].astype(int)

    # save train_df to csv
    train_df.to_csv(save_path, index=False)
