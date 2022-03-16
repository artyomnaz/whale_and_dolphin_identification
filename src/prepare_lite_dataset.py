import os
import pandas as pd
import shutil
from tqdm import tqdm


if __name__ == "__main__":
    # path to train csv
    train_csv_path = 'dataset/modified_train.csv'

    # read data and fix the mistakes
    train_df = pd.read_csv(train_csv_path)
    train_df['species'].replace({
                                'bottlenose_dolpin' : 'bottlenose_dolphin',
                                'kiler_whale' : 'killer_whale',
                                'beluga' : 'beluga_whale',
                                'globis' : 'short_finned_pilot_whale',
                                'pilot_whale' : 'short_finned_pilot_whale'
                            }, inplace =True)

    # collect unique ids
    unique_individual_ids = list(set(train_df['individual_id']))

    # collect simple dataset with one item of each individual_id
    simple_dataset = []
    for unique_individual_id in tqdm(unique_individual_ids):
        simple_dataset.append(train_df[train_df['individual_id'] == unique_individual_id].iloc(0)[0])

    # copy dataset items to the new folder
    new_path = os.path.join('dataset', 'lite_train')
    os.makedirs(new_path, exist_ok=True)

    for item in tqdm(simple_dataset):
        image_path = os.path.join('dataset', 'train_images', item['image'])
        new_image_path = os.path.join(new_path, item['image'])
        shutil.copyfile(image_path, new_image_path)

    # create lite_train.csv
    new_train_df = pd.DataFrame(simple_dataset)
    new_train_df.to_csv(os.path.join('dataset', 'lite_train.csv'), index=False)
