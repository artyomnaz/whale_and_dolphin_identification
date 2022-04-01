import json
import os
from argparse import ArgumentParser

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from clearml import Task
from sklearn.neighbors import NearestNeighbors
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import (WhaleAndDolphinDataset, get_test_transform,
                     get_train_transform)
from model import HappyWhaleModel
from train import train_pipeline
from util import get_image_embeddings_and_labels, set_seed

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        required=True, help="path to yaml config")
    args = parser.parse_args()

    with open(args.config_path, "r") as yml_file:
        opt = yaml.safe_load(yml_file)

    device = torch.device('cuda:0')

    # set random seeds
    set_seed(opt['training']['random_seed'])

    # create save folder
    os.makedirs(opt['save_folder'], exist_ok=True)

    # define dataset, dataloader, dataframe
    print("train dataset...")
    train_dataset = WhaleAndDolphinDataset(
        dataset_path=opt['train_dataset_path'], df_path=opt['train_df_path'], image_size=opt['training']['image_size'],
        transform=get_train_transform(), is_train=True, balanced=True, balance_amount=opt['training']['balance_amount'])
    print("train dataloader...")
    train_dataloader = DataLoader(
        train_dataset, batch_size=opt['training']['batch_size'], num_workers=opt['training']['num_workers'], drop_last=True)
    train_df = train_dataset.df.copy()

    print("test dataset...")
    test_dataset = WhaleAndDolphinDataset(dataset_path=opt['test_dataset_path'], df_path=opt['test_df_path'],
                                          image_size=opt['training']['image_size'], transform=get_test_transform(), is_train=False, balanced=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=opt['training']['batch_size'], num_workers=opt['training']['num_workers'])
    test_df = test_dataset.df.copy()
    print("happy_model...")
    # define model, optimizer, criterion
    happy_model = HappyWhaleModel(
        numClasses=15587, noNeurons=250, embeddingSize=opt['training']['embedding_size'], model_name=opt['training']['model_name']).to(device)

    # ------------------------------------------CLEARML
    if opt['logs']['use_clearml']:
        # tensorboard
        writer = SummaryWriter()

        # init trains
        task = Task.init(
            project_name=opt['logs']['project_name'], task_name=opt['logs']['task_name'])

        # add generator configurations to clearml
        cfg_str = str(happy_model) + str('\n')
        Task.current_task().set_model_config(cfg_str)
        Task.current_task().connect(opt)
    # ------------------------------------------CLEARML

    optimizer = Adam(happy_model.parameters(), lr=float(opt['training']['lr']),
                     weight_decay=float(opt['training']['adam_weight_decay']), amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    print("happy_model_weights...")
    if opt['happy_model_weights'] is None or opt['happy_model_weights'] == '':
        # train model
        happy_model = train_pipeline(happy_model, optimizer, criterion, train_dataloader,
                                     device, opt['training']['n_epochs'], opt['save_folder'],
                                     save_name=opt['logs']['task_name'], writer=writer, vis_freq=opt['logs']['vis_freq'])
    else:
        # load pretrained model
        happy_model.load_state_dict(torch.load(opt['happy_model_weights']))
    print("train_embeddings...")
    # get embeddings
    if opt['train_embeddings_path'] is not None and opt['train_embeddings_path'] != '':
        train_image_embeddings = np.load(opt['train_embeddings_path'])
    else:
        train_image_embeddings, train_labels = get_image_embeddings_and_labels(
            happy_model, train_dataloader, device, is_train=True, save=True, save_folder=opt['save_folder'])

    print("train_labels...")
    if opt['train_labels_path'] is not None and opt['train_labels_path'] != '':
        train_labels = np.load(opt['train_labels_path'])
    print("test_embeddings...")
    if opt['test_embeddings_path'] is not None and opt['test_embeddings_path'] != '':
        test_image_embeddings = np.load(opt['test_embeddings_path'])
    else:
        test_image_embeddings = get_image_embeddings_and_labels(
            happy_model, test_dataloader, device, is_train=False, save=True, save_folder=opt['save_folder'])

    # embeddings shape
    dim = train_image_embeddings.shape[1]

    # build faiss index
    index = faiss.IndexFlatIP(dim)
    # index = faiss.IndexIVFFlat(
    #     quantiser, dim, opt['training']['faiss_clusters'], faiss.METRIC_INNER_PRODUCT)

    # vectors for faiss
    train_vectors = train_image_embeddings.astype('float32')

    # # train for index
    faiss.normalize_L2(train_vectors)
    # index.train(train_vectors)
    index.add(train_vectors)

    # get distances  and indices
    faiss.normalize_L2(test_image_embeddings.astype('float32'))
    D, I = index.search(test_image_embeddings,
                        k=opt['training']['faiss_clusters'])

    # collect submit csv
    answers = []
    d_max = np.max(D)
    d_min = np.min(D)

    print(d_min, d_max)
    for i in range(D.shape[0]):
        cur_d = D[i] #(D[i] - d_min) / (d_max - d_min)

        answer = ''

        if cur_d[-1] > 10: #opt['training']['threshold_distance']:
            answer += 'new_individual '
            for j in range(-2, -6, -1):
                answer += train_df['individual_id'][I[i][j]] + ' '
        else:
            for j in range(-1, -6, -1):
                answer += train_df['individual_id'][I[i][j]] + ' '
        answers.append(answer)

    test_df['predictions'] = answers
    test_df.to_csv(os.path.join(opt['save_folder'],
                   opt['logs']['task_name'] + '.csv'), index=False)
    # print("KNN...")

    # best_threshold_adjusted = 0.5
    # KNN = 100
    # neigh = NearestNeighbors(n_neighbors=KNN, metric='cosine')
    # neigh.fit(train_image_embeddings)

    # print('KNN...')

    # test_ids = []
    # test_nn_distances = []
    # test_nn_idxs = []

    # print('KNN...')
    # for i in tqdm(range(len(test_df))):
    #     row = test_df.iloc[i]
    #     embedding = test_image_embeddings[i]
    #     ids = row.image

    #     distances, idxs = neigh.kneighbors(
    #         embedding[None], KNN, return_distance=True)
    #     test_ids.append(ids)
    #     test_nn_idxs.append(idxs)
    #     test_nn_distances.append(distances)

    # print('KNN...')
    # test_nn_distances = np.concatenate(test_nn_distances)
    # test_nn_idxs = np.concatenate(test_nn_idxs)
    # # test_ids = np.concatenate(test_ids)

    # print("KNN...")

    # sample_submission = pd.read_csv('D:\kaggle\whale_and_dolphin_identification\whale_and_dolphin_identification\dataset\sample_submission.csv', index_col='image')
    # print(len(test_ids), len(sample_submission))

    # test_df = []
    # for i in tqdm(range(len(test_ids))):
    #     id_ = test_ids[i]
    #     targets = train_labels[test_nn_idxs[i]]
    #     distances = test_nn_distances[i]
    #     subset_preds = pd.DataFrame(np.stack([targets, distances], axis=1), columns=[
    #                                 'target', 'distances'])
    #     subset_preds['image'] = id_
    #     test_df.append(subset_preds)

    # target_encodings = {}
    # for i in range(len(train_df)):
    #     target_encodings[train_df.iloc[i].individual_id] = train_df.iloc[i].individual_key
    #     target_encodings[train_df.iloc[i].individual_key] = train_df.iloc[i].individual_id


    # sample_list = ['938b7e931166', '5bf17305f073',
    #                 '7593d2aee842', '7362d7a01d00', '956562ff2888']

    # test_df = pd.concat(test_df).reset_index(drop=True)
    # test_df['confidence'] = 1-test_df['distances']
    # test_df = test_df.groupby(
    #     ['image', 'target']).confidence.max().reset_index()
    # test_df = test_df.sort_values(
    #     'confidence', ascending=False).reset_index(drop=True)
    # test_df['target'] = test_df['target'].map(target_encodings)
    # test_df.to_csv('test_neighbors.csv')
    # test_df.image.value_counts().value_counts()

    # predictions = {}
    # for i, row in tqdm(test_df.iterrows()):
    #     if row.image in predictions:
    #         if len(predictions[row.image]) == 5:
    #             continue
    #         predictions[row.image].append(row.target)
    #     elif row.confidence > best_threshold_adjusted:
    #         predictions[row.image] = [row.target, 'new_individual']
    #     else:
    #         predictions[row.image] = ['new_individual', row.target]

    # for x in tqdm(predictions):
    #     if len(predictions[x]) < 5:
    #         remaining = [y for y in sample_list if y not in predictions]
    #         predictions[x] = predictions[x]+remaining
    #         predictions[x] = predictions[x][:5]
    #     predictions[x] = ' '.join(predictions[x])

    # predictions = pd.Series(predictions).reset_index()
    # predictions.columns = ['image', 'predictions']
    # predictions.to_csv(os.path.join(opt['save_folder'],
    #                                 opt['logs']['task_name'] + '.csv'), index=False)
