import os

import faiss
import numpy as np
import torch
import torch.nn as nn
from clearml import Task
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import (WhaleAndDolphinDataset, get_test_transform,
                     get_train_transform)
from model import HappyWhaleModel
from train import train_pipeline
from util import get_image_embeddings_and_labels, set_seed
import yaml


if __name__ == "__main__":

    with open("options/config.yaml", "r") as yml_file:
        opt = yaml.safe_load(yml_file)

    device = torch.device('cuda:0')

    # set random seeds
    set_seed(opt['training']['random_seed'])

    # create save folder
    os.makedirs(opt['save_folder'], exist_ok=True)

    # define dataset, dataloader, dataframe
    train_dataset = WhaleAndDolphinDataset(
        dataset_path=opt['train_dataset_path'], df_path=opt['train_df_path'], image_size=128, 
        transform=get_train_transform(), is_train=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=opt['training']['batch_size'], num_workers=opt['training']['num_workers'])
    train_df = train_dataset.df.copy()

    test_dataset = WhaleAndDolphinDataset(dataset_path=opt['test_dataset_path'], df_path=opt['test_df_path'],
                                          image_size=128, transform=get_test_transform(), is_train=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=opt['training']['batch_size'], drop_last=True, num_workers=opt['training']['num_workers'])
    test_df = test_dataset.df.copy()

    # define model, optimizer, criterion
    happy_model = HappyWhaleModel(
        numClasses=15587, noNeurons=250, embeddingSize=opt['training']['embedding_size']).to(device)

    # ------------------------------------------CLEARML
    if opt['logs']['use_clearml']:
        # tensorboard
        writer = SummaryWriter()

        # init trains
        task = Task.init(project_name=opt['logs']['project_name'], task_name=opt['logs']['task_name'])

        # add generator configurations to clearml
        cfg_str = str(happy_model) + str('\n')
        Task.current_task().set_model_config(cfg_str)
    # ------------------------------------------CLEARML

    optimizer = Adam(happy_model.parameters(), lr=float(opt['training']['lr']),
                     weight_decay=float(opt['training']['adam_weight_decay']), amsgrad=False)
    criterion = nn.CrossEntropyLoss()

    if opt['happy_model_weights'] is None or opt['happy_model_weights'] == '':
        # train model
        happy_model = train_pipeline(happy_model, optimizer, criterion, train_dataloader,
                                     device, opt['training']['n_epochs'], opt['save_folder'], 
                                     save_name=opt['logs']['task_name'], writer=writer, vis_freq=opt['logs']['vis_freq'])
    else:
        # load pretrained model
        happy_model.load_state_dict(torch.load(opt['happy_model_weights']))

    # get embeddings
    if opt['train_embeddings_path'] is not None and opt['train_embeddings_path'] != '':
        train_image_embeddings = np.load(opt['train_embeddings_path'])
    else:
        train_image_embeddings, train_labels = get_image_embeddings_and_labels(
            happy_model, train_dataloader, device, is_train=True, save=True, save_folder=opt['save_folder'])

    if opt['train_labels_path'] is not None and opt['train_labels_path'] != '':
        train_labels = np.load(opt['train_labels_path'])

    if opt['test_embeddings_path'] is not None and opt['test_embeddings_path'] != '':
        test_image_embeddings = np.load(opt['test_embeddings_path'])
    else:
        test_image_embeddings = get_image_embeddings_and_labels(
            happy_model, test_dataloader, device, is_train=False, save=True, save_folder=opt['save_folder'])

    # embeddings shape
    dim = train_image_embeddings.shape[1]

    # build faiss index
    quantiser = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(
        quantiser, dim, opt['training']['faiss_clusters'], faiss.METRIC_INNER_PRODUCT)

    # vectors for faiss
    train_vectors = train_image_embeddings.astype('float32')

    # # train for index
    faiss.normalize_L2(train_vectors)
    index.train(train_vectors)
    index.add(train_vectors)

    # get distances  and indices
    faiss.normalize_L2(test_image_embeddings.astype('float32'))
    D, I = index.search(test_image_embeddings, k=opt['training']['faiss_clusters'])

    # collect submit csv
    answers = []
    for i in range(D.shape[0]):
        if D[i][0] > opt['training']['threshold_distance']:
            answer = 'new_individual '
            answer += ' '.join(train_df['individual_id'][I[i][:4]].to_list())
        else:
            answer = ' '.join(train_df['individual_id'][I[i][:5]].to_list())
        answers.append(answer)

    test_df['predictions'] = answers
    test_df.to_csv(os.path.join(opt['save_folder'],
                   opt['task_name'] + '.csv'), index=False)
