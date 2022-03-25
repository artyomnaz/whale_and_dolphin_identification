import os
from argparse import ArgumentParser

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


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--lr", default=1e-4, type=float,
                        help="learning rate for happy_model")
    parser.add_argument("--adam_weight_decay", default=1e-6,
                        type=float, help="Adam weight decay")
    parser.add_argument("--n_epochs", default=5,
                        type=int, help="epochs amount")
    parser.add_argument("--batch_size", default=3, type=int, help="batch size")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="num_workers for a dataloader")
    parser.add_argument("--random_seed", default=42,
                        type=int, help="random seed")
    parser.add_argument("--vis_freq", default=50, type=int,
                        help="clearml log frequency")
    parser.add_argument("--train_df_path", type=str, help="path to train df")
    parser.add_argument("--train_dataset_path", type=str,
                        help="path to train dataset")
    parser.add_argument("--test_df_path", type=str,
                        help="path to sample submission df")
    parser.add_argument("--test_dataset_path", type=str,
                        help="path to test dataset")
    parser.add_argument("--save_folder", type=str, help="save_folder")
    parser.add_argument("--embedding_size", default=128,
                        type=int, help="embedding size")
    parser.add_argument("--threshold_distance", default=5,
                        type=float, help="threshold for cluster distances")
    parser.add_argument("--faiss_clusters", default=50,
                        type=int, help="faiss clusters amount")
    parser.add_argument("--project_name", type=str,
                        default='whale_and_dolphin', help="name for the project")
    parser.add_argument("--task_name", type=str, help="name for the task")
    parser.add_argument("--happy_model_weights", type=str,
                        default=None, help="weights for happy model")
    parser.add_argument("--train_embeddings_path", type=str,
                        default=None, help="train embeddings path")
    parser.add_argument("--test_embeddings_path", type=str,
                        default=None, help="test embeddings path")
    parser.add_argument("--train_labels_path", type=str,
                        default=None, help="train labels path")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()

    device = torch.device('cuda:0')

    # set random seeds
    set_seed(opt.random_seed)

    # create save folder
    os.makedirs(opt.save_folder, exist_ok=True)

    # define dataset, dataloader, dataframe
    train_dataset = WhaleAndDolphinDataset(
        dataset_path=opt.train_dataset_path, df_path=opt.train_df_path, image_size=128, transform=get_train_transform(), is_train=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
    train_df = train_dataset.df.copy()

    test_dataset = WhaleAndDolphinDataset(dataset_path=opt.test_dataset_path, df_path=opt.test_df_path,
                                          image_size=128, transform=get_test_transform(), is_train=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
    test_df = test_dataset.df.copy()

    # define model, optimizer, criterion
    happy_model = HappyWhaleModel(
        numClasses=15587, noNeurons=250, embeddingSize=opt.embedding_size).to(device)

    # ------------------------------------------CLEARML
    # tensorboard
    writer = SummaryWriter()

    # init trains
    task = Task.init(project_name=opt.project_name, task_name=opt.task_name)

    # add generator configurations to clearml
    cfg_str = str(happy_model) + str('\n')
    Task.current_task().set_model_config(cfg_str)
    # ------------------------------------------CLEARML

    optimizer = Adam(happy_model.parameters(), lr=opt.lr,
                     weight_decay=opt.adam_weight_decay, amsgrad=False)
    criterion = nn.CrossEntropyLoss()

    if opt.happy_model_weights is None or opt.happy_model_weights == '':
        # train model
        happy_model = train_pipeline(happy_model, optimizer, criterion, train_dataloader,
                                     device, opt.n_epochs, opt.save_folder, save_name=opt.task_name, writer=writer, vis_freq=opt.vis_freq)
    else:
        # load pretrained model
        happy_model.load_state_dict(torch.load(opt.happy_model_weights))

    # get embeddings
    if opt.train_embeddings_path is not None and opt.train_embeddings_path != '':
        train_image_embeddings = np.load(opt.train_embeddings_path)
    else:
        train_image_embeddings, train_labels = get_image_embeddings_and_labels(
            happy_model, train_dataloader, device, is_train=True, save=True, save_folder=opt.save_folder)

    if opt.train_labels_path is not None and opt.train_labels_path != '':
        train_labels = np.load(opt.train_labels_path)

    if opt.test_embeddings_path is not None and opt.test_embeddings_path != '':
        test_image_embeddings = np.load(opt.test_embeddings_path)
    else:
        test_image_embeddings = get_image_embeddings_and_labels(
            happy_model, test_dataloader, device, is_train=False, save=True, save_folder=opt.save_folder)

    # embeddings shape
    dim = train_image_embeddings.shape[1]

    # build faiss index
    quantiser = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(
        quantiser, dim, opt.faiss_clusters, faiss.METRIC_INNER_PRODUCT)

    # vectors for faiss
    train_vectors = train_image_embeddings.astype('float32')

    # # train for index
    faiss.normalize_L2(train_vectors)
    index.train(train_vectors)
    index.add(train_vectors)

    # get distances  and indices
    faiss.normalize_L2(test_image_embeddings.astype('float32'))
    D, I = index.search(test_image_embeddings, k=opt.faiss_clusters)

    # collect submit csv
    answers = []
    for i in range(D.shape[0]):
        if D[i][0] > opt.threshold_distance:
            answer = 'new_individual '
            answer += ' '.join(train_df['individual_id'][I[i][:4]].to_list())
        else:
            answer = ' '.join(train_df['individual_id'][I[i][:5]].to_list())
        answers.append(answer)

    test_df['predictions'] = answers
    test_df.to_csv(os.path.join(opt.save_folder,
                   opt.task_name + '.csv'), index=False)
