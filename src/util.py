import os
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def pad_and_resize_image(image_path, image_size=128):
    """Pad image and resize it to image_size.

    Args:
        image_path (str): path to image
        image_size (int, optional): Image size. Defaults to 128.

    Returns:
        _type_: PIL Image
    """
    # open image
    image = Image.open(image_path).convert('RGB')

    # image shape
    w, h = image.size

    # resize with keeping aspect ratio
    if w > h:
        k = h / w
        image = image.resize((image_size, int(image_size * k)))
    else:
        k = w / h
        image = image.resize((int(image_size * k), image_size))

    # new image shape
    w, h = image.size

    # padded image variable
    image_padded = np.zeros((image_size, image_size, 3))

    # pad image
    if w < h:
        start = (image_size - w) // 2
        image_padded[:, start:(start + w), ...] = image
    else:
        start = (image_size - h) // 2
        image_padded[start:(start + h), ...] = image

    return Image.fromarray(np.uint8(image_padded))


def set_seed(seed=1234):
    """Set seed for random

    Args:
        seed (int, optional): seed. Defaults to 1234.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_image_embeddings_and_labels(happy_model, dataloader, device, is_train=True, save=True, save_folder=None):
    """Function that returns image embeddings and labels which were obtained by happy model.

    Args:
        happy_model (HappyWhaleModel): model for embedding extraction
        dataloader (DataLoader): dataloader
        device (torch.device): device
        is_train (bool, optional): flag for train/test mode. Defaults to True.
        save (bool, optional): save embeddings and labels? Defaults to True.
        save_folder (str, optional): path to folder to save embeddings. Defaults to None.

    Returns:
        _type_: np.array [, np.array]
    """
    all_embeddings = []
    labels = []

    # save embeddings and labels
    happy_model.eval()
    with torch.no_grad():
        for x in tqdm(dataloader):
            x['image'] = x['image'].to(device)

            if is_train:
                x['label'] = x['label'].to(device)
                _, embedding = happy_model(x['image'], x['label'])
                labels.append(x['label'].cpu().numpy())
            else:
                embedding = happy_model(x['image'])

            embedding = embedding.detach().cpu().numpy()
            all_embeddings.append(embedding)

    all_embeddings = np.concatenate(all_embeddings)

    if is_train:
        labels = np.concatenate(labels)

        if save and save_folder is not None:
            np.save(os.path.join(save_folder,
                    'train_happy_model_embeddings.npy'), all_embeddings)
            np.save(os.path.join(save_folder,
                    'train_happy_model_labels.npy'), labels)

        return all_embeddings, labels

    if save and save_folder is not None:
        np.save(os.path.join(save_folder,
                'test_happy_model_embeddings.npy'), all_embeddings)

    return all_embeddings
