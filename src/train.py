import os

import numpy as np
import torch
from tqdm import tqdm


def train_pipeline(model, optimizer, criterion, dataloader, device, n_epochs, save_folder, save_name='embeddings_model_'):
    """Training pipeline

    Args:
        model (HappyWhaleModel): happy model
        optimizer (torch.optim): an optimizer
        criterion (torch.nn): loss function
        dataloader (DataLoader): dataloader
        device (torch.device): device
        n_epochs (int): epochs amount
        save_folder (str): path to folder to save checkpoints
        save_name (str, optional): name for checkpoints saving. Defaults to 'embeddings_model_'.

    Returns:
        _type_: _description_
    """
    BEST_SCORE = 9999

    # epochs cycle
    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        # batch cycle
        for x in tqdm(dataloader, desc='TRAIN'):
            x['image'] = x['image'].to(device)
            x['label'] = x['label'].to(device)

            optimizer.zero_grad()

            out, _ = model(x['image'], x['label'])
            loss = criterion(out, x['label'])
            loss.backward()

            optimizer.step()

            train_losses.append(loss.cpu().detach().numpy().tolist())

        mean_train_loss = np.mean(train_losses)
        print("Mean Train Loss:", mean_train_loss)

        # save the model
        if mean_train_loss < BEST_SCORE:
            save_path = os.path.join(save_folder, 'checkpoints')
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(
                save_path, save_name + f'_{round(mean_train_loss, 3)}.pth')
            torch.save(model.state_dict(), save_path)
            BEST_SCORE = mean_train_loss

    save_path = os.path.join(save_path, save_name + '_final'.pth)
    torch.save(model.state_dict(), save_path)

    return model