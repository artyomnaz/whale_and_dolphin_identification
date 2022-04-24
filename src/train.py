import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.modules import LitDataModule, LitModule


def train(
    train_csv_encoded_folded: str,
    test_csv: str,
    checkpoints_dir: str,
    pretrained_path: str = None,
    val_fold: float = 0.0,
    image_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 2,
    model_name: str = "tf_efficientnet_b0",
    pretrained: bool = True,
    drop_rate: float = 0.0,
    embedding_size: int = 512,
    num_classes: int = 15587,
    arc_s: float = 30.0,
    arc_m: float = 0.5,
    arc_easy_margin: bool = False,
    arc_ls_eps: float = 0.0,
    optimizer: str = "adam",
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-6,
    accumulate_grad_batches: int = 1,
    auto_lr_find: bool = False,
    auto_scale_batch_size: bool = False,
    fast_dev_run: bool = False,
    gpus: int = 1,
    max_epochs: int = 10,
    precision: int = 16,
    stochastic_weight_avg: bool = True,
):
    pl.seed_everything(42)

    datamodule = LitDataModule(
        train_csv_encoded_folded=train_csv_encoded_folded,
        test_csv=test_csv,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    datamodule.setup()
    len_train_dl = len(datamodule.train_dataloader())

    module = LitModule(
        model_name=model_name,
        pretrained=pretrained,
        drop_rate=drop_rate,
        embedding_size=embedding_size,
        num_classes=num_classes,
        arc_s=arc_s,
        arc_m=arc_m,
        arc_easy_margin=arc_easy_margin,
        arc_ls_eps=arc_ls_eps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        len_train_dl=len_train_dl,
        epochs=max_epochs
    )

    model_checkpoint = ModelCheckpoint(
        checkpoints_dir,
        filename=f"{model_name}_{image_size}",
        monitor="val_loss",
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=auto_scale_batch_size,
        benchmark=True,
        callbacks=[model_checkpoint],
        deterministic=True,
        fast_dev_run=fast_dev_run,
        gpus=gpus,
        max_epochs=max_epochs,
        precision=precision,
        stochastic_weight_avg=stochastic_weight_avg,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        resume_from_checkpoint=pretrained_path
    )

    trainer.tune(module, datamodule=datamodule)

    trainer.fit(module, datamodule=datamodule)
