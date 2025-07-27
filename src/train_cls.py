import argparse
import os
import random

import neptune
import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import TensorboardLogger, NeptuneLogger
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Precision, Recall, Loss
from loguru import logger
from neptune.utils import stringify_unsupported
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset.dataset import LycaenidaeDatasetCls
from dataset.samplers import SAMPLERS
from dataset.transform import (
    get_cls_augmentation_transform,
    get_cls_pretrained_transform,
)
from model.classifier import MODELS
from model.classifier import get_pretrained_model
from utils import set_device, _FilterCallback


def train_model(
        model_name: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler: LRScheduler,
        epochs: int,
        device: str or torch.device,
        dirname: str,
        n_saved: int,
        log_interval: int,
        args: argparse.Namespace,
        neptune_logger: NeptuneLogger or None = None,
) -> None:
    """
    Train classification model
    :param model_name:      Model name for checkpoint names
    :param model:           Torch model
    :param train_loader:    Train data loader
    :param val_loader:      Validation data loader
    :param optimizer:       Optimizer
    :param criterion:       Classification criterion
    :param scheduler:       Learning rate scheduler
    :param epochs:          The number of epochs to train
    :param device:          Computing device
    :param dirname:         Directory to save weights
    :param n_saved:         The number of best weights to save
    :param log_interval:    Iteration interval for detailed logging
    :param neptune_logger   Neptune logger object from ignite library
    :param args:            All argparse arguments, for Neptune
    :return: None
    """
    model.to(device)
    precision = Precision(average=False)
    recall = Recall(average=False)
    metrics = {
        "loss": Loss(criterion),
        "f1_score": (precision * recall * 2 / (precision + recall + 1e-6)).mean(),
    }

    def train_step(engine: Engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def val_step(engine: Engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(x)
            return y_pred, y

    # Init trainer and evaluator objects
    trainer = Engine(train_step)
    train_evaluator = Engine(val_step)
    val_evaluator = Engine(val_step)

    # Attach metrics
    for name, metric in metrics.items():
        metric.attach(train_evaluator, name)
    for name, metric in metrics.items():
        metric.attach(val_evaluator, name)

    # Progress bars
    pbar = ProgressBar()
    pbar.attach(trainer)
    pbar.attach(train_evaluator, metric_names=["loss", "f1_score"])
    pbar.attach(val_evaluator, metric_names=["loss", "f1_score"])

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        pbar.log_message(
            f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        logger.debug(train_evaluator.state.metrics)
        metrics = train_evaluator.state.metrics
        pbar.log_message(
            f"Training Results - Epoch[{trainer.state.epoch}] Avg F1-score: {metrics['f1_score']:.2f} Avg loss: {metrics['loss']:.2f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        logger.debug(val_evaluator.state.metrics)
        metrics = val_evaluator.state.metrics
        loss, f_score = metrics["loss"], metrics["f1_score"]
        scheduler.step(loss)
        logger.debug(f"Last learning rate: {scheduler.get_last_lr()}")
        pbar.log_message(
            f"Validation Results - Epoch[{trainer.state.epoch}] Avg F1-score: {f_score:.2f} Avg loss: {loss:.2f}"
        )

    # Score function to return current value of any metric we defined above in val_metrics
    def score_function(engine):
        return 1 / (engine.state.metrics["loss"] + 1e-6)

    # Checkpoint to store n_saved best models wrt score function
    model_checkpoint = ModelCheckpoint(
        dirname=dirname,
        n_saved=n_saved,
        score_function=score_function,
        score_name="inv_loss",
        filename_prefix=model_name,
        global_step_transform=global_step_from_engine(
            trainer
        ),  # helps fetch the trainer's state
    )

    # Save the model after every epoch of val_evaluator is completed
    val_evaluator.add_event_handler(
        Events.COMPLETED, model_checkpoint, {"model": model}
    )

    for tag, evaluator in [
        ("training", train_evaluator),
        ("validation", val_evaluator),
    ]:
        if args.tboard:
            tb_logger = TensorboardLogger(log_dir="tb-logger")
            tb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED(every=log_interval),
                tag="training",
                output_transform=lambda loss: {"loss": loss},
            )
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names="all",
                global_step_transform=global_step_from_engine(trainer),
            )
        if neptune_logger:
            neptune_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED(every=log_interval),
                tag="training",
                output_transform=lambda loss: {"loss": loss},
            )
            neptune_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names="all",
                global_step_transform=global_step_from_engine(trainer),
            )
    logger.info("Starting training...")
    trainer.run(train_loader, max_epochs=epochs)
    logger.success("Finished training")


def main():
    """
    Fine-tune classifiers on a Lycaenidae dataset

    # Example:

    python ./src/train_cls.py --model=resnet50 --epochs=100 --view=bottom -a \
        --train_size=0.8 --val_size=0.1 --test_size=0.1 --sampler cbalanced \
        --metadata_path=./data/meta_all_groups_v3.csv \
        --batch_size=36 --dirname=./weights_cls/<> --neptune \
        --tags mix_of_common_and_target_groups
    """
    parser = argparse.ArgumentParser("Fine-tune classifiers on a Lycaenidae dataset")
    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS,
        help="Prepared torch models from the list",
    )
    parser.add_argument(
        "--epochs", required=True, type=int, help="Number of epochs to train"
    )
    parser.add_argument(
        "--metadata_path", required=True, help="Path to the metadata CSV table"
    )
    parser.add_argument(
        "--view",
        type=str,
        required=True,
        choices=["top", "bottom"],
        help="Image view to use",
    )
    parser.add_argument(
        "--train_size",
        required=True,
        type=float,
        help="The fraction of data for training",
    )
    parser.add_argument(
        "--val_size",
        required=True,
        type=float,
        help="The fraction of data for validation",
    )
    parser.add_argument(
        "--test_size",
        required=True,
        type=float,
        help="The fraction of data for testing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "-d",
        "--dirname",
        required=True,
        type=str,
        help="Checkpoint path to save weights",
    )

    parser.add_argument("--lr", type=str, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--patience", type=str, default=5, help="LR scheduler patience")
    parser.add_argument("--factor", type=float, default=0.9, help="LR scheduler factor")
    parser.add_argument(
        "--min_images_per_class", type=int, default=5, help="Min images per class"
    )
    parser.add_argument(
        "--sampler",
        required=False,
        choices=["default", *SAMPLERS.keys()],
        help="Batch sampler. Default means batch sampler is None for Dataloader class.",
    )
    parser.add_argument(
        "-a", "--augment", action="store_true", help="Augment the training data"
    )

    parser.add_argument(
        "--freeze", action="store_true", help="Freeze the model backbone"
    )
    parser.add_argument("--neptune", action="store_true", help="Use Neptune logger")
    parser.add_argument(
        "--neptune_project",
        type=str,
        default=os.environ.get("NEPTUNE_PROJECT_LYC"),
        help="Neptune project name",
    )
    parser.add_argument(
        "--neptune_token",
        type=str,
        default=os.environ.get("NEPTUNE_API_TOKEN"),
        help="Neptune token",
    )
    parser.add_argument("--tboard", action="store_true", help="Use Tensorboard logger")
    parser.add_argument("--gpu", default=0, type=int, help="Preferred GPU device id")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Iterations between detailed logging",
    )
    parser.add_argument(
        "--n_saved", type=int, default=5, help="Number of best weights to save"
    )
    parser.add_argument(
        "--tags", nargs="*", default=[], help="Additional tags for Neptune"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    args.tags.append(f"{args.sampler}_batch_sampler")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
        _FilterCallback()
    )
    device = set_device(args.gpu)

    pretrained_transforms = get_cls_pretrained_transform(model_name=args.model)
    train_set = LycaenidaeDatasetCls(
        metadata_path=args.metadata_path,
        view=args.view,
        subset="train",
        train_size=args.train_size,
        test_size=args.test_size,
        val_size=args.val_size,
        transform=get_cls_augmentation_transform() if args.augment else None,
        model_transform=pretrained_transforms,
        min_images_per_class=args.min_images_per_class,
        device=device,
        seed=args.seed,
    )
    val_set = LycaenidaeDatasetCls(
        metadata_path=args.metadata_path,
        view=args.view,
        subset="val",
        train_size=args.train_size,
        test_size=args.test_size,
        val_size=args.val_size,
        model_transform=pretrained_transforms,
        min_images_per_class=args.min_images_per_class,
        device=device,
        seed=args.seed,
    )
    model = get_pretrained_model(
        name=args.model, n_classes=train_set.n_classes, freeze=args.freeze
    )
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=args.patience, factor=args.factor
    )

    sampler_class = SAMPLERS.get(args.sampler)
    if args.sampler == "weighted":
        train_data_weights = train_set.get_data_weights()
        train_sampler = sampler_class(
            train_data_weights, num_samples=len(train_data_weights)
        )
        train_loader = DataLoader(
            train_set,
            shuffle=False,
            generator=torch.Generator(device=device),
            sampler=train_sampler,
            batch_size=args.batch_size,
        )
    elif args.sampler == "cbalanced":
        train_sampler = sampler_class(
            train_set.df_subset_metadata,
            batch_size=args.batch_size,
            sample_column="class",
        )
        train_loader = DataLoader(
            train_set,
            shuffle=False,
            generator=torch.Generator(device=device),
            batch_sampler=train_sampler,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator(device=device),
        )
    logger.info(f"Prepared {args.sampler} batch sampler")

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator(device=device),
    )

    if args.neptune:
        npt_logger = NeptuneLogger(
            api_token=args.neptune_token,
            project=args.neptune_project,
            name=f"lycaenidae_classification_{args.model}",
            tags=[
                "pytorch-ignite",
                args.view,
                args.model,
                args.metadata_path,
                *args.tags,
            ],
        )
        npt_logger.experiment["parameters"] = stringify_unsupported(vars(args))
        npt_logger.experiment["dataset/train"] = train_loader.dataset._logger_info
        npt_logger.experiment["dataset/val"] = val_loader.dataset._logger_info
    else:
        npt_logger = None

    weights_dir = f"{args.dirname}_view-{args.view}_{args.model}"
    train_model(
        model_name=args.model,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        dirname=weights_dir,
        n_saved=args.n_saved,
        log_interval=args.log_interval,
        args=args,
        neptune_logger=npt_logger,
    )


if __name__ == "__main__":
    main()
