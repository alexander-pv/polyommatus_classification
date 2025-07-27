import argparse
import os
import random
from io import StringIO

import matplotlib.pyplot as plt
import neptune
import numpy as np
import pandas as pd
import torch
import tqdm
from loguru import logger
from neptune.types import File
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from dataset.dataset import LycaenidaeDatasetCls
from dataset.transform import get_cls_pretrained_transform
from model.classifier import MODELS
from model.classifier import get_pretrained_model
from utils import set_device, get_best_model_weights


def get_confusion_matrix(
        true_values: np.ndarray,
        pred_values: np.ndarray,
        labels: list[str],
        save_path: str,
        title: str
):
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=true_values,
        y_pred=pred_values,
        display_labels=labels,
        cmap=plt.cm.Blues,
        ax=ax,
        colorbar=False,
    )
    ax.tick_params(axis="x", labelrotation=90)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


def evaluate_model(
        name: str,
        weights_dir: str,
        n_classes: int,
        dataset: LycaenidaeDatasetCls,
        device: str or torch.device,
        args: argparse.Namespace,
) -> None:
    """
    Evaluate a trained model
    :param name:           Model name
    :param weights_dir:    Path to weights directory
    :param n_classes:      The number of classes of a model
    :param dataset:        Dataset to use
    :param device:         Computing device
    :param args:           All argparse arguments
    :return: None
    """

    best_model_weights_path = get_best_model_weights(
        dirname=weights_dir,
        score_name="inv_loss"
    )
    model = get_pretrained_model(name=name, n_classes=n_classes, freeze=False)
    model.load_state_dict(torch.load(best_model_weights_path))
    model.to(device)
    model.eval()
    y_pred = []
    y_true = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, total=len(loader)):
            imgs, labels = batch
            imgs = imgs.to(device)
            proba = softmax(model(imgs))
            predicted_class = torch.argmax(proba, dim=1)
            y_pred.extend(predicted_class.tolist())
            y_true.extend(labels.tolist())

    target_names = [dataset.label_to_name(i) for i in range(dataset.n_classes)]
    report = classification_report(y_pred=y_pred, y_true=y_true, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report).T.reset_index()
    df_report.to_csv(os.path.join(args.output_dir, f"df_report_{args.model}_view_{args.view}.csv"))
    get_confusion_matrix(np.array(y_true), np.array(y_pred), target_names,
                         os.path.join(args.output_dir, f"cmatrix_{args.model}_view_{args.view}.png"),
                         f"Model: {args.model}. View: {args.view}")

    if args.neptune_id:
        csv_buffer = StringIO()
        df_report.to_csv(csv_buffer, index=False)
        run = neptune.init_run(
            project=args.neptune_project,
            api_token=args.neptune_token,
            with_id=args.neptune_id
        )
        run["dataset/test"] = dataset._logger_info
        run["eval/report"] = report
        run["eval/weights"] = best_model_weights_path
        run["eval/report-html"].upload(File.as_html(df_report))
        run["eval/report-csv"].upload(File.from_stream(csv_buffer, extension="csv"))
        logger.info(f"Neptune ID: {args.neptune_id}. the results were sent to server.")
    else:
        logger.warning("Neptune ID was not set to send results to the server.")

    logger.info(f"Evaluation report:\n{report}\nBest weights: {best_model_weights_path}")


def main():
    """
    # Example:

    # With Neptune logging
    python ./src/eval_cls.py --model=resnet50 --view=bottom \
           --train_size=0.8 --val_size=0.1 --test_size=0.1 \
           --metadata_path=./data/meta_all_groups_v3.csv \
           --dirname=./weights_cls/<> \
           --neptune_id=LYCAEN-<>

    # Without Neptune logging
    python ./src/eval_cls.py --model=resnet50 --view=bottom \
           --train_size=0.8 --val_size=0.1 --test_size=0.1 \
           --metadata_path=./data/meta_all_groups_v3.csv \
           --dirname=./weights_cls/<>

    :return: None
    """
    parser = argparse.ArgumentParser("Evaluate fine-tuned classifier")

    parser.add_argument("--model", required=True, choices=MODELS, help="Prepared torch models from the list")
    parser.add_argument("--metadata_path", required=True, help="Path to the metadata table")
    parser.add_argument("--view", type=str, required=True, choices=["top", "bottom"],
                        help="Which image view to use")
    parser.add_argument("--output_dir", type=str, default="./eval_output", help="Evaluation output dir")
    parser.add_argument("--train_size", required=True, type=float,
                        help="The fraction of data for training")
    parser.add_argument("--val_size", required=True, type=float,
                        help="The fraction of data for validation")
    parser.add_argument("--test_size", required=True, type=float,
                        help="The fraction of data for testing")
    parser.add_argument("--min_images_per_class", type=int, default=5,
                        help="Min images per class")
    parser.add_argument("--gpu", default=0, type=int,
                        help="Preferred GPU device id")
    parser.add_argument("-d", "--dirname", required=True, type=str,
                        help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--neptune_id", type=str, help="Neptune experiment ID")
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

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = set_device(args.gpu)
    os.makedirs(args.output_dir, exist_ok=True)
    pretrained_transforms = get_cls_pretrained_transform(model_name=args.model)
    train_set = LycaenidaeDatasetCls(
        metadata_path=args.metadata_path,
        view=args.view,
        subset="train",
        train_size=args.train_size,
        test_size=args.test_size,
        val_size=args.val_size,
        transform=None,
        model_transform=pretrained_transforms,
        min_images_per_class=args.min_images_per_class,
        device=device,
        seed=args.seed,
    )
    test_set = LycaenidaeDatasetCls(
        metadata_path=args.metadata_path,
        view=args.view,
        subset="test",
        train_size=args.train_size,
        test_size=args.test_size,
        val_size=args.val_size,
        transform=None,
        model_transform=pretrained_transforms,
        min_images_per_class=args.min_images_per_class,
        device=device,
        seed=args.seed
    )

    evaluate_model(
        name=args.model,
        weights_dir=args.dirname,
        n_classes=train_set.n_classes,
        dataset=test_set,
        device=device,
        args=args
    )


if __name__ == "__main__":
    main()
