import logging
import os
import pathlib

import torch
from loguru import logger


class _FilterCallback(logging.Filterer):
    """
    Neptune logging fix: https://docs.neptune.ai/help/error_step_must_be_increasing/
    """

    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing:"
            )
        )


def set_device(device_id: int or None = None):
    gpus = {
        i: torch.cuda.get_device_name(torch.device(f"cuda:{i}"))
        for i in range(torch.cuda.device_count())
    }
    logger.debug(f"Available GPUs: {gpus}")
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        torch.cuda.set_device(f"cuda:{device_id}")
        device = f"cuda:{device_id}"
        logger.debug(f"Using {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
    logger.debug(f"Prepared device: {device}")
    return device


def get_best_model_weights(dirname: str, score_name: str) -> str:
    """
    Find the best model weights. For fine-tuned classifiers
    :param dirname:     The weights directory
    :param score_name   The name of score
    :return: Path to the best model weights
    """
    weights_files = [x for x in pathlib.Path(dirname).iterdir()]
    epochs = [
        int(file.name.split("model")[1].split(score_name)[0].replace("_", ""))
        for file in weights_files
    ]
    scores = [
        float(file.name.split("=")[-1].replace(".pt", "")) for file in weights_files
    ]
    best_score = max(scores)
    best_model_idx = scores.index(best_score)
    best_model_epoch = epochs[best_model_idx]
    weights_path = weights_files[best_model_idx]
    logger.info(f"Saved weights scores: {scores}")
    logger.info(
        f"Best model. Score: {best_score} Pos: {best_model_idx} Epoch: {best_model_epoch}"
    )
    return str(weights_path)
