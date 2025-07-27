import argparse
import os
import pathlib

import pandas as pd
import tqdm
from loguru import logger


def normalize_filename(filename: str) -> str:
    filename = filename.lower()
    for f in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        filename = filename.replace(f, "")
    return filename


def extract_view(filename: str) -> str:
    """
    Each scanned copy is represented by three pictures:
    * a - top of the wings
    * b - bottom of the wings
    * c - bottom of the wings with overlay
    Each specimen filmed on camera is represented by two pictures:
    * a - top of the wings
    * b - bottom of the wings

    Note:
        Code sets were tuned based on the real data.
        Duplicates represent latin and cyrillic.
    :param filename: image name
    :return: top / bottom
    """
    filename = filename.replace("seq_", "").replace(" ", "")
    code = filename.split("_")[-1]
    if code in {"a", "а", "a!", "a2"}:
        view = "top"
    elif code in {"b", "b1", "b2", "b!", "c", "с", "c2", "c3"}:
        view = "bottom"
    else:
        raise ValueError(f"Unknown view code: {code}")
    return view


def prepare_meta_table(args: argparse.Namespace) -> None:
    p = pathlib.Path(args.images)
    dir_classes = [x for x in p.iterdir() if x.is_dir()]
    files_meta = []

    for dir_class in tqdm.tqdm(dir_classes):
        for root, dirs, files in os.walk(dir_class):
            for filename in files:
                filepath = os.path.join(root, filename)
                logger.debug(f"Filepath: {filepath}")
                filename = normalize_filename(filename)
                img_id_parts = filename.replace("seq_", "").replace(" ", "").split("_")
                logger.info(f"filename: {filename} img_id_parts: {img_id_parts}")

                img_id = "-".join(img_id_parts[:-1])
                files_meta.append(
                    {
                        "class": dir_class.name.lower(),
                        "filepath": str(filepath),
                        "id": img_id,
                        "view": extract_view(filename),
                    }
                )

    df = pd.DataFrame(files_meta)
    df.to_csv(os.path.join(args.output, f"{args.filename}.csv"), index=False)
    df_views_counts = df["view"].value_counts()
    logger.info(f"Views:\n{df_views_counts}")
    logger.info(f"Total number of images: {df_views_counts.sum()}")

    df_by_classes_views_counts = (
        df.groupby(by=["class"])["view"].value_counts().reset_index()
    )
    df_by_classes_views_counts = df_by_classes_views_counts.pivot(
        index=["class"], columns="view", values="count"
    )
    df_by_classes_views_counts = df_by_classes_views_counts.sort_values(
        by="bottom", ascending=False
    )
    logger.info(f"Views by classes:\n{df_by_classes_views_counts}")


def main():
    parser = argparse.ArgumentParser("Prepares metadata CSV table")
    parser.add_argument("-i", "--images", type=str, help="Directory containing images")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./data",
        help="Directory where metadata will be saved",
    )
    parser.add_argument(
        "-f", "--filename", type=str, default="meta", help="Metadata filename"
    )
    args = parser.parse_args()
    logger.debug(f"args: {args}")
    prepare_meta_table(args)


if __name__ == "__main__":
    main()
