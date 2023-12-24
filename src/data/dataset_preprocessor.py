import random
from itertools import product
from pathlib import Path
from shutil import copy, rmtree

import click
from tqdm import tqdm


def split_dataset(
    dataset_path: Path | str,
    output_dataset_path: Path | str,
    ratio: tuple[float, float] = (0.7, 0.3),
) -> None:
    if output_dataset_path.exists():
        rmtree(output_dataset_path)
    output_dataset_path.mkdir(parents=True)
    for data_type, subset in product(["labels", "images"], ["train", "val"]):
        (output_dataset_path / subset / data_type).mkdir(parents=True)

    annotation_path = dataset_path / "annotation" / "annotation" / "YOLO-format"
    annotations = list(annotation_path.glob("*.txt"))
    random.shuffle(annotations)
    total_files = len(annotations)

    train_split = int(ratio[0] * total_files)
    train_files = annotations[:train_split]
    val_files = annotations[train_split:]
    # Копирование файлов в соответствующие подвыборки
    for file_list, subset in zip(
        [train_files, val_files], ["train", "val"]
    ):
        for file in tqdm(file_list, desc=f"Copy {subset} files..."):
            image_path = (
                dataset_path / "images" / "images" / file.with_suffix(".jpg").name
            )
            destination_annotation_path = (
                output_dataset_path / subset / "labels" / file.name
            )
            destination_image_path = (
                output_dataset_path / subset / "images" / file.with_suffix(".jpg").name
            )

            copy(file, destination_annotation_path)
            copy(image_path, destination_image_path)


def create_dataset_config(output_dataset_path: Path | str):
    with open(output_dataset_path / "dataset.yaml", "w") as file:
        for subset in [
            subset_dir
            for subset_dir in output_dataset_path.iterdir()
            if subset_dir.is_dir()
        ]:
            file.write(f"{subset.name}: {(subset / 'images').absolute()}\n")
        file.write("nc: 1\n")
        file.write("names: ['people']")


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=Path))
@click.option("--output-dataset-path", type=click.Path(path_type=Path))
def preprocess_dataset(dataset_path: Path | str, output_dataset_path: Path | str):
    split_dataset(dataset_path=dataset_path, output_dataset_path=output_dataset_path)
    create_dataset_config(output_dataset_path)


if __name__ == "__main__":
    preprocess_dataset()
