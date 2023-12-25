from pathlib import Path
from random import shuffle
from shutil import copy, rmtree

import click
from tqdm import tqdm


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=Path))
@click.option("--output-dataset-path", type=click.Path(path_type=Path))
def split_dataset_on_subsets(
    dataset_path: Path | str, output_dataset_path: Path | str, ratio: float = 0.7
) -> None:
    if output_dataset_path.exists():
        rmtree(output_dataset_path)
    output_dataset_path.mkdir(parents=True)

    for subset in ["train", "val"]:
        (output_dataset_path / subset).mkdir(parents=True)

    dataset_files = list(dataset_path.glob("*.jpg"))
    shuffle(dataset_files)
    train_split = int(len(dataset_files) * ratio)
    train_filenames = dataset_files[:train_split]
    val_filenames = dataset_files[train_split:]

    for file_list, subset in zip([train_filenames, val_filenames], ["train", "val"]):
        for file in tqdm(file_list, desc=f"Copy {subset} files..."):
            image_path = dataset_path / file.name
            destination_image_path = output_dataset_path / subset / file.name
            copy(image_path, destination_image_path)


if __name__ == "__main__":
    split_dataset_on_subsets()
