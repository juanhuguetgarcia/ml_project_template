# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from sklearn.datasets import load_boston
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]


def create_dataset(input_filepath):
    data = load_boston()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["price"] = data.target
    df.to_csv(Path(input_filepath).joinpath("house_prices.csv"), index=False)
    return df


def create_train_test_datasets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sample(frac=1)  # shuffle
    df["price"] = df["price"] * 1000
    test_size = int(df.shape[0] * 0.1)
    df_test = df[:test_size]
    df_train = df[test_size:]

    # Remove target to df_test as it won't have it in real world problem
    df_test = df_test.drop(columns="price")
    return df_test, df_train


def write_to_processed(df: pd.DataFrame, output_filepath: str, file_name) -> bool:
    output_filepath = Path(output_filepath)
    file = output_filepath.joinpath(file_name)
    df.to_csv(file, index=False)
    return file.is_file()


@click.command()
@click.argument(
    "input_filepath",
    type=click.Path(exists=True),
    default=PROJECT_DIR.joinpath("data/raw"),
)
@click.argument(
    "output_filepath", type=click.Path(), default=PROJECT_DIR.joinpath("data/processed")
)
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info("Creating dataset using load_boston from sklearn")
    data_df = create_dataset(input_filepath)
    logger.info(f"Raw Dataset saved to {input_filepath}")

    logger.info("Processing raw data. Separate train test.")
    df_test, df_train = create_train_test_datasets(data_df)

    logger.info(f"Persisting train data in {output_filepath}")
    write_to_processed(df_train, output_filepath, "modelling_data.csv")

    logger.info(f"Persisting test data in {output_filepath}")
    write_to_processed(df_test, output_filepath, "test_data.csv")
    logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
