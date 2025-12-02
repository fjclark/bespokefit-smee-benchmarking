"""Split the dataset into a validation set (for hyperparameter tuning) and a test set."""

import json
import loguru
import pandas as pd
from openff.toolkit import Molecule
import deepchem as dc
from rdkit.Chem import Draw
from pathlib import Path

JSON_FILE = "TNet500_minimal_dataset.json"
SMILES_NAME = "smiles.csv"
FRAC_TEST = 0.96  # We'll use the 4 % = 20 molecules for validation
SEED = 0
VALIDATION_OUTPUT_PATH = Path("validation_set")
TEST_OUTPUT_PATH = Path("test_set")

logger = loguru.logger


def load_smiles(json_file: str) -> pd.DataFrame:
    with open(json_file, "r") as f:
        data = json.load(f)

    smiles_list = [entry["mapped_smiles"] for entry in data["qm_torsions"]]
    id_list = [entry["id"] for entry in data["qm_torsions"]]
    torsion_idxs = [entry["dihedral_indices"] for entry in data["qm_torsions"]]

    return pd.DataFrame(
        {"id": id_list, "smiles": smiles_list, "torsion_idx": torsion_idxs}
    )


def split_dataset_maxmin(
    smiles_df: pd.DataFrame, frac_train: float, seed: int
) -> tuple[list[int], list[int]]:
    splitter = dc.splits.MaxMinSplitter()

    dc_dataset = dc.data.DiskDataset.from_numpy(
        X=smiles_df.id,
        ids=smiles_df.smiles,
    )

    # We will use the small "test" set as our validation set
    # and test on the large "train" set later
    test_dataset, valid_dataset = splitter.train_test_split(
        dc_dataset,
        frac_train=frac_train,
        seed=seed,
    )
    return test_dataset.X, valid_dataset.X


def save_torsion_img(ids: list[int], smiles_df: pd.DataFrame, filename: str) -> None:
    subset_df = smiles_df.iloc[ids]
    rdkit_mols = [
        Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True).to_rdkit()
        for smiles in subset_df["smiles"]
    ]

    img = Draw.MolsToGridImage(
        rdkit_mols,
        highlightAtomLists=subset_df["torsion_idx"].tolist(),
        legends=[f"ID: {id_}" for id_ in subset_df["id"]],
        molsPerRow=5,
        subImgSize=(300, 300),
    )

    # Save to png
    img.save(filename)


def save_smiles(ids: list[int], smiles_df: pd.DataFrame, filename: str) -> None:
    subset_df = smiles_df.iloc[ids]
    subset_df.to_csv(filename, index=False)


def save_sub_dataset(ids: list[int], json_file: str, output_file: str) -> None:
    with open(json_file, "r") as f:
        data = json.load(f)

    subset_data = {"qm_torsions": [data["qm_torsions"][i] for i in ids]}

    with open(output_file, "w") as f:
        json.dump(subset_data, f, indent=2)


def main():
    smiles_df = load_smiles(json_file=JSON_FILE)
    test_inds, valid_inds = split_dataset_maxmin(smiles_df, FRAC_TEST, SEED)

    VALIDATION_OUTPUT_PATH.mkdir(exist_ok=True)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    save_torsion_img(
        valid_inds, smiles_df, VALIDATION_OUTPUT_PATH / "validation_set_torsions.png"
    )
    save_smiles(valid_inds, smiles_df, VALIDATION_OUTPUT_PATH / SMILES_NAME)
    save_sub_dataset(
        valid_inds, JSON_FILE, VALIDATION_OUTPUT_PATH / "validation_set.json"
    )

    save_torsion_img(test_inds, smiles_df, TEST_OUTPUT_PATH / "test_set_torsions.png")
    save_smiles(test_inds, smiles_df, TEST_OUTPUT_PATH / SMILES_NAME)
    save_sub_dataset(test_inds, JSON_FILE, TEST_OUTPUT_PATH / "test_set.json")

    logger.info(f"Validation set size: {len(valid_inds)} molecules")
    logger.info(f"Test set size: {len(test_inds)} molecules")


if __name__ == "__main__":
    main()
