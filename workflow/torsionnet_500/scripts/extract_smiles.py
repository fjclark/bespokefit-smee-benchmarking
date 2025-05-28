import json

import loguru
import pandas as pd

JSON_FILE = "TNet500_minimal_dataset.json"
SMILES_FILE = "smiles.csv"

logger = loguru.logger


def json_to_smiles_csv(json_file: str, output_csv: str):
    """Convert a JSON file containing SMILES strings to a CSV file."""
    with open(json_file, "r") as f:
        data = json.load(f)

    smiles_list = [entry["mapped_smiles"] for entry in data["qm_torsions"]]
    id_list = [entry["id"] for entry in data["qm_torsions"]]

    # Create a DataFrame and save it to CSV
    df = pd.DataFrame({"id": id_list, "smiles": smiles_list})
    df.to_csv(output_csv, index=False)


def main():
    logger.info("Converting JSON to SMILES CSV...")
    json_to_smiles_csv(JSON_FILE, SMILES_FILE)
    logger.info(f"SMILES CSV saved to {SMILES_FILE}")


if __name__ == "__main__":
    main()
