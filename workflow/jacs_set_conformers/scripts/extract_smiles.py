"""Get one molecule from each of the JACS set ligands."""

from openff.toolkit import Molecule
import pandas as pd

SDF_FILES = {
    "CDK2": "CDK2_ligand_ids.txt",
    "JNK1": "JNK1_ligand_ids.txt",
    "P38": "p38_ligand_ids.txt",
    "TYK2": "TYK2_ligand_ids.txt",
}


def main():
    """Get one molecule from each of the JACS set ligands."""

    smiles_by_id = {}

    for target, id_file in SDF_FILES.items():
        with open(id_file, "r") as f:
            first_line = f.readline().strip()
            smiles = first_line.split()[-2]
            lig_id = first_line.split()[-1]
        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        tot_charge = sum([atom.formal_charge for atom in molecule.atoms])
        assert tot_charge == 0
        lig_id = f"{target}_{lig_id}"
        smiles_by_id[lig_id] = smiles

    df = pd.DataFrame(smiles_by_id.items(), columns=["id", "smiles"])

    # Save the df as a CSV
    df.to_csv("smiles.csv")


if __name__ == "__main__":
    main()
