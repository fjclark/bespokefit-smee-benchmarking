from pathlib import Path
from typing import Optional, Union

import loguru
import qcelemental
from qcelemental.models.procedures import TorsionDriveResult
from yammbs.torsion.inputs import QCArchiveTorsionDataset, QCArchiveTorsionProfile

HARTREE2KCALMOL = qcelemental.constants.hartree2kcalmol
BOHR2ANGSTROMS = qcelemental.constants.bohr2angstroms

DATASET_NAME = "TNet500_minimal"

logger = loguru.logger


def get_qca_torsion_dataset(
    input_dir: Union[str, Path], selected_smiles: Optional[list[str]] = None
) -> QCArchiveTorsionDataset:

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"The directory {input_dir} does not exist.")

    qm_torsions = []

    # Order the files in the directory
    input_files = sorted(input_dir.glob("*.json"))

    for i, json_file in enumerate(input_files):
        if i == 150:
            break

        result = TorsionDriveResult.parse_file(json_file)

        # Check if the smiles string is in the provided list
        smiles = result.initial_molecule[0].extras[
            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        ]
        if selected_smiles is not None and smiles not in selected_smiles:
            continue

        assert len(result.keywords.dihedrals) == 1, "Exactly one dihedral is expected."
        dihedral_indices = result.keywords.dihedrals[0]

        coordinates, energies = {}, {}
        for angle, molecule in result.final_molecules.items():
            coordinates[angle] = molecule.geometry * BOHR2ANGSTROMS
            energies[angle] = result.final_energies[angle] * HARTREE2KCALMOL

        qm_torsions.append(
            QCArchiveTorsionProfile(
                id=i,
                mapped_smiles=smiles,
                dihedral_indices=dihedral_indices,
                coordinates=coordinates,
                energies=energies,
            )
        )

    return QCArchiveTorsionDataset(qm_torsions=qm_torsions)


def main() -> None:
    logger.info("Creating QCArchive Torsion Dataset...")
    dataset = get_qca_torsion_dataset(DATASET_NAME)
    with open(f"{DATASET_NAME}_dataset.json", "w") as f:
        f.write(dataset.json())
    logger.info(f"Dataset saved to {DATASET_NAME}_dataset.json")


if __name__ == "__main__":
    main()
