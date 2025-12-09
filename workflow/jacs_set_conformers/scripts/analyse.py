"""Compare the bespoke, sage and espaloma force fields to the ESPALOMA reference on the CREST ensemble conformers."""

from pathlib import Path
import subprocess
from openff.toolkit import Molecule, ForceField
from tqdm import tqdm
from openbabel import pybel
from openff.units import unit
import openmm.unit as omm_unit
from openff.interchange import Interchange
import numpy as np
import pandas as pd
import typer

from bespokefit_smee.sample import _get_ml_omm_system, _get_integrator
from openmm.app import Simulation
import openmm
import matplotlib.pyplot as plt
import pickle as pkl
from loguru import logger

plt.style.use("ggplot")


LIGANDS_CSV_PATH = Path("input/smiles.csv")
OVERALL_STATS_PATH = Path("crest_ensemble_overall_stats.pkl")
CREST_DIR = Path("crest")
OUTPUT_DIR = Path("analysis")
CREST_RMSD_THRESHOLDS = {
    "CDK2_17": 1.0,
    "JNK1_18629-1": 1.0,
    "P38_p38a_2n": 1.0,
    "TYK2_ejm_31": 0.125,
}


def openmm_to_openff_positions(
    omm_positions: list[openmm.unit.Quantity],
) -> list[unit.Quantity]:
    """
    Convert OpenMM positions to OpenFF positions.

    Parameters:
        omm_positions (list[openmm.unit.Quantity]): List of OpenMM positions.

    Returns:
        list[unit.Quantity]: List of OpenFF positions.
    """
    return [
        unit.Quantity(pos.value_in_unit(omm_unit.angstrom), "angstrom")
        for pos in omm_positions
    ]


def smiles_to_xyz(smiles: str, output_path: str = "struc.xyz") -> None:
    """
    Converts a SMILES string to an XYZ file.

    Parameters:
        smiles (str): The SMILES string of the molecule.
        output_path (str): The path to save the XYZ file.
    """
    # Create a molecule from the SMILES string
    molecule = Molecule.from_smiles(smiles)

    # Generate 3D coordinates
    molecule.generate_conformers(n_conformers=1)

    # Write the molecule to an XYZ file
    molecule.to_file(str(output_path), file_format="xyz")


def fix_xyz_with_obabel(xyz_path: str) -> None:
    """
    Fixes an XYZ file using Open Babel.

    Parameters:
        xyz_path (str): The path to the XYZ file to be fixed.
    """
    # Read the XYZ file using Pybel
    mol = next(pybel.readfile("xyz", xyz_path))

    # Write the molecule back to an XYZ file to fix formatting issues
    mol.write("xyz", xyz_path, overwrite=True)


def write_crest_toml(output_path: str = "crest.toml", rthr: float = 0.125) -> None:
    """
    Writes a basic CREST configuration file.

    Parameters:
        output_path (str): The path to save the CREST configuration file.
    """
    file_contents = [
        "input='struc.xyz'",
        "runtype='imtd-gc'",
        "#parallelization",
        "threads=30",
        "[[calculation.level]]",
        "method='gfnff'",
        "[cregen]",
        "ewin=6.0",
        "ethr=0.2",
        "bthr=99",
        f"rthr=5000",
    ]

    output_path.write_text("\n".join(file_contents))


def convert_crest_output_to_sdf(crest_dir: Path, output_sdf: Path) -> None:
    """
    Converts the CREST output to an SDF file.

    Parameters:
        crest_dir (Path): The directory containing the CREST output files.
        output_sdf (Path): The path to save the output SDF file.
    """
    # Convert the crest_conformers.xyz to SDF using pybel
    input_xyz = crest_dir / "crest_conformers.xyz"

    cmds = [
        "obabel",
        "-ixyz",
        str(input_xyz),
        "-osdf",
        "-O",
        str(output_sdf),
    ]

    subprocess.run(cmds, check=True)


def run_crest(name: str, smiles: str, output_dir: Path, rthr: float = 0.125) -> None:
    """
    Runs the CREST program for a given molecule.

    Parameters:
        name (str): The name of the molecule.
        smiles (str): The SMILES string of the molecule.
        output_dir (Path): The directory to save the CREST output.
    """

    # Make the crest directory
    crest_dir = output_dir
    crest_dir.mkdir(exist_ok=True)

    # Write the XYZ and toml files
    xyz_path = crest_dir / "struc.xyz"
    toml_path = crest_dir / "crest_input.toml"
    smiles_to_xyz(smiles, output_path=xyz_path)
    write_crest_toml(output_path=toml_path, rthr=rthr)
    fix_xyz_with_obabel(str(xyz_path))

    # Run crest from the crest directory
    stdout = crest_dir / "crest.log"
    with open(stdout, "w") as f:
        subprocess.run(
            ["crest", "crest_input.toml"],
            cwd=crest_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    # Convert the output to SDF
    output_sdf = crest_dir / f"{name}_conformers.sdf"
    convert_crest_output_to_sdf(crest_dir, output_sdf)


def run_crest_for_all(
    names_and_smiles: list[tuple[str, str]], output_dir: Path
) -> None:
    for name, smiles in tqdm(names_and_smiles, desc="Running CREST for all molecules"):
        run_crest(name, smiles, output_dir / name)


def get_energies_and_positions_mlp(
    mol: Molecule, mlp_name: str = "egret-1"
) -> tuple[list[unit.Quantity], list[unit.Quantity]]:
    """
    Get single-point energies for all conformers in an SDF file using a machine learning potential.

    Parameters:
        mol (Molecule): The molecule containing conformers.
        mlp_name (str): The name of the machine learning potential to use.

    Returns:
        list[unit.Quantity]: A list of energies for each conformer.
        list[unit.Quantity]: A list of minimised positions for each conformer.
    """
    ml_system = _get_ml_omm_system(mol, mlp_name=mlp_name)
    integrator = _get_integrator(300 * omm_unit.kelvin, 1.0 * omm_unit.femtoseconds)
    ml_simulation = Simulation(mol.to_topology().to_openmm(), ml_system, integrator)

    energies = []
    min_positions = []
    for positions in mol.conformers:
        ml_simulation.context.setPositions(positions.to_openmm())
        ml_simulation.minimizeEnergy(maxIterations=0)
        state = ml_simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilocalorie_per_mole)
        energies.append(energy * unit.kilocalorie / unit.mole)
        min_positions.append(state.getPositions())

    return energies, min_positions


def get_energies_ff(
    mol: Molecule, omm_system: openmm.openmm.System
) -> tuple[list[unit.Quantity], list[unit.Quantity]]:
    """
    Get single-point energies for all conformers in an SDF file using a force field.

    Parameters:
        mol (Molecule): The molecule containing conformers.
        omm_system (openmm.openmm.System): The OpenMM system containing the force field parameters.

    Returns:
        list[unit.Quantity]: A list of energies for each conformer.
        list[unit.Quantity]: A list of minimised positions for each conformer.
    """
    integrator = _get_integrator(300 * omm_unit.kelvin, 1.0 * omm_unit.femtoseconds)
    simulation = Simulation(mol.to_topology().to_openmm(), omm_system, integrator)

    energies = []
    min_positions = []
    for positions in mol.conformers:
        simulation.context.setPositions(positions.to_openmm())
        simulation.minimizeEnergy(maxIterations=0)
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilocalorie_per_mole)
        energies.append(energy * unit.kilocalorie / unit.mole)
        min_positions.append(state.getPositions())

    return energies, min_positions


def get_rmsd(
    molecule: Molecule,
    reference: list[unit.Quantity],
    target: list[unit.Quantity],
) -> float:
    """Compute the RMSD between two sets of coordinates."""
    from openeye import oechem

    molecule1 = Molecule(molecule)
    molecule2 = Molecule(molecule)

    reference_unitless = [coord.value_in_unit(omm_unit.angstrom) for coord in reference]
    target_unitless = [coord.value_in_unit(omm_unit.angstrom) for coord in target]

    for molecule in (molecule1, molecule2):
        if molecule.conformers is not None:
            molecule.conformers.clear()

    molecule1.add_conformer(unit.Quantity(reference_unitless, "angstrom"))

    molecule2.add_conformer(unit.Quantity(target_unitless, "angstrom"))

    # oechem appears to not support named arguments, but it's hard to tell
    # since the Python API is not documented
    return oechem.OERMSD(
        molecule1.to_openeye(),
        molecule2.to_openeye(),
        True,
        True,
        True,
    )


def main(bespoke_ff_name: str, bespoke_ff_path: str) -> None:
    """
    Evaluate the performance of the bespoke force field on low
    energy conformers generated by CREST.

    Parameters:
        bespoke_ff_name (str): Name of the bespoke force field to use.
        bespoke_ff_path (str): Path to the bespoke force field file.
    """

    input_molecules = pd.read_csv(LIGANDS_CSV_PATH)

    crest_output_dirs = {
        row["id"]: CREST_DIR / row["id"] for idx, row in input_molecules.iterrows()
    }

    for idx, row in input_molecules.iterrows():
        logger.info(f"Running CREST for {row['id']}")
        crest_dir = crest_output_dirs[row["id"]]
        if crest_dir.exists():
            logger.info(f"Skipping {row['id']} as CREST directory already exists.")
            continue
        crest_dir.mkdir(parents=True, exist_ok=True)
        rthr = CREST_RMSD_THRESHOLDS[row["id"]]
        run_crest(row["id"], row["smiles"], crest_dir, rthr=rthr)

    overall_results = {}

    for idx, row in tqdm(
        input_molecules.iterrows(),
        total=len(input_molecules),
        desc="Processing all molecules",
    ):

        # Get the base and output directories
        crest_dir = crest_output_dirs[row["id"]]
        analysis_dir = OUTPUT_DIR / row["id"]

        if analysis_dir.exists():
            logger.info(f"Skipping {row['id']} as analysis directory already exists.")
            continue

        analysis_dir.mkdir()

        # Get the Molecules and force fields
        mols = Molecule.from_file(crest_dir / f"{row['id']}_conformers.sdf")

        # Randomly sample 30 conformers if there are more than that
        if len(mols) > 30:
            mols = list(np.random.choice(mols, size=30, replace=False))

        ffs = {
            "bespoke": ForceField(
                bespoke_ff_path,
            ),
            "sage": ForceField("openff_unconstrained-2.3.0-rc2.offxml"),
            "bespokefit_1": ForceField(
                "../input_forcefields/sage-default-bespoke.offxml"
            ),
            "espaloma": ForceField(
                "../input_forcefields/espaloma.offxml", load_plugins=True
            ),
        }

        # Get the energies and positions from the ML potential
        mlp_energies = []
        mlp_positions = []
        ff_energies_and_positions = {
            ff_name: {"energies": [], "positions": []} for ff_name in ffs.keys()
        }

        ff_interchanges = {
            ff_name: Interchange.from_smirnoff(ffs[ff_name], mols[0].to_topology())
            for ff_name in ffs.keys()
        }
        ff_systems = {
            ff_name: interchange.to_openmm()
            for ff_name, interchange in ff_interchanges.items()
        }

        for mol in tqdm(mols, desc="Calculating energies for all conformers"):
            mol_mlp_energies, mol_mlp_positions = get_energies_and_positions_mlp(
                mol, mlp_name="egret-1"
            )
            mlp_energies.extend(mol_mlp_energies)
            mlp_positions.extend(mol_mlp_positions)
            off_positions = openmm_to_openff_positions(mol_mlp_positions)
            mol._conformers = off_positions

            for ff_name, ff_system in ff_systems.items():
                mol_ff_energies, mol_ff_positions = get_energies_ff(mol, ff_system)
                ff_energies_and_positions[ff_name]["energies"].extend(mol_ff_energies)
                ff_energies_and_positions[ff_name]["positions"].extend(mol_ff_positions)

        # Calculate the energy stats
        mlp_energies_array = np.array([energy.magnitude for energy in mlp_energies])
        ff_energies_arrays = {
            ff_name: np.array([energy.magnitude for energy in data["energies"]])
            for ff_name, data in ff_energies_and_positions.items()
        }

        # Subtract the mean energy from each set
        mlp_energies_array -= np.mean(mlp_energies_array)
        ff_energies_arrays = {
            ff_name: energies - np.mean(energies)
            for ff_name, energies in ff_energies_arrays.items()
        }

        # Calculate the RMSE and MAE
        energy_stats = {}
        with open(analysis_dir / "energy_stats.txt", "w") as f:
            for ff_name, energies in ff_energies_arrays.items():
                rmse = np.sqrt(np.mean((energies - mlp_energies_array) ** 2))
                mae = np.mean(np.abs(energies - mlp_energies_array))
                energy_stats[ff_name] = {"rmse": rmse, "mae": mae}
                logger.info(
                    f"{ff_name} RMSE: {rmse:.4f} kcal/mol, MAE: {mae:.4f} kcal/mol"
                )
                f.write(
                    f"{ff_name} RMSE: {rmse:.4f} kcal/mol, MAE: {mae:.4f} kcal/mol\n"
                )

        # Plot the energies and save
        fig, ax = plt.subplots(figsize=(7, 6))
        for ff_name, energies in ff_energies_arrays.items():
            ax.scatter(mlp_energies_array, energies, label=ff_name)
        min_val = np.array(
            np.min(mlp_energies_array),
            np.min([np.min(energies) for energies in ff_energies_arrays.values()]),
        ).min()
        max_val = np.array(
            np.max(mlp_energies_array),
            np.max([np.max(energies) for energies in ff_energies_arrays.values()]),
        ).min()
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="black",
            linestyle="--",
            label="y=x",
        )
        ax.set_xlabel("MLP Energies (kcal/mol)")
        ax.set_ylabel("FF Energies (kcal/mol)")
        ax.set_title("Comparison of MLP and FF Energies")
        ax.legend()
        fig.savefig(analysis_dir / "energy_comparison.png", dpi=300)

        # Calculate RMSDs to the MLP minimised structures
        ff_rmsds = {ff_name: [] for ff_name in ffs.keys()}
        for i in range(len(mlp_positions)):
            mlp_pos = mlp_positions[i]
            for ff_name, data in ff_energies_and_positions.items():
                ff_pos = data["positions"][i]
                rmsd = get_rmsd(mols[0], mlp_pos, ff_pos)
                ff_rmsds[ff_name].append(rmsd)

        # Print RMSD statistics
        rmsd_stats = {}
        with open(analysis_dir / "rmsd_stats.txt", "w") as f:
            for ff_name, rmsds in ff_rmsds.items():
                mean_rmsd = np.mean(rmsds)
                max_rmsd = np.max(rmsds)
                rms_rmsd = np.sqrt(np.mean(np.array(rmsds) ** 2))
                rmsd_stats[ff_name] = {
                    "rms_rmsd": rms_rmsd,
                    "mean_rmsd": mean_rmsd,
                    "max_rmsd": max_rmsd,
                }
                logger.info(
                    f"{ff_name} RMS RMSD: {rms_rmsd:.4f} Å, Mean RMSD: {mean_rmsd:.4f} Å, Max RMSD: {max_rmsd:.4f} Å"
                )
                f.write(
                    f"{ff_name} RMS RMSD: {rms_rmsd:.4f} Å, Mean RMSD: {mean_rmsd:.4f} Å, Max RMSD: {max_rmsd:.4f} Å\n"
                )

        # Plot the RMSDs for each conformer for each force field with a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Figure out offset and width based on the number of force fields
        n_ffs = len(ffs)
        bar_width = 0.2
        indices = np.arange(len(mlp_positions))

        # Plot in order of increasing MLP energy
        sorted_indices = np.argsort(mlp_energies_array)
        for i, (ff_name, rmsds) in enumerate(ff_rmsds.items()):
            sorted_rmsds = [rmsds[j] for j in sorted_indices]
            offset = (i - n_ffs / 2) * bar_width + bar_width / 2
            ax.bar(indices + offset, sorted_rmsds, width=bar_width, label=ff_name)

        ax.set_xlabel("Conformer Index\n(MLP Energy in kcal/mol)")
        ax.set_ylabel("RMSD to MLP Minimised Structure (Å)")
        ax.set_title("RMSD of FF Minimised Structures to MLP Minimised Structures")
        # Show all idx labels on x axis, and add the energy under each tick
        ax.set_xticks(indices)
        ax.set_xticklabels(
            [
                f"{i}\n{mlp_energies_array[sorted_indices[i]]:.2f}"
                for i in range(len(mlp_positions))
            ]
        )
        ax.legend()
        fig.savefig(analysis_dir / "rmsd_comparison.png", dpi=300)

        # Save the overall results
        overall_results[row["id"]] = {
            "energy_stats": energy_stats,
            "rmsd_stats": rmsd_stats,
        }

    # Write the overall results to a file
    with open(OVERALL_STATS_PATH, "wb") as f:
        pkl.dump(overall_results, f)


if __name__ == "__main__":
    typer.run(main)
