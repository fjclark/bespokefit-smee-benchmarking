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
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit.utils.exceptions import RadicalsNotSupportedError

from bespokefit_smee.sample import _get_ml_omm_system, _get_integrator
from openmm.app import Simulation
import openmm
import matplotlib.pyplot as plt
import pickle as pkl
from loguru import logger

# Set the random seed for reproducibility
np.random.seed(42)

plt.style.use("ggplot")


LIGANDS_CSV_PATH = Path("input/smiles.csv")
CREST_DIR = Path("crest")
OUTPUT_DIR = Path("analysis")
CREST_RMSD_THRESHOLDS = {
    "CDK2_17": 1.0,
    "JNK1_18629-1": 1.0,
    "P38_p38a_2n": 1.0,
    "TYK2_ejm_31": 0.125,
}
MAX_CONFORMERS_PER_MOL = 50


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
        # "ethr=0.2",
        # "bthr=99",
        f"rthr={rthr}",
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
    mol: Molecule, mlp_name: str = "egret-1", minimise: bool = True
) -> tuple[list[unit.Quantity], list[unit.Quantity]]:
    """
    Get single-point energies for all conformers in an SDF file using a machine learning potential.

    Parameters:
        mol (Molecule): The molecule containing conformers.
        mlp_name (str): The name of the machine learning potential to use.
        minimise (bool): Whether to minimise the conformers before calculating energies.

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
        if minimise:
            ml_simulation.minimizeEnergy(maxIterations=0)
        state = ml_simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilocalorie_per_mole)
        energies.append(energy * unit.kilocalorie / unit.mole)
        min_positions.append(state.getPositions())

    return energies, min_positions


def get_energies_ff(
    mol: Molecule, ff: ForceField, minimise: bool = True
) -> tuple[list[unit.Quantity], list[unit.Quantity]]:
    """
    Get single-point energies for all conformers in an SDF file using a force field.

    Parameters:
        mol (Molecule): The molecule containing conformers.
        ff (ForceField): The force field to use.
        minimise (bool): Whether to minimise the conformers before calculating energies.

    Returns:
        list[unit.Quantity]: A list of energies for each conformer.
        list[unit.Quantity]: A list of minimised positions for each conformer.
    """
    omm_system = Interchange.from_smirnoff(ff, mol.to_topology()).to_openmm()

    integrator = _get_integrator(300 * omm_unit.kelvin, 1.0 * omm_unit.femtoseconds)
    simulation = Simulation(mol.to_topology().to_openmm(), omm_system, integrator)

    energies = []
    min_positions = []
    for positions in mol.conformers:
        simulation.context.setPositions(positions.to_openmm())
        if minimise:
            simulation.minimizeEnergy(maxIterations=0)
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilocalorie_per_mole)
        energies.append(energy * unit.kilocalorie / unit.mole)
        min_positions.append(state.getPositions())

    return energies, min_positions


def save_conformers_to_sdf(
    mol: Molecule,
    positions: list,
    energies: list[unit.Quantity],
    output_path: Path,
    name_prefix: str = "conformer",
) -> None:
    """
    Save conformers to an SDF file with titles set as relative energies.

    Parameters:
        mol: The base molecule.
        positions: List of positions for each conformer.
        energies: List of energies for each conformer.
        output_path: Path to save the SDF file.
        name_prefix: Prefix for conformer names.
    """
    # Calculate relative energies (relative to first conformer)
    energies_array = np.array([e.magnitude for e in energies])
    relative_energies = energies_array - energies_array[0]

    # Create a molecule with all conformers
    output_mols = []
    for i, (pos, rel_energy) in enumerate(zip(positions, relative_energies)):
        mol_copy = Molecule(mol)
        mol_copy.conformers.clear()

        # Convert positions if needed
        if isinstance(pos[0], openmm.unit.Quantity):
            pos_unitless = [coord.value_in_unit(omm_unit.angstrom) for coord in pos]
            mol_copy.add_conformer(unit.Quantity(pos_unitless, "angstrom"))
        else:
            mol_copy.add_conformer(pos)

        # Set the name/title as the relative energy
        mol_copy.name = f"{name_prefix}_{i:03d}_rel_energy_{rel_energy:.4f}_kcal_mol"
        output_mols.append(mol_copy.to_rdkit())

    # Write all molecules to SDF
    with Chem.SDWriter(str(output_path)) as sdf_writer:
        for mol in output_mols:
            sdf_writer.write(mol)


def read_conformers_from_sdf(
    sdf_path: Path,
) -> tuple[list, list[unit.Quantity]]:
    """
    Read conformers from an SDF file and extract energies from titles.

    Parameters:
        sdf_path: Path to the SDF file.

    Returns:
        Tuple of (energies, positions) where positions are OpenMM quantities
        and energies are relative to the first conformer.
    """
    mols = Molecule.from_file(str(sdf_path))
    if not isinstance(mols, list):
        mols = [mols]

    positions = []
    relative_energies = []

    for mol in mols:
        # Extract relative energy from the molecule name
        # Format: prefix_XXX_rel_energy_Y.YYYY_kcal_mol
        if mol.name and "rel_energy" in mol.name:
            parts = mol.name.split("_rel_energy_")
            if len(parts) > 1:
                energy_str = parts[1].replace("_kcal_mol", "")
                rel_energy = float(energy_str)
                relative_energies.append(rel_energy)
            else:
                relative_energies.append(0.0)
        else:
            relative_energies.append(0.0)

        # Get positions from the conformer
        if len(mol.conformers) > 0:
            positions.append(mol.conformers[0].to_openmm())

    # Convert relative energies to absolute energies
    # (add back the first conformer's energy, which is 0)
    energies = [
        (rel_e + relative_energies[0]) * unit.kilocalorie / unit.mole
        for rel_e in relative_energies
    ]

    return energies, positions


# def get_rmsd(
#     molecule: Molecule,
#     reference: list[unit.Quantity],
#     target: list[unit.Quantity],
# ) -> float:
#     """Compute the RMSD between two sets of coordinates using RDKit with alignment."""
#     # Create two seperate mols for alignment
#     ref_mol = Molecule(molecule)
#     tgt_mol = Molecule(molecule)

#     # Add the reference conformer
#     ref_mol.conformers.clear()
#     ref_mol.add_conformer(openmm_to_openff_positions([reference])[0])

#     # Add the target conformer
#     tgt_mol.conformers.clear()
#     tgt_mol.add_conformer(openmm_to_openff_positions([target])[0])

#     # Convert to RDKit molecules
#     ref_rdkit = ref_mol.to_rdkit()
#     tgt_rdkit = tgt_mol.to_rdkit()

#     # Align target to reference and calculate RMSD
#     rmsd = Chem.rdMolAlign.AlignMol(tgt_rdkit, ref_rdkit)

#     return rmsd


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
        rthr = CREST_RMSD_THRESHOLDS.get(row["id"], 0.125)
        run_crest(row["id"], row["smiles"], crest_dir, rthr=rthr)

    overall_results = {}

    for idx, row in tqdm(
        input_molecules.iterrows(),
        total=len(input_molecules),
        desc="Processing all molecules",
    ):

        # Get the base and output directories
        crest_dir = crest_output_dirs[row["id"]]

        # Get the Molecules and force fields
        try:
            mols = Molecule.from_file(crest_dir / f"{row['id']}_conformers.sdf")
            if not isinstance(mols, list):
                logger.warning(
                    f"Skipping {row['id']} as only a single conformer found in SDF."
                )
                continue

        except RadicalsNotSupportedError:
            logger.warning(
                f"Skipping {row['id']} due to radicals not supported by OpenFF Toolkit."
            )
            continue

        # Combine all conformers into a single Molecule
        mol = Molecule(mols[0])
        mol.conformers.clear()
        for m in mols:
            for conf in m.conformers:
                mol.add_conformer(conf)

        # Randomly sample MAX_CONFORMERS_PER_MOL conformers if there are more than that
        if len(mol.conformers) > MAX_CONFORMERS_PER_MOL:

            idxs = np.random.choice(
                len(mol.conformers), size=MAX_CONFORMERS_PER_MOL, replace=False
            )
            # Sort to ensure we retain order of increasing energy
            idxs.sort()
            mol._conformers = [mol.conformers[i] for i in idxs]

        logger.info(f"Loaded {len(mol.conformers)} conformers for {row['id']}")

        ffs = {
            "bespoke": ForceField(
                bespoke_ff_path,
            ),
            # "bespoke1": ForceField("output/egret1_opt_reg/combined_forcefield.offxml"),
            # "bespoke2": ForceField(
            #     "output/egret1_opt_reg_repeat_2/combined_forcefield.offxml"
            # ),
            # "bespoke3": ForceField(
            #     "output/egret1_opt_reg_repeat_3/combined_forcefield.offxml"
            # ),
            "sage": ForceField("openff_unconstrained-2.3.0-rc2.offxml"),
            "bespokefit_1": ForceField(
                "../input_forcefields/sage-default-bespoke.offxml"
            ),
            "espaloma": ForceField(
                "../input_forcefields/espaloma.offxml", load_plugins=True
            ),
        }

        # Check if MLP SDF already exists
        mlp_sdf_path = crest_dir / f"{row['id']}_mlp_minimised.sdf"

        if mlp_sdf_path.exists():
            logger.info(f"MLP minimised SDF already exists for {row['id']}")
            mlp_energies, mlp_positions = read_conformers_from_sdf(mlp_sdf_path)
        else:
            logger.info(f"Calculating MLP minimised structures for {row['id']}")
            mlp_energies, mlp_positions = get_energies_and_positions_mlp(
                mol, mlp_name="egret-1"
            )

            # Save MLP-minimised structures to CREST directory
            save_conformers_to_sdf(
                mols[0], mlp_positions, mlp_energies, mlp_sdf_path, name_prefix="mlp"
            )
            logger.info(f"Saved MLP-minimised structures to {mlp_sdf_path}")

        # Update the molecule with the minimised MLP positions
        mol._conformers = openmm_to_openff_positions(mlp_positions)

        analysis_dir = OUTPUT_DIR / row["id"]

        if analysis_dir.exists():
            logger.info(f"Skipping {row['id']} as analysis directory already exists.")
            continue

        analysis_dir.mkdir()

        ff_energies_and_positions = {}

        for ff_name in ffs.keys():
            ff_sdf_path = analysis_dir / f"{row['id']}_{ff_name}_minimised.sdf"
            if ff_sdf_path.exists():
                logger.info(f"{ff_name} minimised SDF already exists for {row['id']}")
                ff_energies, ff_positions = read_conformers_from_sdf(ff_sdf_path)
            else:
                logger.info(
                    f"Calculating {ff_name} minimised structures for {row['id']}"
                )

                ff_energies, ff_positions = get_energies_ff(
                    mol, ffs[ff_name], minimise=True
                )

                # Save FF-minimised structures to analysis directory
                save_conformers_to_sdf(
                    mols[0],
                    ff_positions,
                    ff_energies,
                    ff_sdf_path,
                    name_prefix=ff_name,
                )
                logger.info(f"Saved {ff_name} minimised structures to {ff_sdf_path}")

            ff_energies_and_positions[ff_name] = {
                "energies": ff_energies,
                "positions": ff_positions,
            }

        # # Also add on the aimnet2 energies and forces
        # logger.info("Calculating energies and positions with AIMNet2")
        # aimnet2_energies, aimnet2_positions = get_energies_and_positions_mlp(
        #     mol, "aimnet2_wb97m_d3_ens", minimise=False
        # )
        # ff_energies_and_positions["aimnet2"] = {
        #     "energies": aimnet2_energies,
        #     "positions": aimnet2_positions,
        # }

        # if mlp_sdf_path.exists() and all_ff_sdfs_exist:
        #     # Read back energies and positions from existing SDF files
        #     logger.info(f"Reading existing minimised structures for {row['id']}")
        #     mlp_positions, mlp_energies = read_conformers_from_sdf(mlp_sdf_path)

        #     ff_energies_and_positions = {}
        #     for ff_name in ffs.keys():
        #         ff_sdf_path = analysis_dir / f"{row['id']}_{ff_name}_minimised.sdf"
        #         ff_positions, ff_energies = read_conformers_from_sdf(ff_sdf_path)
        #         ff_energies_and_positions[ff_name] = {
        #             "energies": ff_energies,
        #             "positions": ff_positions,
        #         }

        #     logger.info(f"Loaded {len(mlp_positions)} conformers from existing files")
        # else:
        #     # Calculate energies and positions
        #     logger.info(f"Calculating energies for {row['id']}")

        #     # Get the energies and positions from the ML potential
        #     mlp_energies = []
        #     mlp_positions = []
        #     ff_energies_and_positions = {
        #         ff_name: {"energies": [], "positions": []} for ff_name in ffs.keys()
        #     }

        #     ff_interchanges = {
        #         ff_name: Interchange.from_smirnoff(ffs[ff_name], mols[0].to_topology())
        #         for ff_name in ffs.keys()
        #     }
        #     ff_systems = {
        #         ff_name: interchange.to_openmm()
        #         for ff_name, interchange in ff_interchanges.items()
        #     }

        #     for mol in tqdm(mols, desc="Calculating energies for all conformers"):
        #         mol_mlp_energies, mol_mlp_positions = get_energies_and_positions_mlp(
        #             mol, mlp_name="egret-1"
        #         )
        #         mlp_energies.extend(mol_mlp_energies)
        #         mlp_positions.extend(mol_mlp_positions)
        #         off_positions = openmm_to_openff_positions(mol_mlp_positions)
        #         mol._conformers = off_positions

        #         for ff_name, ff_system in ff_systems.items():
        #             mol_ff_energies, mol_ff_positions = get_energies_ff(mol, ff_system)
        #             ff_energies_and_positions[ff_name]["energies"].extend(
        #                 mol_ff_energies
        #             )
        #             ff_energies_and_positions[ff_name]["positions"].extend(
        #                 mol_ff_positions
        #             )

        #     # Save MLP-minimised structures to CREST directory
        #     save_conformers_to_sdf(
        #         mols[0], mlp_positions, mlp_energies, mlp_sdf_path, name_prefix="mlp"
        #     )
        #     logger.info(f"Saved MLP-minimised structures to {mlp_sdf_path}")

        #     # Save MM force field minimised structures to analysis directory
        #     for ff_name, data in ff_energies_and_positions.items():
        #         ff_sdf_path = analysis_dir / f"{row['id']}_{ff_name}_minimised.sdf"
        #         save_conformers_to_sdf(
        #             mols[0],
        #             data["positions"],
        #             data["energies"],
        #             ff_sdf_path,
        #             name_prefix=ff_name,
        #         )
        #         logger.info(f"Saved {ff_name} minimised structures to {ff_sdf_path}")

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

        # Calculate the RMSE and MAE and store in DataFrame
        energy_stats = {}
        energy_stats_list = []
        for ff_name, energies in ff_energies_arrays.items():
            rmse = np.sqrt(np.mean((energies - mlp_energies_array) ** 2))
            mae = np.mean(np.abs(energies - mlp_energies_array))
            energy_stats[ff_name] = {"rmse": rmse, "mae": mae}
            energy_stats_list.append(
                {
                    "force_field": ff_name,
                    "rmse_kcal_mol": rmse,
                    "mae_kcal_mol": mae,
                }
            )
            logger.info(f"{ff_name} RMSE: {rmse:.4f} kcal/mol, MAE: {mae:.4f} kcal/mol")

        # Save energy stats to CSV
        energy_stats_df = pd.DataFrame(energy_stats_list)
        energy_stats_df.to_csv(analysis_dir / "energy_stats.csv", index=False)

        # Also save to text file for backward compatibility
        with open(analysis_dir / "energy_stats.txt", "w") as f:
            for ff_name, energies in ff_energies_arrays.items():
                rmse = energy_stats[ff_name]["rmse"]
                mae = energy_stats[ff_name]["mae"]
                f.write(
                    f"{ff_name} RMSE: {rmse:.4f} kcal/mol, MAE: {mae:.4f} kcal/mol\n"
                )

        # Plot the energies and save
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, (ff_name, energies) in enumerate(ff_energies_arrays.items()):
            ax.scatter(mlp_energies_array, energies, label=ff_name)
            # Annotate every dot with the index for the first force field only
            if i == 0:
                for i in range(len(mlp_energies_array)):
                    ax.annotate(str(i), (mlp_energies_array[i], energies[i]))

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
        ff_rmsds = {ff_name: [] for ff_name in ff_energies_and_positions.keys()}
        for i in range(len(mlp_positions)):
            mlp_pos = mlp_positions[i]
            for ff_name, data in ff_energies_and_positions.items():
                ff_pos = data["positions"][i]
                rmsd = get_rmsd(mols[0], mlp_pos, ff_pos)
                ff_rmsds[ff_name].append(rmsd)

        # Calculate RMSD statistics and store in DataFrame
        rmsd_stats = {}
        rmsd_stats_list = []
        for ff_name, rmsds in ff_rmsds.items():
            mean_rmsd = np.mean(rmsds)
            max_rmsd = np.max(rmsds)
            rms_rmsd = np.sqrt(np.mean(np.array(rmsds) ** 2))
            rmsd_stats[ff_name] = {
                "rms_rmsd": rms_rmsd,
                "mean_rmsd": mean_rmsd,
                "max_rmsd": max_rmsd,
            }
            rmsd_stats_list.append(
                {
                    "force_field": ff_name,
                    "rms_rmsd_angstrom": rms_rmsd,
                    "mean_rmsd_angstrom": mean_rmsd,
                    "max_rmsd_angstrom": max_rmsd,
                }
            )
            logger.info(
                f"{ff_name} RMS RMSD: {rms_rmsd:.4f} Å, Mean RMSD: {mean_rmsd:.4f} Å, Max RMSD: {max_rmsd:.4f} Å"
            )

        # Save RMSD stats to CSV
        rmsd_stats_df = pd.DataFrame(rmsd_stats_list)
        rmsd_stats_df.to_csv(analysis_dir / "rmsd_stats.csv", index=False)

        # Also save to text file for backward compatibility
        with open(analysis_dir / "rmsd_stats.txt", "w") as f:
            for ff_name, rmsds in ff_rmsds.items():
                rms_rmsd = rmsd_stats[ff_name]["rms_rmsd"]
                mean_rmsd = rmsd_stats[ff_name]["mean_rmsd"]
                max_rmsd = rmsd_stats[ff_name]["max_rmsd"]
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

    # Create summary DataFrames for all targets
    all_energy_stats = []
    all_rmsd_stats = []

    for target_id, stats in overall_results.items():
        # Energy stats
        for ff_name, energy_stat in stats["energy_stats"].items():
            all_energy_stats.append(
                {
                    "target": target_id,
                    "force_field": ff_name,
                    "rmse_kcal_mol": energy_stat["rmse"],
                    "mae_kcal_mol": energy_stat["mae"],
                }
            )

        # RMSD stats
        for ff_name, rmsd_stat in stats["rmsd_stats"].items():
            all_rmsd_stats.append(
                {
                    "target": target_id,
                    "force_field": ff_name,
                    "rms_rmsd_angstrom": rmsd_stat["rms_rmsd"],
                    "mean_rmsd_angstrom": rmsd_stat["mean_rmsd"],
                    "max_rmsd_angstrom": rmsd_stat["max_rmsd"],
                }
            )

    # Create DataFrames
    energy_summary_df = pd.DataFrame(all_energy_stats)
    rmsd_summary_df = pd.DataFrame(all_rmsd_stats)

    # Save summary CSVs
    energy_summary_df.to_csv(OUTPUT_DIR / "all_targets_energy_stats.csv", index=False)
    rmsd_summary_df.to_csv(OUTPUT_DIR / "all_targets_rmsd_stats.csv", index=False)

    logger.info(f"Saved summary statistics to {OUTPUT_DIR}")

    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: RMSE by force field
    ax = axes[0, 0]
    energy_pivot = energy_summary_df.pivot(
        index="target", columns="force_field", values="rmse_kcal_mol"
    )
    energy_pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Target")
    ax.set_ylabel("RMSE (kcal/mol)")
    ax.set_title("Energy RMSE by Target and Force Field")
    ax.legend(title="Force Field", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)

    # Plot 2: MAE by force field
    ax = axes[0, 1]
    mae_pivot = energy_summary_df.pivot(
        index="target", columns="force_field", values="mae_kcal_mol"
    )
    mae_pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Target")
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("Energy MAE by Target and Force Field")
    ax.legend(title="Force Field", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)

    # Plot 3: RMS RMSD by force field
    ax = axes[1, 0]
    rms_rmsd_pivot = rmsd_summary_df.pivot(
        index="target", columns="force_field", values="rms_rmsd_angstrom"
    )
    rms_rmsd_pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Target")
    ax.set_ylabel("RMS RMSD (Å)")
    ax.set_title("RMS RMSD by Target and Force Field")
    ax.legend(title="Force Field", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)

    # Plot 4: Mean RMSD by force field
    ax = axes[1, 1]
    mean_rmsd_pivot = rmsd_summary_df.pivot(
        index="target", columns="force_field", values="mean_rmsd_angstrom"
    )
    mean_rmsd_pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Target")
    ax.set_ylabel("Mean RMSD (Å)")
    ax.set_title("Mean RMSD by Target and Force Field")
    ax.legend(title="Force Field", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_targets_summary.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved summary plots to {OUTPUT_DIR / 'all_targets_summary.png'}")

    # Create box plots for each metric across all targets
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Box plot 1: RMSE
    ax = axes[0, 0]
    energy_summary_df.boxplot(column="rmse_kcal_mol", by="force_field", ax=ax)
    ax.set_xlabel("Force Field")
    ax.set_ylabel("RMSE (kcal/mol)")
    ax.set_title("Energy RMSE Distribution Across All Targets")
    plt.sca(ax)
    plt.xticks(rotation=45)

    # Box plot 2: MAE
    ax = axes[0, 1]
    energy_summary_df.boxplot(column="mae_kcal_mol", by="force_field", ax=ax)
    ax.set_xlabel("Force Field")
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("Energy MAE Distribution Across All Targets")
    plt.sca(ax)
    plt.xticks(rotation=45)

    # Box plot 3: RMS RMSD
    ax = axes[1, 0]
    rmsd_summary_df.boxplot(column="rms_rmsd_angstrom", by="force_field", ax=ax)
    ax.set_xlabel("Force Field")
    ax.set_ylabel("RMS RMSD (Å)")
    ax.set_title("RMS RMSD Distribution Across All Targets")
    plt.sca(ax)
    plt.xticks(rotation=45)

    # Box plot 4: Mean RMSD
    ax = axes[1, 1]
    rmsd_summary_df.boxplot(column="mean_rmsd_angstrom", by="force_field", ax=ax)
    ax.set_xlabel("Force Field")
    ax.set_ylabel("Mean RMSD (Å)")
    ax.set_title("Mean RMSD Distribution Across All Targets")
    plt.sca(ax)
    plt.xticks(rotation=45)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_targets_boxplots.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved box plots to {OUTPUT_DIR / 'all_targets_boxplots.png'}")


if __name__ == "__main__":
    typer.run(main)
