"""Utilities to help with the workflow."""

import copy
from pathlib import Path
from typing import List

import loguru
import pandas as pd
from bespokefit_smee.settings import WorkflowSettings
from bespokefit_smee.workflow import get_bespoke_force_field
from openff.toolkit import ForceField
from openff.toolkit.typing.engines.smirnoff import ForceField
from tqdm import tqdm

logger = loguru.logger

BASE_FF = "openff_unconstrained-2.3.0-rc2.offxml"


def combine_force_fields(
    ff_to_combine: dict[str, ForceField],
    output_file: str,
    base_ff: str = "openff_unconstrained-2.3.0-rc2.offxml",
) -> ForceField:
    """
    Combines multiple OpenFF force field XML files into a single force field, starting with a base force field.
    Parameters:
        base_file (str): Path to the base force field XML file.
        output_file (str): Path to the output XML file for the combined force field.
        *input_files (str): Paths to the input force field XML files to combine.
    Returns:
        ForceField: The combined force field.
    """
    # Load the base force field
    original_force_field = ForceField(base_ff)
    combined_force_field = ForceField(base_ff)

    for ff_name, ff in ff_to_combine.items():
        # Load each force field and add its parameters to the combined force field
        for handler_name in ff.registered_parameter_handlers:
            handler = ff.get_parameter_handler(handler_name)
            combined_handler = combined_force_field.get_parameter_handler(handler_name)
            original_handler = original_force_field.get_parameter_handler(handler_name)
            existing_parameter_smirks = {
                param.smirks for param in combined_handler.parameters
            }
            original_parameter_smirks = {
                param.smirks for param in original_handler.parameters
            }
            new_parameters = existing_parameter_smirks - original_parameter_smirks

            for parameter in handler.parameters:
                # Skip constraints
                if handler_name == "Constraints":
                    continue

                # Make the parameter id unique by adding the input file directory name
                parameter.id += f"_{ff_name}"

                # Skip parameters that are already included in the base force field
                if parameter.smirks in original_parameter_smirks:
                    continue

                # Raise an error if a parameter is already present in the combined force field
                if parameter.smirks in new_parameters:
                    current_new_params = combined_handler[parameter.smirks]
                    # Check if the dicts (other than id) are the same
                    if all(
                        parameter.to_dict()[key] == current_new_params.to_dict()[key]
                        for key in parameter.to_dict()
                        if key != "id"
                    ):
                        logger.info(
                            f"Parameter {parameter.smirks} from {ff_name} is identical to existing parameter. Skipping."
                        )
                        continue
                    raise ValueError(
                        f"New parameter ID {parameter.id} {parameter} already exists in the combined force field."
                    )

                combined_handler.add_parameter(parameter.to_dict())

    # Save the combined force field to the output file
    combined_force_field.to_file(output_file)
    print(f"Combined force field saved to {output_file}")

    return combined_force_field


def run_bespokefit_single_smiles(
    config: WorkflowSettings,
    output_dir: Path,
    smiles: str,
    smiles_id: str,
    from_precomputed: bool = False,
) -> ForceField:
    """Run BespokeFit with the given configuration, output directory, and SMILES."""

    # Avoid modifying the original config
    config_copy = copy.deepcopy(config)

    # Load the training configuration
    config_copy.parameterisation_settings.smiles = smiles
    config_copy.output_dir = output_dir

    # If from precomputed is True, we need to overwrite the CHANGEME placeholders
    # in the dataset and force field paths
    if from_precomputed:
        config_copy.training_sampling_settings.dataset_path = (
            config_copy.training_sampling_settings.dataset_path.as_posix().replace(
                "CHANGEME", smiles_id
            )
        )
        config_copy.testing_sampling_settings.dataset_path = (
            config_copy.testing_sampling_settings.dataset_path.as_posix().replace(
                "CHANGEME", smiles_id
            )
        )
        config_copy.parameterisation_settings.initial_force_field = (
            config_copy.parameterisation_settings.initial_force_field.replace(
                "CHANGEME", smiles_id
            )
        )

    final_ff = get_bespoke_force_field(config_copy)

    return final_ff


def run_bespokefit_all_smiles(
    config_path: str,
    smiles_file: str,
    workflow_dir: str,
    name: str,
    from_precomputed: bool = False,
) -> ForceField:
    """Run BespokeFit for all SMILES in the workflow directory. Expects a 'smiles.csv' in
    workflow_dir / input"""

    # Load the training configuration
    config = WorkflowSettings.from_yaml(config_path)

    # Load the SMILES from the input file
    input_path = Path(smiles_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    smiles_df = pd.read_csv(input_path)

    for smiles_id, smiles in tqdm(
        zip(smiles_df["id"], smiles_df["smiles"]),
        desc="Processing SMILES",
        total=len(smiles_df),
    ):
        # Create a directory for each SMILES
        smiles_dir = (Path(workflow_dir) / "output" / name / str(smiles_id)).resolve()

        # If it already exists, skip it
        if smiles_dir.exists():
            logger.info(f"Skipping existing directory: {smiles_dir}")
            continue

        logger.info(f"Processing SMILES: {smiles[1]} in directory {smiles_dir}")
        smiles_dir.mkdir(exist_ok=True, parents=True)

        # Save the SMILES to a file
        with open(smiles_dir / "molecule.smi", "w") as f:
            f.write(str(smiles))

        # Run BespokeFit
        run_bespokefit_single_smiles(
            config,
            output_dir=smiles_dir,
            smiles=smiles,
            smiles_id=str(smiles_id),
            from_precomputed=from_precomputed,
        )

    # Combine all force fields in the output directory
    force_fields = {}
    output_path = Path(workflow_dir) / "output" / name
    for smiles_id in smiles_df["id"]:
        # file = output_path / str(smiles_id) / f"trained-{config.n_iterations}.offxml"
        # Find the highest iteration number available
        max_iteration = 0
        for iteration_dir in (output_path / str(smiles_id)).glob(
            "training_iteration_*"
        ):
            iteration_number = int(iteration_dir.name.split("_")[-1])
            if iteration_number > max_iteration:
                max_iteration = iteration_number
        if max_iteration == 0:
            raise ValueError(
                f"No training iterations found for SMILES ID {smiles_id} in {output_path / str(smiles_id)}"
            )
        file = (
            output_path / str(smiles_id) / f"training_iteration_1" / "bespoke_ff.offxml"
        )
        logger.info(f"Loading force field from {file}")
        force_fields[str(smiles_id)] = ForceField(str(file))

    output_ff_path = output_path / "combined_forcefield.offxml"
    logger.info(f"Combining {len(force_fields)} force fields into {output_ff_path}")

    return combine_force_fields(
        force_fields,
        output_file=str(output_ff_path),
        base_ff=BASE_FF,
    )
