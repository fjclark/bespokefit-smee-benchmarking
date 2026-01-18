"""Create configuration files for hyperparameter tuning."""

from bespokefit_smee.settings import (
    WorkflowSettings,
    ParameterisationSettings,
    PreComputedDatasetSettings,
)
from itertools import product
from pathlib import Path
from loguru import logger

import sys

PLACEHOLDER_SMILES = "C"
OUTPUT_PATH = Path("configs")

HYPERPARAMETER_OPTIONS = {
    "Torsion_reg_weight": [1.0, 10.0, 100.0],
    "Angle_reg_weight": [1.0, 10.0, 100.0],
}


def main(workflow_dir: str) -> None:
    """Create configuration files for hyperparameter tuning."""

    reference_output_dir = (
        Path(workflow_dir) / "output" / "ABC_no_map_msm_reference_sampling"
    ).resolve()

    permutations = list(
        product(
            HYPERPARAMETER_OPTIONS["Torsion_reg_weight"],
            HYPERPARAMETER_OPTIONS["Angle_reg_weight"],
        )
    )

    # Add unregularised case
    permutations.append((0.0, 0.0))

    for torsion_weight, angle_weight in permutations:

        config = WorkflowSettings(
            n_iterations=1,
            parameterisation_settings=ParameterisationSettings(
                smiles=PLACEHOLDER_SMILES,
                initial_force_field=str(
                    reference_output_dir
                    / "CHANGEME"
                    / "initial_statistics"
                    / "bespoke_ff.offxml"
                ),
                msm_settings=None,
            ),
            training_sampling_settings=PreComputedDatasetSettings(
                dataset_path=reference_output_dir
                / "CHANGEME"
                / "training_iteration_1"
                / "energy_and_force_data_mol0"
            ),
            testing_sampling_settings=PreComputedDatasetSettings(
                dataset_path=reference_output_dir
                / "CHANGEME"
                / "test_data"
                / "energy_and_force_data_mol0"
            ),
        )

        for name, weight in [
            ("Torsions", torsion_weight),
            ("LinearAngles", angle_weight),
        ]:

            if name == "Torsions":
                for specific_name in ["ProperTorsions", "ImproperTorsions"]:
                    config.training_settings.parameter_configs[
                        specific_name
                    ].regularize = {"k": weight}

            elif name == "LinearAngles":
                config.training_settings.parameter_configs[name].regularize = {
                    "k1": weight,
                    "k2": weight,
                }

        config_name = f"config_torsion_{torsion_weight}_angle_{angle_weight}.yaml"
        output_file = OUTPUT_PATH / config_name
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        config.to_yaml(output_file)
        logger.info(f"Created configuration file: {output_file}")


if __name__ == "__main__":
    main(sys.argv[1])
