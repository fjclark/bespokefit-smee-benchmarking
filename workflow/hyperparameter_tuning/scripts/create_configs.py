"""Create configuration files for hyperparameter tuning."""

from bespokefit_smee.settings import WorkflowSettings, ParameterisationSettings
from itertools import product
from pathlib import Path
from loguru import logger

PLACEHOLDER_SMILES = "C"
OUTPUT_PATH = Path("configs")


def main() -> None:
    """Create configuration files for hyperparameter tuning."""

    config = WorkflowSettings(
        n_iterations=1,
        parameterisation_settings=ParameterisationSettings(
            smiles=PLACEHOLDER_SMILES,
            initial_force_field="openff_unconstrained-2.3.0-rc2.offxml",
        ),
    )
    for name in ["ProperTorsions", "ImproperTorsions"]:
        config.training_settings.parameter_configs[name].regularize = {"k": 1.0}

    config_name = "config_metad_std.yaml"
    output_file = OUTPUT_PATH / config_name
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    config.to_yaml(output_file)
    logger.info(f"Created configuration file: {output_file}")

    # Make a new config with updated metadynamics settings
    config.training_sampling_settings.temperature = "300 K"
    config.training_sampling_settings.production_sampling_time_per_conformer = "100 ps"
    config.training_sampling_settings.bias_frequency = "0.4 ps"
    config.training_sampling_settings.bias_width = 0.017453292519943295
    config.training_sampling_settings.bias_height = "0.2 kJ * mol**-1"
    config.training_sampling_settings.bias_factor = 5.0

    config_name = "config_metad_tuned.yaml"
    output_file = OUTPUT_PATH / config_name
    config.to_yaml(output_file)
    logger.info(f"Created configuration file: {output_file}")


if __name__ == "__main__":
    main()
