"""Create configuration files for hyperparameter tuning."""

from bespokefit_smee.settings import WorkflowSettings, ParameterisationSettings
from itertools import product
from pathlib import Path
from loguru import logger

PLACEHOLDER_SMILES = "C"
OUTPUT_PATH = Path("configs")

HYPERPARAMETER_OPTIONS = {
    "ProperTorsion_reg_weight": [0.1, 1.0, 10.0, 100.0],
    "ImproperTorsion_reg_weight": [0.1, 1.0, 10.0, 100.0],
}


def main() -> None:
    """Create configuration files for hyperparameter tuning."""

    permutations = list(
        product(
            HYPERPARAMETER_OPTIONS["ProperTorsion_reg_weight"],
            HYPERPARAMETER_OPTIONS["ImproperTorsion_reg_weight"],
        )
    )

    # Add unregularised case
    permutations.append((0.0, 0.0))

    # Add case where we don't train propers and impropers at all (inf weight)
    permutations.append((float("inf"), float("inf")))

    for proper_weight, improper_weight in permutations:

        config = WorkflowSettings(
            n_iterations=1,
            parameterisation_settings=ParameterisationSettings(
                smiles=PLACEHOLDER_SMILES,
                initial_force_field="openff_unconstrained-2.3.0-rc2.offxml",
            ),
        )

        for name, weight in [
            ("ProperTorsions", proper_weight),
            ("ImproperTorsions", improper_weight),
        ]:
            if weight == float("inf"):
                del config.training_settings.parameter_configs[name]
            else:
                config.training_settings.parameter_configs[name].regularize = {
                    "k": weight
                }

        config_name = f"config_proper{proper_weight}_improper{improper_weight}.yaml"
        output_file = OUTPUT_PATH / config_name
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        config.to_yaml(output_file)
        logger.info(f"Created configuration file: {output_file}")


if __name__ == "__main__":
    main()
