import subprocess

import loguru

logger = loguru.logger


def analyse(
    experiment_name: str,
    force_field_path: Path,
    base_force_field: str,
    torsion_data_path: Path,
) -> None:

    command = [
        "yammbs_analyse_torsions",
        "--base-force-fields",
        base_force_field,
        "--extra-force-fields",
        str(force_field_path),
        "--qcarchive-torsion-data",
        str(torsion_data_path),
    ]

    logger.info(f"Running command: {' '.join(command)}")

    subprocess.run(
        command,
        check=True,
        capture_output=True,
    )

    logger.info(f"Analysis completed for experiment: {experiment_name}")
