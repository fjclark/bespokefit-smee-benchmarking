PACKAGE_NAME := bespokefit_smee_benchmark

CONDA_ENV_RUN=conda run --no-capture-output --name $(PACKAGE_NAME)

TEST_ARGS := -v --cov=$(PACKAGE_NAME) --cov-report=term --cov-report=xml --junitxml=unit.xml --color=yes

.PHONY: env

env:
	mamba create --name $(PACKAGE_NAME)
	mamba env update --name $(PACKAGE_NAME) --file devtools/envs/base.yaml
	$(CONDA_ENV_RUN) pip install mace-torch
	$(CONDA_ENV_RUN) conda remove --force smee
	$(CONDA_ENV_RUN) pip install git+https://github.com/thomasjamespope/smee.git
	$(CONDA_ENV_RUN) pip install git+https://github.com/fjclark/bespokefit_smee.git --no-deps
	$(CONDA_ENV_RUN) pip install git+https://github.com/openforcefield/yammbs.git@torsion-analysis --no-deps
