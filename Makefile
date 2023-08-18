
install:
	pip install -e .[dev,docs,database,devsim,femwell,gmsh,meow,meshwell,ray,sax,schematic,tidy3d,web]
	pre-commit install

dev: test-data meep gmsh elmer palace install

gmsh:
	sudo apt-get install -y python3-gmsh gmsh
	sudo apt install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev

meep:
	mamba install -c conda-forge pymeep=*=mpi_mpich_* nlopt -y

elmer:
	sudo apt-add-repository ppa:elmer-csc-ubuntu/elmer-csc-ppa
	sudo apt-get update
	sudo apt-get install -y elmerfem-csc mpich

palace:
	@if command -v singularity >/dev/null 2>&1; then \
		singularity pull palace.sif oras://ghcr.io/awslabs/palace:main; \
		echo "#!/bin/bash" > palace; \
		echo 'singularity exec ~/palace.sif /opt/palace/bin/palace "$@"' >> palace; \
		chmod +x palace; \
	else \
		sudo apt-get update; \
		sudo apt-get install -y build-essential ca-certificates coreutils curl environment-modules gfortran git \
		    gpg lsb-release python3 python3-distutils python3-venv unzip zip; \
		git clone -c feature.manyFiles=true https://github.com/spack/spack.git; \
		(export SPACK_ROOT=`pwd`/spack && . spack/share/spack/setup-env.sh && spack install palace); \
	fi

test:
	pytest

cov:
	pytest --cov=gplugins

test-data:
	git clone https://github.com/gdsfactory/gdsfactory-test-data.git -b test-data test-data

test-data-developers:
	git clone git@github.com:gdsfactory/gdsfactory-test-data.git -b test-data test-data

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

update-pre:
	pre-commit autoupdate

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

release:
	git push
	git push origin --tags

build:
	rm -rf dist
	pip install build
	python -m build

jupytext:
	jupytext docs/**/*.ipynb --to py

notebooks:
	jupytext docs/**/*.py --to ipynb

docs:
	jb build docs

.PHONY: drc doc docs
