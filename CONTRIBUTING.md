# CONTRIBUTING

- [CONTRIBUTING](#contributing)
  - [Docker and singularity](#docker-and-singularity)
  - [Dependencies version](#dependencies-version)
  - [Java dependencies version](#java-dependencies-version)

## Docker and singularity

The base recipe was generated using
[neurodocker](https://github.com/ReproNim/neurodocker) from the script
`create_container_recipe.sh`.

If possible do not modify the Dockerfile directly but modify it via this script.

Creating the Dockerfile, building it and testing it can be done with:

```bash
make docker_test
```

The test only run examples 1, 2 and 3 from the `examples` folder.

## Dependencies version

The `conda-nighres.yml` specifies the version of the minimum dependencies.

If you want to freeze all your dependencies you can do so with
`conda env export > conda-nighres.yml`.

Note that if you do this you will need to remove somes lines (for example that
concerns nighres as it is a local pacakage.)

```bash
make clean_env_file
```

If you need to update the following packages, do it in the
[`setup.py`](./setup.py) file:

- `numpy`
- `nibabel`
- `psutil`
- `antspyx`

Then run the following to update `conda-nighres.yml`

```bash
pip install .
conda env export > conda-nighres.yml
make clean_env_file
```

The following pacakages that are necessary for setting up the environment
building nighres JAVA or for running examples can be updated directly.

- `pip`
- `jcc`
- `Nilearn`
- `dipy`
- `gcc_linux-64`
- `gxx_linux-64`

This can be done with:

```bash
conda update -n nighres pip jcc Nilearn dipy gcc_linux-64 gxx_linux-64
pip install .
conda env export > conda-nighres.yml
make clean_env_file
```

## Java dependencies version

The versions of the Java dependencies for nighres are in `dependencies_sha.sh`.

They contain the commit id (SHA1) from the cbstools and imcntk repo.

For example.

```bash
cbstools_sha=7a34255
imcntk_sha=ea901d8
```

Those values are then used by the build process (`build.sh`).

To update those to the latest commit on the master branch simply use
`make update_dep_shasum`.
