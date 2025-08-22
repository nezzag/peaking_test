## Peaking Test
This is a repository where a test for identifying whether emissions have entered a structural peak or not is developed

### Contributors
- Neil Grant
- Claire Fyson

### What's Here
The repository contains the following core folders:

- `data/` can be used to store data (largely emissions timeseries that are being tested)

- `notebooks` is where notebooks that enable exploratory analysis to be conducted exist

- `scripts` is where the EmissionsPeakTest class is defined, and has a default implementation of this using the Global Carbon Budget data

## Getting started
- Download and clone this repository.
- Create project enviroment using the config file in this directory (optional, the cpt_generic_environment.yml is also sufficient for this repository):

```
mamba env create -f environment.yml
```

### To Do