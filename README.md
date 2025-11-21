## Setup

### 1) Create the conda environment

This installs `photochem=0.6.8` and the dependencies needed for `makegrid.py`.

```bash
conda env create -f environment.yaml
conda activate climate-grid
```

### 2) Running makegrid.py

With the conda environment active:

```bash
mpiexec -np 4 python makegrid.py
```

Exchange `4` with any number of parallel processes. This will generate/update `ClimateGrid.h5` and log to `ClimateGrid.log`.

### 3) Build the C library and run the C test

Uses the provided `CMakeLists.txt` to build the static library and the test executable.

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./test_climate_c # expects ClimateGrid.h5 in the repo root
```
