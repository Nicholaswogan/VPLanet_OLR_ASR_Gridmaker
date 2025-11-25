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
./test_climate # expects ClimateGrid.h5 in the repo root
```

Requires HDF5 development headers/libraries on your system. The C test prints a Modern-Earth benchmark surface temperature and the average solve time in microseconds.

### 4) Compare Python column solver vs grid solver

Run the standalone comparison script to sample random cases, solve with both solvers, and save a scatter+histogram plot to `surface_temperature_comparison.png`:

```bash
python test_climate.py
```

It prints the Modern-Earth benchmark (including average solve time in microseconds) and reports the plot location. The script expects `ClimateGrid.h5` in the repo root.

