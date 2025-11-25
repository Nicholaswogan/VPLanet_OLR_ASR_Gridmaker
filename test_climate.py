import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from makegrid import get_gridvals
from utils import AdiabatClimateVPL, ClimateModel

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=4)

def compare_surface_temperature_models(
    grid_file="ClimateGrid.h5",
    n_samples=100,
    seed=0,
    output_path="surface_temperature_comparison.png",
):
    """Compare ClimateModel vs AdiabatClimateVPL surface temperatures."""
    if not os.path.exists(grid_file):
        print(f"Skipping: {grid_file} not found. Generate the grid before running this script.")
        return None

    gridvals, _ = get_gridvals()
    T_grid, log10N_CO2_grid, stellar_flux_grid, surface_albedo_grid = gridvals
    T_bounds = (float(T_grid.min()), float(T_grid.max()))

    rng = np.random.default_rng(seed)
    log10N_CO2_samples = rng.uniform(-4.0, 1.0, n_samples)
    stellar_flux_samples = rng.uniform(800.0, 1300.0, n_samples)
    surface_albedo_samples = rng.uniform(float(surface_albedo_grid.min()), float(surface_albedo_grid.max()), n_samples)
    T_guess_samples = np.full(n_samples, 300.0)

    adiabat = AdiabatClimateVPL()
    climate = ClimateModel(grid_file)

    idx_h2o = adiabat.species_names.index("H2O")
    idx_n2 = adiabat.species_names.index("N2")
    idx_co2 = adiabat.species_names.index("CO2")

    adiabat_t = []
    climate_t = []
    failures = 0

    for logN, flux, alb, guess in zip(
        log10N_CO2_samples, stellar_flux_samples, surface_albedo_samples, T_guess_samples
    ):
        N_i = np.ones(len(adiabat.species_names)) * 1e-15
        N_i[idx_h2o] = 15e3  # Assume 1 ocean of H2O
        N_i[idx_n2] = 36.0   # ~1 bar N2
        N_i[idx_co2] = 10.0 ** logN

        try:
            t_column = adiabat.surface_temperature_column_custom(
                N_i=N_i,
                stellar_flux=float(flux),
                surface_albedo=float(alb),
                RH=1.0,
                bond_albedo=0.3,
                T_bounds=T_bounds,
                T_surf_guess=float(guess),
            )
        except Exception as exc:
            failures += 1
            print(
                "Failure in surface_temperature_column_custom:"
                f" log10N_CO2={logN:.16f}, stellar_flux={flux:.16f},"
                f" surface_albedo={alb:.16f}, T_guess={guess:.2f},"
                f" N_i={N_i.tolist()}, err={exc}"
            )
            continue

        try:
            t_grid = climate.surface_temperature(
                N_CO2=float(N_i[idx_co2]),
                stellar_flux=float(flux),
                surface_albedo=float(alb),
                T_bounds=T_bounds,
                T_surf_guess=float(guess),
            )
        except Exception as exc:
            failures += 1
            print(
                "Failure in surface_temperature:"
                f" log10N_CO2={logN:.16f}, stellar_flux={flux:.16f},"
                f" surface_albedo={alb:.16f}, T_guess={guess:.2f},"
                f" N_i={N_i.tolist()}, err={exc}"
            )
            continue

        adiabat_t.append(t_column)
        climate_t.append(t_grid)

    if not adiabat_t:
        print("No successful surface temperature evaluations; nothing to plot.")
        return None

    diffs = np.array(adiabat_t) - np.array(climate_t)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax_scatter = axes[0]
    ax_scatter.scatter(adiabat_t, climate_t, s=18, alpha=0.7)
    lims = [min(adiabat_t + climate_t), max(adiabat_t + climate_t)]
    ax_scatter.plot(lims, lims, "k--", linewidth=1)
    ax_scatter.set_xlabel("AdiabatClimateVPL surface T (K)")
    ax_scatter.set_ylabel("ClimateModel surface T (K)")
    ax_scatter.set_title(f"Scatter (n={len(adiabat_t)}, failures={failures})")

    ax_hist = axes[1]
    ax_hist.hist(diffs, bins=20, alpha=0.8, edgecolor="k")
    ax_hist.set_xlabel("Adiabat - Climate (K)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Temperature differences")

    fig.tight_layout()

    # If an explicit path wasn't provided, keep output in a temp directory to avoid clutter.
    if output_path is None:
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "surface_temperature_comparison.png")

    fig.savefig(output_path, dpi=200)
    print(f"Saved comparison plot to {output_path}")
    return output_path

def test_ClimateGrid():
    "Modern Earth-like test case"
    c = ClimateModel('ClimateGrid.h5')
    N_CO2 = 23*400e-6
    stellar_flux = 1370
    surface_albedo = 0.2

    # Benchmark average compute time over multiple runs
    import time
    n_runs = 100
    start = time.perf_counter()
    for _ in range(n_runs):
        T_surf = c.surface_temperature(
            N_CO2=N_CO2,
            stellar_flux=stellar_flux,
            surface_albedo=surface_albedo
        )
    elapsed = time.perf_counter() - start
    avg_us = (elapsed / n_runs) * 1e6

    print(
        "Modern-Earth benchmark:"
        f" N_CO2={N_CO2:.4e} mol/cm^2,"
        f" stellar_flux={stellar_flux} W/m^2,"
        f" surface_albedo={surface_albedo},"
        f" T_surf={T_surf:.3f} K,"
        f" avg_time={avg_us:.3f} µs over {n_runs} runs"
    )

if __name__ == "__main__":
    test_ClimateGrid()
    compare_surface_temperature_models()
