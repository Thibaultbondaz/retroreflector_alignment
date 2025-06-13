"""
retroreflector_alignment.py
===========================
Interactive helper for aligning a three‑mirror cube‑corner retroreflector.
Written with a considerable amount generative AI, so apologies for the "bot-like" coding.

I did test the code myself and reached a deviation of ~45 arcsec by just eyeballing the correction angles. I imagine a much greater precision can be achieved by employing better tools/finer threaded screws. 
---------------------------------------------
Dependencies: only *numpy* (pip install numpy).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def spot_shift_to_error(dx_mm: float, dy_mm: float, L_m: float) -> np.ndarray:
    """Convert screen shift (dx,dy) in *mm* at distance *L_m* to (αx, αy) in radians."""
    return np.array([dx_mm, dy_mm]) / 1000.0 / (2.0 * L_m)

def measure_sensitivity(L_m: float, num_screws: int = 6) -> np.ndarray:
    """Interactively acquire the 2 × *num_screws* sensitivity matrix **J**."""
    print("\n📏  Sensitivity‑matrix acquisition (leave blank to quit)\n")
    J = np.zeros((2, num_screws))
    for j in range(num_screws):
        print(f"→ Screw {j+1}")
        try:
            dturn = float(input("   test turn [+ turns CCW]: "))
            dx    = float(input("   spot Δx [mm] (+ right):  "))
            dy    = float(input("   spot Δy [mm] (+ up):     "))
        except ValueError:
            print("   aborted…\n")
            sys.exit(1)
        dalpha   = spot_shift_to_error(dx, dy, L_m)
        J[:, j]  = dalpha / dturn
    return J

def save_matrix(path: Path, J: np.ndarray):
    np.savetxt(path, J, delimiter=",", fmt="%.8e")
    print(f"Saved sensitivity matrix → {path}")

def load_matrix(path: Path) -> np.ndarray:
    try:
        J = np.loadtxt(path, delimiter=",")
        if J.shape != (2, 6):
            raise ValueError("J must be 2×6")
        return J
    except Exception as e:
        raise IOError(f"Could not read {path}: {e}")

# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute screw corrections for a cube‑corner retroreflector.")
    parser.add_argument("--distance", "-L", type=float, default=None,
                        help="screen distance L [metres] (ask if omitted)")
    parser.add_argument("--deltaL", type=float, default=None,
                        help="slide‑test cube displacement ΔL [metres] (positive toward screen)")
    parser.add_argument("--sensitivity", "-J", type=Path, default=None,
                        help="CSV file containing 2×6 sensitivity matrix")
    parser.add_argument("--fraction", "-f", type=float, default=0.5,
                        help="fraction of computed Δθ to apply (0–1)")
    args = parser.parse_args()

    # 1 – Screen distance L
    L = args.distance
    if L is None:
        try:
            L = float(input("Screen distance L [metres]: "))
        except ValueError:
            print("Invalid number; exiting.")
            sys.exit(1)

    # 2 – Mode selection / ΔL
    deltaL = args.deltaL
    if deltaL is None:
        mode = input("Choose mode — [1] absolute‑spot  [2] slide‑test : ").strip()
        if mode == "2":
            try:
                deltaL = float(input("Cube translation ΔL [metres] (positive toward screen): "))
            except ValueError:
                print("Invalid number; exiting.")
                sys.exit(1)
        else:
            deltaL = None  # absolute‑spot mode
    if deltaL is not None and abs(deltaL) < 1e-6:
        print("ΔL must be non‑zero.")
        sys.exit(1)

    frac = args.fraction

    # 3 – Load or measure sensitivity matrix J
    if args.sensitivity and args.sensitivity.exists():
        J = load_matrix(args.sensitivity)
        print(f"Loaded sensitivity matrix from {args.sensitivity}")
    else:
        print("No existing sensitivity matrix provided – entering measurement mode.")
        J = measure_sensitivity(L)
        if args.sensitivity:
            save_matrix(args.sensitivity, J)

    # Pre‑compute pseudo‑inverse
    J_pinv = np.linalg.pinv(J)  # 6×2

    # ------------------------------------------------------------------
    # Alignment loop
    # ------------------------------------------------------------------
    if deltaL is None:
        print("\n⚙️  Alignment loop (absolute‑spot) – Ctrl‑C to quit")
    else:
        print("\n⚙️  Alignment loop (slide‑test, ΔL = %.3f m) – Ctrl‑C to quit" % deltaL)

    while True:
        try:
            if deltaL is None:
                dx = float(input("\n  Spot shift dx [mm]: "))
                dy = float(input("  Spot shift dy [mm]: "))
                dx_equiv, dy_equiv = dx, dy
            else:
                dxp = float(input("\n  Spot shift after slide dx' [mm]: "))
                dyp = float(input("  Spot shift after slide dy' [mm]: "))
                dx_equiv = dxp * (L / deltaL)
                dy_equiv = dyp * (L / deltaL)
                print(f"  (Equivalent baseline error: dx = {dx_equiv:.3f} mm, dy = {dy_equiv:.3f} mm)")
        except KeyboardInterrupt:
            print("\nbye")
            break
        except ValueError:
            print("  invalid number – try again")
            continue

        # Convert to angular error and solve for screw moves
        e      = spot_shift_to_error(dx_equiv, dy_equiv, L)
        dtheta = -J_pinv @ e
        adj    = dtheta * frac

        print(f"\n→ Apply {frac:g} × Δθ:")
        for idx, val in enumerate(adj, 1):
            print(f"  Screw{idx}: {val:+.3f} turn(s)")

if __name__ == "__main__":  # pragma: no cover
    main()
