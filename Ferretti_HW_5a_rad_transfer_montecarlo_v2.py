import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import astropy.constants as const
import astropy.units as u
import time
import matplotlib.animation as animation


rng = np.random.default_rng()
sigma_t = const.sigma_T.to(u.cm**2).value #cm^2
n = 10e20 #cm^




def trace_photon(l, L, rng):
    """Fast 1D version (z only). Returns (outcome, number of scatterings)."""
    x, mu = 0.0, 1.0                 # enter at x = 0 moving along +x, mu is cos(theta)
    n_scat = 0
    while True:
        s = -l * np.log(1.0 - rng.random())   # sample path length
        x += s * mu
        if x < 0:
            return "reflected", n_scat
        if x > L:
            return "transmitted", n_scat
        n_scat += 1
        mu = 2.0 * rng.random() - 1.0         # isotropic: mu uniform in [-1, 1]


def trace_photon_3d(l, L, rng):
    """3D version for plotting. Returns (N x 3 array of positions in cm, outcome)."""
    pos = np.zeros(3)
    d = np.array([0.0, 0.0, 1.0])
    pts = [pos.copy()]
    while True:
        s = -l * np.log(1.0 - rng.random())
        new = pos + s * d
        if new[2] < 0 or new[2] > L:              # photon left the slab
            zb = 0.0 if new[2] < 0 else L
            f = (zb - pos[2]) / (new[2] - pos[2])
            pts.append(pos + f * (new - pos))     # clip last step to boundary
            outcome = "reflected" if zb == 0.0 else "transmitted"
            return np.array(pts), outcome
        pos = new
        pts.append(pos.copy())
        mu = 2.0 * rng.random() - 1.0
        sinth = np.sqrt(1.0 - mu**2)
        phi = 2.0 * np.pi * rng.random()
        d = np.array([sinth * np.cos(phi), sinth * np.sin(phi), mu])


def find_good_path(l, L, rng, min_points=15, max_tries=5000):
    """Keep tracing photons until one has a path long enough to be interesting."""
    best = None
    for _ in range(max_tries):
        pts, outcome = trace_photon_3d(l, L, rng)
        if best is None or len(pts) > len(best[0]):
            best = (pts, outcome)
        if len(pts) >= min_points:
            return pts, outcome
    return best


#Animation
def animate_photon(pts, L, outcome, save=None):
    """Side-view animation: depth x (horizontal) vs y (vertical)."""
    pts_m = pts / 100.0              
    L_m = L / 100.0
    xs, ys = pts_m[:, 2], pts_m[:, 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvspan(0, L_m, color="lightgray", alpha=0.5, label="slab")
    pad = 0.1 * L_m
    ax.set_xlim(min(xs.min(), 0) - pad, max(xs.max(), L_m) + pad)
    span = max(np.abs(ys).max(), 0.1 * L_m) * 1.2
    ax.set_ylim(-span, span)
    ax.set_xlabel("depth x (m)")
    ax.set_ylabel("y (vertical motion) (m)")
    ax.set_title(f"Photon path ({len(xs) - 2} scatterings, {outcome})")
    ax.legend(loc="upper right")

    line, = ax.plot([], [], "-", lw=1, color="tab:blue")
    dot, = ax.plot([], [], "o", color="red")

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        return line, dot

    def update(i):
        line.set_data(xs[:i + 1], ys[:i + 1])    # path so far
        dot.set_data([xs[i]], [ys[i]])           # current position
        return line, dot

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=len(xs), interval=500, blit=True)
    if save:
        anim.save(save, writer=animation.PillowWriter(fps=2))
        print(f"Saved animation to {save}")
    else:
        plt.show()
    return anim      


#Command line Arg Parsing
def parse_args():
    p = ap.ArgumentParser(
        description="Monte Carlo photon transport through a slab of electrons "
                    "(isotropic Thomson scattering)."
    )
    p.add_argument("-n", "--nphotons", type=int, default=10000,
                   help="number of photons to simulate (default: 10000)")
    p.add_argument("-w", "--width", type=float, default=1000.0,
                   help="slab width in meters (default: 1000)")
    p.add_argument("-d", "--density", type=float, default=1e20,
                   help="electron density in cm^-3 (default: 1e20)")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducible runs")
    p.add_argument("--animate", action="store_true",
                   help="animate the path of one photon")
    p.add_argument("--save", type=str, default=None,
                   help="save the animation to this file (e.g. photon.gif)")
    args = p.parse_args()

    if args.nphotons <= 0:
        p.error("--nphotons must be positive")
    if args.width <= 0:
        p.error("--width must be positive")
    if args.density <= 0:
        p.error("--density must be positive")
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    l = 1.0 / (args.density * sigma_t)    # mean free path, cm
    L = args.width * 100.0                # slab width, m -> cm
    tau = L / l                           # optical depth

    print(f"Thomson cross-section: {sigma_t:.4e} cm^2")
    print(f"Mean free path:        {l / 100:.1f} m")
    print(f"Slab width:            {args.width:.1f} m  (tau = {tau:.2f})")
    print(f"Simulating {args.nphotons} photons...")

    n_trans = 0
    total_scat = 0
    for _ in range(args.nphotons):
        outcome, n_scat = trace_photon(l, L, rng)
        total_scat += n_scat
        if outcome == "transmitted":
            n_trans += 1
    n_refl = args.nphotons - n_trans

    print(f"Transmitted: {n_trans / args.nphotons:.3f}")
    print(f"Reflected:   {n_refl / args.nphotons:.3f}")
    print(f"Mean scatterings per photon: {total_scat / args.nphotons:.1f}")

    if args.animate or args.save:
        pts, outcome = find_good_path(l, L, rng)
        animate_photon(pts, L, outcome, save=args.save)


if __name__ == "__main__":
    main()