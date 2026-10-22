import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import astropy.constants as const
import astropy.units as u
from astropy.io import fits
from pathlib import Path

parser = ap.ArgumentParser(
    description='Creates the Fourier transform for a system of occulting binaries and plots the Power Spectra'
    'The script runs using TESS data for TIC 360824561. '
    'It displays the flux recordings over time for three different epochs,'
    ' and then calculates the Power Spectra using a fast Fourier Transform.'
)


hdul = fits.open(Path(__file__).parent / "tic0360824561.fits") #find the data in a the folder from which this script is running

hdul.info() #I ran this first to know what data title to use in the next line
TESS_data = hdul['TESS-SPOC'].data
times = TESS_data['times']
fluxes = TESS_data['fluxes'] #brightness measurement
ferrs = TESS_data['ferrs'] #uncertainty on each flux measurement

dt = np.diff(times)

# Splits wherever there is a gap longer than 1 day
breaks = np.where(dt > 1.0)[0]
starts = np.r_[0, breaks + 1]
ends = np.r_[breaks, len(times) - 1]

epochs=[(1657,1682.3),(2036,2060.2),(3126,3150.8)] #I got this from printing the data found using starts and ends

if __name__ == '__main__':

    for s, e in zip(starts, ends):
        seg = times[s:e + 1]
        step = np.median(np.diff(seg)) * 24 * 60
        print(f"{seg[0]:.1f} to {seg[-1]:.1f}   N={len(seg)}   cadence={step:.2f} min")

    #looking at each epoch of data

    #looking at epoch 1
    end_epoch1 = np.searchsorted(times, 1682)
    t_epoch1 = times[0:end_epoch1]
    fluxes_epoch1 = fluxes[0:end_epoch1]

    #looking at epoch 2
    start_epoch2 = np.searchsorted(times, 2036)
    end_epoch2 = np.searchsorted(times, 2060.2)
    t_epoch2 = times[start_epoch2:end_epoch2]
    fluxes_epoch2 = fluxes[start_epoch2:end_epoch2]

    #looking at epoch 3
    start_epoch3 = np.searchsorted(times, 3126)
    end_epoch3 = np.searchsorted(times, 3150.8)
    t_epoch3 = times[start_epoch3:end_epoch3]
    fluxes_epoch3 = fluxes[start_epoch3:end_epoch3]


    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    fig.suptitle("Fluxes Over All Recorded Time and in Recorded Epochs", fontsize=16)

    
    # Panels 1-3: one per epoch
    for i, (lo, hi) in enumerate(epochs):
        ax = axes.flat[i]
        m = (times > lo) & (times < hi)
        ax.plot(times[m], fluxes[m], '.', ms=2, label="TESS-SPOC flux")
        ax.set_title(f"Epoch {i + 1}: days {lo} to {hi}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Normalized flux")
        ax.legend(loc="lower right")

    # Panel 4 shows everything
    ax = axes.flat[3]
    ax.plot(times, fluxes, '.', ms=1, label="TESS-SPOC flux")
    ax.set_title("All recorded data")
    ax.set_xlabel("Day")
    ax.set_ylabel("Normalized flux")
    ax.legend(loc="lower right")

    plt.show()

    P_orb = 1.495248760514598          # catalog period (days)
    f_orb = 1 / P_orb                  # orbital frequency (cycles/day)
    epochs = [(1657, 1682.3), (2036, 2060.2), (3126, 3150.8)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), layout="constrained")
    fig.suptitle("Interpolated Light Curves and Power Spectra for Each Epoch", fontsize=16)

    results = {}   #set up empty dictionary to fill with results
    for k, (lo, hi) in enumerate(epochs):
        # 1. slice the epoch
        m = (times >= lo) & (times <= hi)
        t, f = times[m], fluxes[m]

        #interpolate onto an even grid
        cadence = np.median(np.diff(t))
        n = int(round((t[-1] - t[0]) / cadence)) + 1
        t_u = np.linspace(t[0], t[-1], n)
        f_u = np.interp(t_u, t, f)

        # flag the grid points that sit inside a gap
        i = np.searchsorted(t, t_u)
        left = t[np.clip(i - 1, 0, len(t) - 1)]
        right = t[np.clip(i, 0, len(t) - 1)]
        filled = (right - left) > 1.5 * cadence

        # FFT
        coeffs = np.fft.rfft(f_u - f_u.mean())
        freqs = np.fft.rfftfreq(n, d=t_u[1] - t_u[0])      # cycles per day
        power = np.abs(coeffs)**2
        results[k + 1] = dict(t=t_u, f=f_u, mean=f_u.mean(),
                            coeffs=coeffs, freqs=freqs, power=power)

        # top row: light curve
        ax = axes[0, k]
        ax.plot(t, f, '.', ms=2, label="original data")
        ax.plot(t_u[filled], f_u[filled], '.', ms=2, color="red", label="interpolated")
        ax.set_title(f"Epoch {k + 1}: days {lo} to {hi}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Normalized flux")
        ax.legend(loc="lower right")

        # bottom row: power spectrum
        ax = axes[1, k]
        ax.semilogy(freqs[1:], power[1:], lw=0.8, label="power")   # skip the zero-frequency bin
        for h in range(1, 11):
            ax.axvline(h * f_orb, color="gray", ls=":", lw=0.6,
                    label=r"harmonics of $f_{orb}$" if h == 1 else None)
        ax.set_xlim(0, 10)
        ax.set_xlabel("Frequency (cycles/day)")
        ax.set_ylabel("Power")
        ax.set_title(f"Epoch {k + 1} power spectrum")
        ax.legend(loc="upper right", fontsize=8)

    plt.show()

    def harmonic_reconstruction(coeffs, freqs, n, mean, K, f0=f_orb, halfwidth=1):
        """Keep only the bins at k*f0 (k = 1..K), plus `halfwidth` neighbors on each side."""
        keep = np.zeros(len(coeffs), dtype=bool)
        df = freqs[1]
        for k in range(1, K + 1):
            j = int(round(k * f0 / df))
            keep[max(j - halfwidth, 0): j + halfwidth + 1] = True
        rec = np.fft.irfft(np.where(keep, coeffs, 0), n=n) + mean
        return rec, keep.sum()

    for ep, r in results.items():
        n = len(r['t'])
        print(f"\nEpoch {ep}: {n} grid points, {len(r['coeffs'])} FFT coefficients")
        for K in (1, 2, 4, 8, 12, 16, 20, 30):
            rec, ncoef = harmonic_reconstruction(r['coeffs'], r['freqs'], n, r['mean'], K)
            rms = np.sqrt(np.mean((rec - r['f'])**2))
            print(f"  K={K:2d} harmonics ({ncoef:3d} coeffs): min flux={rec.min():.4f}, rms residual={rms:.2e}")

    r = results[2]
    n = len(r['t'])
    fig4, ax4 = plt.subplots(figsize=(14, 5), layout="constrained")
    win = (r['t'] > 2040) & (r['t'] < 2046)
    ax4.plot(r['t'][win], r['f'][win], '.', ms=3, color="lightgray", label="interpolated data")
    for K in (2, 6, 16):
        rec, ncoef = harmonic_reconstruction(r['coeffs'], r['freqs'], n, r['mean'], K)
        ax4.plot(r['t'][win], rec[win], label=f"{K} harmonics ({ncoef} coeffs)")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Normalized flux")
    ax4.set_title("Epoch 2 reconstruction from the first K harmonics")
    ax4.legend()
    plt.show()