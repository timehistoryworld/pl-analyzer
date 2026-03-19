"""
Fitting models and utilities for PL Analyzer
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings


# ── Gaussian / Voigt models ──────────────────────────────────────────────────

def gaussian(x, amp, center, sigma):
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def lorentzian(x, amp, center, gamma):
    return amp * (gamma**2) / ((x - center)**2 + gamma**2)

def voigt_approx(x, amp, center, sigma, gamma):
    """Pseudo-Voigt approximation (Thompson et al. 1987)"""
    f_G = 2 * sigma * np.sqrt(2 * np.log(2))
    f_L = 2 * gamma
    f = (f_G**5 + 2.69269*f_G**4*f_L + 2.42843*f_G**3*f_L**2 +
         4.47163*f_G**2*f_L**3 + 0.07842*f_G*f_L**4 + f_L**5) ** 0.2
    eta = 1.36603*(f_L/f) - 0.47719*(f_L/f)**2 + 0.11116*(f_L/f)**3
    return amp * (eta * lorentzian(x, 1, center, f/2) +
                  (1-eta) * gaussian(x, 1, center, f/(2*np.sqrt(2*np.log(2)))))

def multi_gaussian(x, *params):
    """Sum of N Gaussians. params = [amp1,cen1,sig1, amp2,cen2,sig2, ...]"""
    n = len(params) // 3
    result = np.zeros_like(x, dtype=float)
    for i in range(n):
        amp, cen, sig = params[3*i], params[3*i+1], params[3*i+2]
        result += gaussian(x, amp, cen, abs(sig))
    return result


# ── Peak detection ───────────────────────────────────────────────────────────

def detect_peaks(wavelength, intensity, prominence=0.05, min_distance_nm=10):
    """
    Detect peaks in spectrum.
    Returns indices, peak wavelengths, peak intensities.
    """
    norm_intensity = intensity / np.max(np.abs(intensity) + 1e-30)
    spacing = np.mean(np.diff(wavelength))
    min_dist_pts = max(1, int(min_distance_nm / spacing))
    
    peaks, props = find_peaks(norm_intensity,
                               prominence=prominence,
                               distance=min_dist_pts)
    return peaks, wavelength[peaks], intensity[peaks]


def fit_peak_gaussian(wavelength, intensity, peak_idx, window_nm=30):
    """
    Fit a single Gaussian around a peak.
    Returns (popt, pcov, fit_x, fit_y) or None on failure.
    """
    center_wl = wavelength[peak_idx]
    spacing = np.mean(np.diff(wavelength))
    half_pts = int((window_nm / 2) / spacing)
    
    i_lo = max(0, peak_idx - half_pts)
    i_hi = min(len(wavelength), peak_idx + half_pts)
    
    x = wavelength[i_lo:i_hi]
    y = intensity[i_lo:i_hi]
    
    amp0 = intensity[peak_idx]
    sig0 = window_nm / 6
    
    try:
        popt, pcov = curve_fit(gaussian, x, y,
                                p0=[amp0, center_wl, sig0],
                                maxfev=10000)
        fit_x = np.linspace(x[0], x[-1], 300)
        fit_y = gaussian(fit_x, *popt)
        return popt, pcov, fit_x, fit_y
    except Exception:
        return None


def fwhm_from_gaussian_sigma(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * abs(sigma)


# ── TRPL decay models ────────────────────────────────────────────────────────

def mono_exp(t, A, tau, y0):
    return A * np.exp(-t / tau) + y0

def bi_exp(t, A1, tau1, A2, tau2, y0):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + y0

def tri_exp(t, A1, tau1, A2, tau2, A3, tau3, y0):
    return (A1 * np.exp(-t / tau1) +
            A2 * np.exp(-t / tau2) +
            A3 * np.exp(-t / tau3) + y0)

def amplitude_weighted_lifetime(amps, taus):
    """<τ> = Σ(Ai * τi) / Σ(Ai)"""
    amps = np.array(amps)
    taus = np.array(taus)
    return np.sum(amps * taus) / np.sum(amps)

def intensity_weighted_lifetime(amps, taus):
    """<τ>_int = Σ(Ai * τi²) / Σ(Ai * τi)"""
    amps = np.array(amps)
    taus = np.array(taus)
    return np.sum(amps * taus**2) / np.sum(amps * taus)


# ── Stern-Volmer models ──────────────────────────────────────────────────────

def stern_volmer_linear(C, I0, Ksv):
    """I0/I = 1 + Ksv*[Q]"""
    return I0 / (1 + Ksv * C)

def stern_volmer_modified(C, I0, Ksv, fa):
    """Modified SV for two-population (upward curve): I0/I = 1/(fa/(1+Ksv*C) + (1-fa))"""
    return I0 / (fa / (1 + Ksv * C) + (1 - fa))

def stern_volmer_combined(C, I0, Kd, Ka):
    """Combined static+dynamic: I0/I = (1+Kd*C)(1+Ka*C)"""
    return I0 / ((1 + Kd * C) * (1 + Ka * C))


# ── Temperature-dependent PL models ─────────────────────────────────────────

def varshni(T, E0, alpha, beta):
    """Varshni equation: E(T) = E0 - alpha*T^2/(T+beta)"""
    return E0 - alpha * T**2 / (T + beta)

def bose_einstein(T, E0, a_B, theta_B):
    """Bose-Einstein model: E(T) = E0 - a_B/(exp(theta_B/T) - 1)"""
    return E0 - a_B / (np.exp(theta_B / (T + 1e-10)) - 1)

def pl_intensity_thermal(T, I0, E_a, kB=8.617e-5):
    """PL quenching: I(T) = I0 / (1 + A*exp(-Ea/kBT))"""
    return I0  # placeholder; actual use requires A param

def pl_quenching(T, I0, A, E_a, kB=8.617e-5):
    """Thermal quenching: I(T) = I0 / (1 + A * exp(-Ea / kB*T))"""
    return I0 / (1 + A * np.exp(-E_a / (kB * T)))

def linewidth_phonon(T, Gamma0, sigma_inh, S, hbar_omega, gamma_LO, E_LO, kB=8.617e-5):
    """
    Linewidth: Γ(T) = Γ0_inh + σ_inh (not used here) 
    Simplified: Γ(T) = Γ0 + γ_LO / (exp(E_LO/kBT) - 1)
    """
    return Gamma0 + gamma_LO / (np.exp(E_LO / (kB * T + 1e-30)) - 1)


# ── Wavelength ↔ eV conversion (Jacobian) ───────────────────────────────────

def wavelength_to_eV(wavelength_nm):
    """λ (nm) → E (eV):  E = hc/λ"""
    hc = 1239.84193  # eV·nm
    return hc / wavelength_nm

def intensity_jacobian_transform(wavelength_nm, intensity):
    """
    Apply Jacobian transform when converting from wavelength to energy axis.
    I(E) = I(λ) * |dλ/dE| = I(λ) * λ² / hc
    """
    hc = 1239.84193
    jacobian = wavelength_nm**2 / hc
    energy_eV = wavelength_to_eV(wavelength_nm)
    intensity_eV = intensity * jacobian
    # Sort by increasing energy
    idx = np.argsort(energy_eV)
    return energy_eV[idx], intensity_eV[idx]


# ── Raman subtraction ────────────────────────────────────────────────────────

def raman_shift_to_wavelength(excitation_nm, raman_shift_cm1):
    """
    Convert Raman shift to emission wavelength.
    1/λ_emission = 1/λ_excitation - shift/10^7  (λ in nm, shift in cm^-1)
    """
    excitation_cm1 = 1e7 / excitation_nm
    emission_cm1 = excitation_cm1 - raman_shift_cm1
    return 1e7 / emission_cm1


# ── PLQY ────────────────────────────────────────────────────────────────────

def calculate_plqy(abs_sample, abs_ref, pl_sample, pl_ref,
                   abs_wl, pl_wl, excitation_nm,
                   pl_range=None):
    """
    PLQY by comparative method (Williams 1983):
    QY_sample = QY_ref * (PL_sample/PL_ref) * (Abs_ref/Abs_sample) * (n_sample/n_ref)^2
    
    Here n assumed equal (same solvent → ratio = 1).
    Absorption must be evaluated at excitation wavelength.
    PL integrals computed over pl_range (nm tuple).
    """
    def get_abs_at_excitation(wl_arr, abs_arr, exc_nm):
        return float(np.interp(exc_nm, wl_arr, abs_arr))

    def integrate_pl(wl, intensity, pl_range):
        if pl_range:
            mask = (wl >= pl_range[0]) & (wl <= pl_range[1])
            wl, intensity = wl[mask], intensity[mask]
        return np.trapz(intensity, wl)

    A_samp = get_abs_at_excitation(abs_wl, abs_sample, excitation_nm)
    A_ref  = get_abs_at_excitation(abs_wl, abs_ref,    excitation_nm)

    PL_samp = integrate_pl(pl_wl[0], pl_sample, pl_range)
    PL_ref  = integrate_pl(pl_wl[1], pl_ref,    pl_range)

    return A_samp, A_ref, PL_samp, PL_ref
