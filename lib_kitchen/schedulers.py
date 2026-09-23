# SPDX-License-Identifier: AGPL-3.0-or-later
# Kitchen Samplers. Third-party code and licenses: NOTICE.md

"""Schedule types from reForge, Neo and Extra Samplers, registered where the host WebUI lacks them.

Math follows the source. Exceptions: torch.pi -> math.pi; settings via _opt();
Bong Tangent start/middle/end are fractions of sigma_max (unchanged on flow, where sigma_max is 1);
Sinusoidal / Invcosinusoidal / React Cosinusoidal DynSF end on 0;
Align Your Steps last table entry is a ratio of sigma_max.

Not ported: Neo FlowMatchEulerDiscrete and Flux2.
"""

import math

import numpy as np
import torch

from lib_kitchen import host

DEFAULTS = {
    "sinusoidal_sf_factor": 3.5,
    "invcosinusoidal_sf_factor": 3.5,
    "react_cosinusoidal_dynsf_factor": 2.15,
    "cosine_sf_factor": 1.0,
    "cosexpblend_exp_decay": 0.9,
    "phi_power": 2.0,
    "laplace_mu": 0.0,
    "laplace_beta": 0.5,
    "ays_custom_sigmas": "[14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]",
    "karras_dynamic_rho": 7.0,
}


def _opt(name):
    return host.opt(name, DEFAULTS[name])


def _is_sdxl():
    """True if the loaded checkpoint is SDXL. Safe while the model is unloaded."""
    try:
        from modules import shared
    except Exception:
        return False
    return bool(getattr(getattr(shared, "sd_model", None), "is_sdxl", False))


pi, atan = math.pi, math.atan


def _loglinear_interp(t_steps, num_steps):
    """Log-linear interpolation of a decreasing array."""
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    return np.exp(new_ys)[::-1].copy()


# The Align Your Steps tables are sigma_max divided by fixed ratios, except the
# last entry, which reForge wrote as an absolute 0.029 (0.015 for the 32-step
# table): SD's own sigma_min. On any model whose sigma_max is not 14.61 - every
# flow model, where it is 1.0 - that made the schedule climb back up to 0.029
# before the final 0. Written as the same ratio of sigma_max, it is unchanged
# on SD 1.x / SDXL (sigma_max 14.6146) and correct everywhere else.
_SD_SIGMA_MAX = 14.614642
_AYS_LAST = 0.029 / _SD_SIGMA_MAX
_AYS32_LAST = 0.015 / _SD_SIGMA_MAX


def _fit(sigmas, n, device):
    """The tail every Align Your Steps variant shares."""
    if n != len(sigmas):
        sigmas = np.append(_loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)
    return torch.FloatTensor(sigmas).to(device)


def get_sigmas_sinusoidal_sf(n, sigma_min, sigma_max, device="cpu"):
    sf = _opt("sinusoidal_sf_factor")
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (1 - torch.sin(math.pi / 2 * x))) / sigma_max
    sigmas = sigmas ** sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_invcosinusoidal_sf(n, sigma_min, sigma_max, device="cpu"):
    sf = _opt("invcosinusoidal_sf_factor")
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (0.5 * (torch.cos(x * math.pi) + 1))) / sigma_max
    sigmas = sigmas ** sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_react_cosinusoidal_dynsf(n, sigma_min, sigma_max, device="cpu"):
    sf = _opt("react_cosinusoidal_dynsf_factor")
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (torch.cos(x * (math.pi / 2)))) / sigma_max
    sigmas = sigmas ** (sf * (n * x / n))
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def _ays_default(n, sigma_min, sigma_max, device):
    """reForge's plain Align Your Steps table - the fallback for Custom."""
    if _is_sdxl():
        sigmas = [sigma_max, sigma_max/2.314, sigma_max/3.875, sigma_max/6.701, sigma_max/10.89, sigma_max/16.954, sigma_max/26.333, sigma_max/38.46, sigma_max/62.457, sigma_max/129.336, sigma_max * _AYS_LAST]
    else:
        sigmas = [sigma_max, sigma_max/2.257, sigma_max/3.785, sigma_max/5.418, sigma_max/7.749, sigma_max/10.469, sigma_max/15.176, sigma_max/22.415, sigma_max/36.629, sigma_max/96.151, sigma_max * _AYS_LAST]
    return _fit(sigmas, n, device)


def get_sigmas_ays_custom(n, sigma_min, sigma_max, device="cpu"):
    try:
        sigmas_str = _opt("ays_custom_sigmas")
        sigmas_values = sigmas_str.strip("[]").split(",")
        sigmas = np.array([float(x.strip()) for x in sigmas_values])

        if n != len(sigmas):
            sigmas = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sigmas)), sigmas)
        sigmas = np.append(sigmas, [0.0])
        return torch.FloatTensor(sigmas).to(device)
    except Exception as e:
        print(f"[Kitchen Samplers] Error parsing custom sigmas: {e}")
        print("[Kitchen Samplers] Falling back to default AYS sigmas")
        return _ays_default(n, sigma_min, sigma_max, device)


def get_align_your_steps_sigmas_GITS(n, sigma_min, sigma_max, device):
    if _is_sdxl():
        sigmas = [sigma_max, sigma_max/3.087, sigma_max/5.693, sigma_max/9.558, sigma_max/14.807, sigma_max/22.415, sigma_max/34.964, sigma_max/54.533, sigma_max/81.648, sigma_max/115.078, sigma_max * _AYS_LAST]
    else:
        sigmas = [sigma_max, sigma_max/3.165, sigma_max/5.829, sigma_max/11.824, sigma_max/20.819, sigma_max/36.355, sigma_max/60.895, sigma_max/93.685, sigma_max/140.528, sigma_max/155.478, sigma_max * _AYS_LAST]
    return _fit(sigmas, n, device)


def ays_11_sigmas(n, sigma_min, sigma_max, device="cpu"):
    if _is_sdxl():
        sigmas = [sigma_max, sigma_max/2.314, sigma_max/3.875, sigma_max/6.701, sigma_max/10.89, sigma_max/16.954, sigma_max/26.333, sigma_max/38.46, sigma_max/62.457, sigma_max/129.336, sigma_max * _AYS_LAST]
    else:
        sigmas = [sigma_max, sigma_max/2.257, sigma_max/3.785, sigma_max/5.418, sigma_max/7.749, sigma_max/10.469, sigma_max/15.176, sigma_max/22.415, sigma_max/36.629, sigma_max/96.151, sigma_max * _AYS_LAST]
    return _fit(sigmas, n, device)


def ays_32_sigmas(n, sigma_min, sigma_max, device="cpu"):
    if _is_sdxl():
        sigmas = [sigma_max, sigma_max/1.310860875657935, sigma_max/1.718356235075352, sigma_max/2.252525958180810, sigma_max/2.688026675053433, sigma_max/3.174423075322040, sigma_max/3.748832539417044, sigma_max/4.463856789920335, sigma_max/5.326233593328242, sigma_max/6.355213820679800, sigma_max/7.477672611007930, sigma_max/8.745803592589411, sigma_max/10.228995682978878, sigma_max/11.864653584709637, sigma_max/13.685783347784952, sigma_max/15.786441921021279, sigma_max/18.202564111697559, sigma_max/20.980440157432400, sigma_max/24.182245076323649, sigma_max/27.652401723193991, sigma_max/31.246429590323925, sigma_max/35.307579021272943, sigma_max/40.308138967569972, sigma_max/47.132212095147923, sigma_max/55.111585405517003, sigma_max/65.460441760115945, sigma_max/82.786347724072168, sigma_max/104.698036963744033, sigma_max/138.041693219503482, sigma_max/264.794761864988552, sigma_max/507.935470821253285, sigma_max * _AYS32_LAST]
    else:
        sigmas = [sigma_max, sigma_max/1.300323183382763, sigma_max/1.690840379611262, sigma_max/2.198638945761486, sigma_max/2.622696705671493, sigma_max/3.098705619671305, sigma_max/3.661108232617473, sigma_max/4.152506637972936, sigma_max/4.662023756728857, sigma_max/5.234059175875519, sigma_max/5.874818853387466, sigma_max/6.593316416277412, sigma_max/7.399687115002039, sigma_max/8.213824943635682, sigma_max/9.050917900247738, sigma_max/9.973321246245751, sigma_max/11.115344803852001, sigma_max/12.529738625194212, sigma_max/14.124109921351757, sigma_max/15.959814856974724, sigma_max/18.099481611774999, sigma_max/20.526004748634670, sigma_max/23.506648288108032, sigma_max/27.541589307433523, sigma_max/32.269132736422456, sigma_max/38.982216080970984, sigma_max/53.219344283057142, sigma_max/72.656173487928834, sigma_max/103.609326413189740, sigma_max/218.693105563304210, sigma_max/461.605857767280530, sigma_max * _AYS32_LAST]
    return _fit(sigmas, n, device)


def cosine_scheduler(n, sigma_min, sigma_max, device="cpu"):
    sf = _opt("cosine_sf_factor")
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        for x in range(n):
            p = x / (n - 1)
            C = sigma_min + 0.5 * (sigma_max - sigma_min) * (1 - math.cos(math.pi * (1 - p ** 0.5)))
            sigmas[x] = C * sf
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def cosexpblend_scheduler(n, sigma_min, sigma_max, device="cpu"):
    decay = _opt("cosexpblend_exp_decay")
    sigmas = []
    if n == 1:
        sigmas.append(sigma_max ** 0.5)
    else:
        K = decay ** (1 / (n - 1))
        E = sigma_max
        for x in range(n):
            p = x / (n - 1)
            C = sigma_min + 0.5 * (sigma_max - sigma_min) * (1 - math.cos(math.pi * (1 - p ** 0.5)))
            sigmas.append(C + p * (E - C))
            E *= K
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)


def phi_scheduler(n, sigma_min, sigma_max, device="cpu"):
    power = _opt("phi_power")
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        phi = (1 + 5 ** 0.5) / 2
        for x in range(n):
            sigmas[x] = sigma_min + (sigma_max - sigma_min) * ((1 - x / (n - 1)) ** (phi ** power))
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_laplace(n, sigma_min, sigma_max, device="cpu"):
    mu = _opt("laplace_mu")
    beta = _opt("laplace_beta")
    epsilon = 1e-5  # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)  # noqa: E731
    lmb = mu - beta * torch.sign(0.5 - x) * torch.log(1 - 2 * torch.abs(0.5 - x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_karras_dynamic(n, sigma_min, sigma_max, device='cpu'):
    rho = _opt("karras_dynamic_rho")
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.zeros_like(ramp)
    for i in range(n):
        sigmas[i] = (max_inv_rho + ramp[i] * (min_inv_rho - max_inv_rho)) ** (math.cos(i*math.tau/n)*2+rho) 
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def linear_quadratic(n, sigma_min, sigma_max, device, *, threshold_noise=0.025):
    if n == 1:
        sigma_schedule = [1.0, 0.0]
    else:
        linear_steps = n // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * n
        quadratic_steps = n - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
        const = quadratic_coef * (linear_steps**2)
        quadratic_sigma_schedule = [quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, n)]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.FloatTensor(sigma_schedule).to(device) * sigma_max


def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
    smax = ((2 / pi) * atan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / pi) * atan(-slope * ((steps - 1) - pivot)) + 1) / 2

    srange = smax - smin
    sscale = start - end

    sigmas = [((((2 / pi) * atan(-slope * (x - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale + end for x in range(steps)]

    return sigmas


def _bong_tangent_neo(n, sigma_min, sigma_max, device, *, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2, pad=False):
    """https://github.com/ClownsharkBatwing/RES4LYF/blob/main/sigmas.py#L4076"""
    n += 2

    midpoint = int((n * pivot_1 + n * pivot_2) / 2)
    pivot_1 = int(n * pivot_1)
    pivot_2 = int(n * pivot_2)

    slope_1 = slope_1 / (n / 40)
    slope_2 = slope_2 / (n / 40)

    stage_2_len = n - midpoint
    stage_1_len = n - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)

    tan_sigmas_1 = tan_sigmas_1[:-1]
    if pad:
        tan_sigmas_2 = tan_sigmas_2 + [0]

    tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

    return tan_sigmas.to(device)


def bong_tangent_scheduler(n, sigma_min, sigma_max, device):
    """Neo Bong Tangent; 1.0 / 0.5 / 0.0 are fractions of sigma_max."""
    s = float(sigma_max)
    return _bong_tangent_neo(n, sigma_min, sigma_max, device, start=1.0 * s, middle=0.5 * s, end=0.0)


def beta_57(n, sigma_min, sigma_max, inner_model=None, device=None, **kwargs):
    """Beta schedule with alpha = 0.5, beta = 0.7. Restores the host Beta settings in `finally`."""
    from modules import shared
    from modules.sd_schedulers import beta_scheduler

    old_alpha = getattr(shared.opts, "beta_dist_alpha", 0.6)
    old_beta = getattr(shared.opts, "beta_dist_beta", 0.6)
    try:
        shared.opts.beta_dist_alpha = 0.5
        shared.opts.beta_dist_beta = 0.7
        return beta_scheduler(n, sigma_min, sigma_max, inner_model, device)
    finally:
        shared.opts.beta_dist_alpha = old_alpha
        shared.opts.beta_dist_beta = old_beta


def linear_log(
    n: int,
    sigma_min: float,
    sigma_max: float,
    inner_model,
    device: torch.device,
    eta: float = 0.1,
    nu: float = 2.0,
    sgm: bool = False,
    floor=False,
    final_step_full: bool = True,
) -> torch.Tensor:
    """
    Creates a log-linear (geometric) noise schedule as recommended in the paper.

    Args:
        n: Number of sampling steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        inner_model: Inner model
        device: Device to place the tensor on
        eta: Error parameter (default 0.1, as estimated in the paper for CIFAR-10)
        nu: Accuracy parameter for distance estimates (default 2.0)
        sgm: Whether to include the final sigma=0-step
        floor: Whether to floor sigma values at sigma_min
        final_step_full: Whether to take a full step (β=1) for the final iteration

    Returns:
        A tensor of sigma values in descending order with a geometric progression.
    """

    # TODO: Add adjustable eta/nu parameters for more flexibility

    # Calculate the maximum allowable beta based on the admissibility criteria
    # β*,N = c/(η+c) where c = 1 - ν^(-1/N)
    c = 1 - nu ** (-1 / n)
    beta_max = c / (eta + c)

    # Calculate the ratio that would give us exactly sigma_min from sigma_max in n steps
    exact_ratio = (sigma_min / sigma_max) ** (1 / (n - 1))

    # Use the smaller of the two to ensure admissibility
    ratio = max(1 - beta_max, exact_ratio)

    # Generate the geometric sequence
    sigs = [sigma_max]
    for i in range(1, n):
        next_sigma = sigs[-1] * ratio

        # For the final step, optionally set beta=1 (as recommended in the paper)
        if final_step_full and i == n - 1:
            next_sigma = sigma_min

        sigs.append(next_sigma)

    if not sgm:
        # Add final value of 0.0
        sigs.append(0.0)

    # Convert to tensor
    return torch.tensor(sigs)


# (name, label, function, need_inner_model, aliases, options it reads)
SCHEDULERS = [
    ("sinusoidal_sf", "Sinusoidal SF", get_sigmas_sinusoidal_sf, False, [], ["sinusoidal_sf_factor"]),
    ("invcosinusoidal_sf", "Invcosinusoidal SF", get_sigmas_invcosinusoidal_sf, False, [], ["invcosinusoidal_sf_factor"]),
    ("react_cosinusoidal_dynsf", "React Cosinusoidal DynSF", get_sigmas_react_cosinusoidal_dynsf, False, [], ["react_cosinusoidal_dynsf_factor"]),
    ("align_your_steps_custom", "Align Your Steps Custom", get_sigmas_ays_custom, False, [], ["ays_custom_sigmas"]),
    ("align_your_steps_11", "Align Your Steps 11", ays_11_sigmas, False, [], []),
    ("align_your_steps_32", "Align Your Steps 32", ays_32_sigmas, False, [], []),
    ("align_your_steps_GITS", "Align Your Steps GITS", get_align_your_steps_sigmas_GITS, False, [], []),
    ("cosine", "Cosine", cosine_scheduler, False, [], ["cosine_sf_factor"]),
    ("cosine-exponential blend", "Cosine-exponential Blend", cosexpblend_scheduler, False, [], ["cosexpblend_exp_decay"]),
    ("phi", "Phi", phi_scheduler, False, [], ["phi_power"]),
    ("laplace", "Laplace", get_sigmas_laplace, False, [], ["laplace_mu", "laplace_beta"]),
    ("karras dynamic", "Karras Dynamic", get_sigmas_karras_dynamic, False, [], ["karras_dynamic_rho"]),
    ("linear_quadratic", "Linear Quadratic", linear_quadratic, False, [], []),
    ("bong_tangent", "Bong Tangent", bong_tangent_scheduler, False, [], []),
    ("beta_57", "Beta 57", beta_57, True, ["Beta57", "beta57"], []),
    ("linear_log", "Linear Log", linear_log, True, [], []),
]
