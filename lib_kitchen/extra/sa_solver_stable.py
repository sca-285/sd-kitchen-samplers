# SPDX-License-Identifier: Apache-2.0
# SA-Solver Stable: eddyhhlure1Eddy / kabachuha, via MisterChief95 Extra Samplers.
# See NOTICE.md.

import torch
from tqdm.auto import trange

from lib_kitchen.canon.common import to_d
from lib_kitchen.extra.utils import sampler_metadata


# Original sampler by https://github.com/eddyhhlure1Eddy/ode-ComfyUI-WanVideoWrapper
# Adapted into ComfyUI by https://github.com/kabachuha/ComfyUI-SA-ODE-Stable-Sampler
# Apache 2.0 License
@torch.no_grad()
@sampler_metadata("SA-Solver Stable", {"uses_ensd": True, "scheduler": "karras"})
def sample_sa_solver_stable(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    solver_order=3,
    use_adaptive_order=True,
    use_velocity_smoothing=True,
    convergence_threshold=0.15,
    smoothing_factor=0.8,
):
    """Deterministic SA-ODE stable-convergence variant."""
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    velocity_buffer = []
    smoothed_velocity = None
    num_inference_steps = len(sigmas)

    def get_adaptive_order(sigma):
        if not use_adaptive_order:
            return solver_order

        if num_inference_steps <= 8:
            return min(2, solver_order)

        if sigma > 0.7:
            return min(2, solver_order)
        if sigma > convergence_threshold:
            return solver_order
        return max(1, solver_order - 1)

    def compute_multistep_velocity(order):
        if not velocity_buffer:
            raise RuntimeError("velocity_buffer is empty")

        order = min(order, len(velocity_buffer))
        if order >= 3 and len(velocity_buffer) >= 3:
            return (23 / 12) * velocity_buffer[-1] - (16 / 12) * velocity_buffer[-2] + (5 / 12) * velocity_buffer[-3]
        if order >= 2 and len(velocity_buffer) >= 2:
            return 1.5 * velocity_buffer[-1] - 0.5 * velocity_buffer[-2]
        if len(velocity_buffer) >= 1:
            return velocity_buffer[-1]
        raise RuntimeError("No velocity data available")

    def apply_velocity_smoothing(velocity, sigma):
        nonlocal smoothed_velocity
        if not use_velocity_smoothing:
            return velocity

        if num_inference_steps <= 8:
            return velocity

        if sigma < convergence_threshold:
            if smoothed_velocity is None:
                smoothed_velocity = velocity
            else:
                alpha = smoothing_factor
                smoothed_velocity = alpha * smoothed_velocity + (1 - alpha) * velocity
            return smoothed_velocity

        smoothed_velocity = velocity
        return velocity

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(x, sigma * s_in, **extra_args)
        d = to_d(x, sigma, denoised)

        velocity_buffer.append(d)
        while len(velocity_buffer) > solver_order + 1:
            velocity_buffer.pop(0)

        current_order = get_adaptive_order(sigma.item())
        velocity = compute_multistep_velocity(current_order) if len(velocity_buffer) >= 2 else d
        velocity = apply_velocity_smoothing(velocity, sigma.item())

        dt = sigma_next - sigma
        if num_inference_steps > 8 and sigma.item() < convergence_threshold:
            damping = 0.5 + 0.5 * (sigma.item() / convergence_threshold)
            dt = dt * damping

        if sigma_next == 0:
            x = denoised
        else:
            x = x + velocity * dt

        if num_inference_steps > 8 and sigma.item() < 0.05 and len(velocity_buffer) >= 3:
            avg_velocity = sum(velocity_buffer[-3:]) / 3
            stabilized = x + avg_velocity * dt
            blend_factor = sigma.item() / 0.05
            x = blend_factor * x + (1 - blend_factor) * stabilized

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

    return x
