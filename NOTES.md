# Notes

Measurements taken while building Kitchen Samplers, and the reasons for the main choices. Numbers were measured against:

- Forge (lllyasviel) at `dfdcbab`
- reForge at `739b2e1`
- Forge Classic (Neo) at `76586f6`

## 1. One entry per algorithm

A sampler is only left out when it was **measured** to be the same algorithm as one the WebUI already ships. A matching name alone is not enough.

| Entry here | Is the same as | Measured |
|---|---|---|
| every `lib_kitchen/canon` copy | the reForge / Neo original it was generated from | max diff 0.0 (Neo's Euler CFG++ 2e-6); SD and flow |
| RES4LYF res_2m ODE | Res Multistep | 2e-6 SD, 7e-7 flow |
| RES4LYF res_2s_stable ODE | EXP Heun 2 x0 (reForge) | 6e-7 on SD; see below for flow |
| Extra Samplers' Euler Dy / SMEA Dy / Negative / Dy Negative | reForge's `sd_samplers_kdiffusion_smea` | identical code |
| Extra Samplers' Gradient Estimation, Res Multistep ×4 | the ComfyUI versions shipped by reForge / Neo | copies of the same ComfyUI source (compared by reading, not re-measured) |

**EXP Heun 2 x0 on flow models.** On flow models reForge's version steps in the model's own log-SNR, and RES4LYF's steps in −log σ. Both are second-order and equally accurate: the error against a converged solution is 6.04e-2 here and 5.86e-2 for reForge's. They differ by at most 7e-3, so a PNG from reForge reproduces closely here, but not bit for bit.

**Not merged: RES4LYF res_2m and Res Multistep Ancestral.** Without noise they are identical. Given the same noise they differ by up to 3.0, because they carry the multistep history differently once noise is added. So `RES4LYF res_2m` stays an entry of its own.

Names are compared case- and punctuation-insensitively, so "SA-Solver" matches "SA Solver". That avoids registering a second copy of a sampler reForge already lists under different spelling in `forge_alter_samplers`.

## 2. Differences between the WebUIs

`lib_kitchen/host.py` holds everything that differs between the WebUIs.

- **Model sampling.** reForge exposes it through `model_patcher.get_model_object("model_sampling")`, with flow models detected by the CONST class. Forge and Neo use `inner_model.predictor`, with `prediction_type == "const"`.
- **Seeded noise.** Each WebUI installs its `TorchHijack` on its own k-diffusion module. The copies here draw noise through that hijack. Same seed gives the same image on all three.
- **CFG++ hooks.** reForge and Neo pass `extra_args["model_options"]` down to the CFG function. Forge ignores it. On Forge, the post-CFG hook is added to the UNet's own `model_options` for the length of the call and then removed.
- **`model.last_noise_uncond`.** This exists in reForge and Forge but not in Neo. Extra Samplers' Euler Multipass CFG++ used it and failed on Neo. It now reads the uncond prediction from a post-CFG hook.
- **reForge's alter samplers** use a hard-coded scheduler table and do not go through the WebUI's scheduler list. See section 3.

## 3. Fixes

| What | Before | After | Test |
|---|---|---|---|
| reForge: its own samplers (DEIS, SEEDS, Gradient Estimation, …) with a schedule type they did not know | silently Normal | Beta 57, Linear Log, Bong Tangent, Linear Quadratic are used; reForge's own table untouched | `test_sampling_in_host` (reForge) |
| Sinusoidal SF, Invcosinusoidal SF, React Cosinusoidal DynSF | n sigmas, last one about 5e-9 to 2e-5 instead of 0; one step fewer than asked; DEIS and DDPM gave NaN | end on 0 | `test_sampling_in_host` |
| Align Your Steps 11 / 32 / GITS (copies here) | last entry an absolute 0.029 / 0.015; on flow models the schedule went 0.016 → 0.029 → 0 | same ratio of sigma_max as the other entries; unchanged on SD 1.x / SDXL | `test_sampling_in_host` |
| Heun Ancestral (Extra Samplers) | built its own Brownian noise from `extra_args["seed"]`, which no WebUI sets | uses the WebUI's seeded noise sampler | `test_sampling_in_host` |
| Euler Multipass CFG++ (Extra Samplers) | AttributeError on Neo | works on all three | `test_sampling_in_host` |
| SSPRK3 (Extra Samplers) | `sigmas` keyword-only: `func(model, x, sigmas)` raised TypeError. The WebUIs pass it by keyword, so they were not affected. | positional or keyword | `test_catalogue` |
| RES4LYF abnorsett_3m / 4m | 4 / 9 start-up model calls not counted | counted in total_steps | `test_catalogue`, `test_sampling_in_host` |
| Adaptive Progressive, Langevin Euler (Extra Samplers) | settings from an always-visible accordion | Settings → Kitchen Samplers; values in the infotext; old keys and PNG info still work | `test_in_host` |

At the same step count, a missing final 0 does not change image quality for Euler, DPM++ 2M, Heun or RES4LYF res_3s. It does break samplers that expect the schedule to end at 0. Without it, DEIS and DDPM produced NaN on SD and flow models alike.

## 4. Kept as the source has them

These are faithful copies. Their behaviour is the source's design, or at least not clearly a slip, so they were left alone.

- **Cosine-exponential Blend** does not decrease. With the default decay of 0.9 it dips to about 0.56·σmax, climbs back to 0.9·σmax, and then jumps to 0. For example, on SD at 8 steps: 14.6, 10.7, 8.7, 8.2, 8.8, 9.9, 11.5, 13.2, 0.
- **Karras Dynamic** starts above σmax: 31.4 against 14.6 on SD. Its rho oscillates with the step, `cos(i·2π/n)·2 + rho`, and the first step uses rho + 2.
- **Laplace** on flow models repeats σ = 1.0 for the first few steps, clamped at σmax. Those steps spend model calls without moving.
- **Align Your Steps Custom** takes absolute sigmas and is meant for SD. On a flow model, enter a list that starts at 1.0.
- **DDPM with React Cosinusoidal DynSF** gives NaN. The penultimate sigma of about 2e-5 makes DDPM's alpha round to 1 in float32. reForge's own DDPM does the same with reForge's own React Cosinusoidal DynSF.
- WebUI schedules this extension does not touch:
  - **KL Optimal** ends at σmin rather than 0 (Forge, reForge and Neo).
  - Neo's and reForge's own **Align Your Steps** has the absolute-0.029 ending on flow models.

## 5. Flow models (Flux, SD3, Wan…)

Every sampler here runs on flow models on all three WebUIs. The tests check finite output, reproducible seeds and correct model-call counts. Accuracy against Euler at the same number of model calls (`test_catalogue.py`, 12 steps, CFG 3):

- **On SD models**, most of RES4LYF's 3- to 6-stage methods are 20–100× more accurate than Euler. A few are only 1–1.5× better: res_4s krogstad_alt, strehmel_weiner_alt, friedli, minchev, munthe-kaas, and res_5s hochbruck-ostermann. DEIS, IPNDM and IPNDM_V are 5–11× more accurate.
- **On flow models the gain mostly disappears.** With a flow schedule the final step (σ → 0, `x = denoised`) dominates the error. High-order methods come out roughly level with Euler: res_3s is 0.82× Euler's error, and res_6s is 1.67× (worse). The step to σ = 0 has to be a plain `x = denoised`: evaluating a final stage at σ ≈ 0 blew it up by about 1e3. On flow models, spend the model calls on more steps rather than more stages.
- The update rule `dx/dσ = (x − D)/σ` holds for rectified flow as well, so every VE-form integrator here is valid on flow models. No flow-specific adapter is needed.

## 6. Numbers per WebUI

| | Forge | reForge | Neo |
|---|---|---|---|
| samplers added / already present | 84 / 6 | 62 / 28 | 84 / 6 |
| schedule types added | 13 | 4 | 14 |
| settings added | 34 | 5 | 41 |
| `test_in_host` | 34 pass | 34 pass | 34 pass |
| `test_sampling_in_host` | 198 pass | 151 pass | 198 pass |

reForge has fewer sampling checks because 28 of the entries are its own samplers, which are not re-tested.
