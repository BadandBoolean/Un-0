"""Memristive Kuramoto dynamics: oscillators coupled through Yakopcic memristors.

Torch port of the Yakopcic memristor variant in the coupled-oscillator Ising
solver's `sanitychecks/yakopcicmemristor.py` (threshold voltage switching,
direction-aware Biolek window, passive decay — no Joglekar window), wired into
Un-0 the same way `src/n_oscillator.py` wires it into Kuramoto oscillators:
each unordered oscillator pair `(i, j)` is coupled through a single memristor
whose state `x[i, j]` lives inside the ODE next to the phases, with

    V_ij = V_osc * cos(θ_i - θ_j)                    (Hebbian pair voltage)
    J_ij = coupling_gain * (G_off + x_ij (G_on - G_off))   (unsigned coupling)

so nearly-synchronized pairs (cos Δθ close to 1, V above Vp) strengthen their
coupling, anti-phase pairs (V below -Vn) weaken it, and idle devices decay:
"use it or lose it". The voltage is even in Δθ, so `V_ij = V_ji` and a crossbar
initialized symmetric stays symmetric under the dynamics.

This replaces the static learned coupling `K` of `ConditionalKuramotoDynamics`
with physics; the learned quantities are the memristor *initial* states
(`x0_logits`), a global `coupling_gain`, and the usual frequencies and
conditioning block.

Memory note: the ODE state is `(batch, n + n_cond + n^2)` (the crossbar is
stored as a full matrix for simplicity; only the upper triangle is
independent), so this scales as O(batch * n^2) — intended for small-`n`
experiments, not the released sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn

from un0.model import (
    ConditionalImplicitKuramotoGenerator,
    Encoding,
    ReadoutTransform,
    Relativization,
    ResizeConvDecoder,
    Solver,
    _kuramoto_velocity,
)


@dataclass
class MemristorParams:
    """Yakopcic memristor parameters (defaults from the solver notebooks).

    Attributes:
        Vp: Positive threshold voltage for SET switching.
        Vn: Negative threshold voltage magnitude for RESET switching.
        Ap: Filament formation rate above the positive threshold.
        An: Filament dissolution rate below the negative threshold.
        p: Biolek window order; higher sharpens the boundary clamping.
        G_on: ON conductance bound (coupling at x = 1).
        G_off: OFF conductance bound (coupling at x = 0).
        eta: Timescale separation between memristor and oscillator
            (multiplies the switching term only, not the decay).
        lambda_decay: Passive decay rate toward the OFF state.
    """

    Vp: float = 0.16
    Vn: float = 0.15
    Ap: float = 4000.0
    An: float = 4000.0
    p: int = 2
    G_on: float = 2.0
    G_off: float = 0.001
    eta: float = 0.001
    lambda_decay: float = 0.04


# Knowm W+SDC (tungsten self-directed-channel), the most physically faithful
# set we can source. Provenance per field, so nothing here is mistaken for a
# fit it isn't:
#
#   Vp, Vn   MEASURED. Knowm datasheet Rev 3.2 (knowm.org/downloads/
#            Knowm_Memristors.pdf), W device: forward threshold typ 0.26 V,
#            reverse typ -0.11 V. Erase triggers far more easily than write;
#            that asymmetry is real and is *not* corrected for here.
#   G_on,    MEASURED (approx). LRS ~50 kOhm, HRS ~1 MOhm from the datasheet's
#   G_off    LRS/HRS vs write-compliance-current figure.
#   lambda   MEASURED. Datasheet Figure 7 (state retention): programmed low
#   _decay   resistance is flat over 30 min at 23/50/140 C, so no passive
#            decay on any timescale a training run cares about. (Figure only
#            covers LRS and 30 min.)
#   Ap, An   NOT an SDC fit — no published Yakopcic fit for this part is
#            available to us. 4000 is the generic value in Knowm's
#            memristor-models-4-all SPICE collection, whose thresholds
#            (0.65/0.56 and 0.16/0.15) match neither the SDC datasheet nor
#            each other, so those sets are not SDC-derived. Independent
#            sanity check: the datasheet's endurance conditions (1.5 Vpp,
#            500 Hz sine, 50 kOhm series) imply appreciable switching within
#            a ~0.8 ms above-threshold window, giving Ap ~1400-2900 — same
#            order, so 4000 is retained. Ap == An because we have no evidence
#            of *rate* asymmetry; the device's asymmetry lives in the
#            thresholds above.
#   eta      DERIVED, most uncertain. Timescale ratio, not a device property:
#            a 100 MHz ring oscillator with a ~100 us device write time puts
#            ~1.6e-5 of a full switch in one rollout (integration_time=1),
#            i.e. eta ~3e-8. Plausible oscillator/device pairings span
#            3e-9..3e-7, so sweep it rather than trusting one value. Note the
#            consequence: real devices barely move within a run, but the
#            drift accumulates meaningfully across a whole training.
#   p        Model artifact (Biolek window order), not physical.
#
# Drive with v_osc in the datasheet's 0.1-0.75 V operating window and above
# Vp; 0.35 V is the default (SET fires within +-42 deg of alignment, RESET
# beyond 108 deg).
KNOWM_W_SDC = MemristorParams(
    Vp=0.26,
    Vn=0.11,
    Ap=4000.0,
    An=4000.0,
    G_on=2e-5,
    G_off=1e-6,
    eta=3e-8,
    lambda_decay=0.0,
)


# Sandia TiN/TaOx/Ta/TiN ReRAM cell, from the device extraction in Yakopcic
# et al., "Memristor Model Optimization Based on Parameter Extraction From
# Device Characterization Data," IEEE TCAD 39(5), 2020 (Sec. IV, Eqs. 10-12;
# model avg. error ~6%). Provenance per field:
#
#   Vp, Vn   FITTED to the device: SET 0.91 V, RESET 1.425 V. Note the
#            asymmetry is the REVERSE of the Knowm W-SDC: here SET is the
#            easy direction. Under our symmetric pair voltage at v_osc=1.5,
#            SET outrates RESET ~21:1 — this device potentiates strongly;
#            not corrected for (that would be engineering, and is left to
#            the --memristor-an override, disclosed).
#   Ap, An   FITTED (72.5 / 21.0), already normalized to x in [0,1] for
#            exactly our g(V) = Ap(e^V - e^Vp) form. Caveat: fitted against
#            Yakopcic's native window, not our Biolek window, so switching
#            sharpness near the boundaries is approximate.
#   G_on,    FITTED (2.021e-3 / 7.490e-6 S). The paper's OFF branch is
#   G_off    gmin*sinh(2.62 V), which our linear conductance map flattens to
#            a constant G_off — an approximation we accept; the sinh shape
#            is dropped.
#   lambda   0 — TaOx ReRAM is nonvolatile.
#   eta      DERIVED BY US, not from the paper (the paper's eta is a +-1
#            direction convention, a different quantity from our timescale
#            ratio). 100 MHz oscillator with a ~100 ns TaOx write time gives
#            ~1e-4; plausible pairings span 1e-5..1e-3. The eta sweep of
#            2026-07-19 found final loss insensitive to eta over two orders
#            of magnitude (n=256, 50 epochs, Knowm preset).
#   p        Model artifact (Biolek window order), not physical.
#
# Drive with v_osc above Vp; the default 1.5 V clears both thresholds (SET
# fires within +-53 deg of alignment, RESET only beyond ~162 deg). A v_osc
# in (0.91, 1.425) would make the crossbar write-only (SET, never RESET).
TAOX_SANDIA = MemristorParams(
    Vp=0.91,
    Vn=1.425,
    Ap=72.5,
    An=21.0,
    G_on=2.021e-3,
    G_off=7.490e-6,
    eta=1e-4,
    lambda_decay=0.0,
)


def voltage_switching(V: Tensor, Ap: float, An: float, Vp: float, Vn: float) -> Tensor:
    """Voltage-dependent switching g(V): SET above Vp, RESET below -Vn, else 0."""
    zero = torch.zeros((), dtype=V.dtype, device=V.device)
    set_term = Ap * (torch.exp(V) - math.exp(Vp))
    reset_term = -An * (torch.exp(-V) - math.exp(Vn))
    g = torch.where(V > Vp, set_term, zero)
    return torch.where(V < -Vn, reset_term, g)


def memristor_dxdt(
    V: Tensor,
    x: Tensor,
    params: MemristorParams,
    eta_scale: Tensor | float = 1.0,
) -> Tensor:
    """State derivative dx/dt of an array of memristors under voltage V.

    Matches `YakopcicMemristor.memristor_dxdt` in the solver's sanitychecks:
    the drive combines switching and passive decay, and the Biolek window
    blocks motion toward the nearest boundary::

        drive = eta_scale * eta * g(V) - lambda_decay * x
        f = 1 - (x - stp)^(2p),  stp = 1 if drive < 0 else 0
        dx/dt = drive * f

    `eta_scale` (default 1) is a runtime multiplier on the switching term
    used for disturb annealing: shrinking it alongside the LR keeps the
    gradient-programming vs. read-disturb ratio constant over training.
    """
    g = voltage_switching(V, params.Ap, params.An, params.Vp, params.Vn)
    drive = eta_scale * params.eta * g - params.lambda_decay * x
    stp = (drive < 0.0).to(x.dtype)
    f = 1.0 - (x - stp) ** (2 * params.p)
    return drive * f


class MemristiveConditionalKuramotoDynamics(nn.Module):
    """Class-conditional Kuramoto dynamics with a memristive main-block coupling.

    Drop-in alternative to `ConditionalKuramotoDynamics` for
    `ConditionalImplicitKuramotoGenerator`: same `forward(state, t, drive)`
    contract and `K_drive` attribute, but `state` carries the memristor
    crossbar `(batch, n, n)` flattened after the phases, and the module
    exposes `sample_initial_state` so the generator starts the crossbar at
    the learned initial states instead of random phases.

    The conditioning block (`K_cond`, `K_drive`, standard parameterization)
    is unchanged from `ConditionalKuramotoDynamics`; memristors replace only
    the main-block coupling.
    """

    def __init__(
        self,
        *,
        n_oscillators: int,
        n_conditional_oscillators: int,
        num_classes: int,
        memristor_params: MemristorParams | None = None,
        v_osc: float = 0.18,
        init_freq_scale: float = 1.0,
        init_k_scale: float = 1.0,
        init_drive_scale: float = 1.0,
        crossbar: Literal["reflash", "persistent", "device"] = "reflash",
        n_chains: int = 128,
        signed_coupling: bool = False,
    ) -> None:
        """Initialize memristive conditional Kuramoto dynamics.

        Args:
            n_oscillators: Main oscillators `n`; the crossbar is `n x n`.
            n_conditional_oscillators: Conditioning driver oscillators.
            num_classes: Number of class labels for `K_drive`.
            memristor_params: Yakopcic device parameters (notebook defaults).
                With those defaults the state moves by only ~0.1 over Un-0's
                t in [0, 1]; raise `eta`/`lambda_decay` to make the coupling
                dynamics matter more within one generation rollout.
            v_osc: Oscillator voltage amplitude. The notebook default 0.18
                sits just above Vp = 0.16, so only pairs within ~27 degrees
                of phase alignment strengthen their memristor.
            init_freq_scale: Scale of the natural-frequency init.
            init_k_scale: Scale of the conditioning-block coupling init.
            init_drive_scale: Extra multiplier on the class-drive (`K_drive`)
                init only. Stronger steering makes which pairs align during a
                run more repeatable (less initial-phase luck), which is what
                makes Hebbian crossbar writes carry signal.
            crossbar: `"reflash"` (default) resets every run's crossbar to the
                learned `x0_logits` table. `"persistent"` deletes that table:
                conductances live in the circuit itself (`x_persist`, one
                independent chain per batch lane), start at the neutral 0.5
                once, and carry over from run to run — each training run's
                final states are committed back via `commit_crossbar`, so the
                coupling table is grown by device physics, not learned.
                `"device"` makes the devices themselves the weight store: one
                trainable crossbar (`x_weights`) that every run starts from
                and that nothing ever re-flashes — gradient steps program the
                conductances directly, and each training run's batch-mean
                physics drift is committed on top, so the stored weights are
                gradient programming plus accumulated device drift.
                Persistent/device modes are single-process only: DDP's buffer
                broadcast (or diverging per-rank commits) would corrupt them.
            n_chains: Number of persistent chains (use the training batch
                size). Ignored in reflash and device modes.
            signed_coupling: If True, use a differential device pair per
                oscillator pair (the src/adv design): crossbar entry (i, j)
                with i < j is the plus device (driven by +V, strengthens on
                in-phase history) and (j, i) is the minus device (driven by
                -V, strengthens on anti-phase history), with
                J = gain * (G(x_plus) - G(x_minus)). Coupling can then be
                negative (repulsive), so pairs can lock in anti-phase. Same
                state size; the two triangles stop being redundant copies.
        """
        super().__init__()
        if n_oscillators < 2 or n_conditional_oscillators < 1 or num_classes < 1:
            raise ValueError(
                "Need n_oscillators >= 2, n_conditional_oscillators >= 1, num_classes >= 1."
            )

        if crossbar not in ("reflash", "persistent", "device"):
            raise ValueError(
                f"crossbar must be 'reflash', 'persistent', or 'device', got {crossbar!r}."
            )

        self.n = int(n_oscillators)
        self.n_cond = int(n_conditional_oscillators)
        self.num_classes = int(num_classes)
        self.memristor_params = memristor_params or MemristorParams()
        self.v_osc = float(v_osc)
        self.crossbar = crossbar

        self.omega = nn.Parameter(init_freq_scale * torch.randn(1, self.n))
        self.omega_cond = nn.Parameter(init_freq_scale * torch.randn(1, self.n_cond))
        K_cond_init = init_k_scale * self.n_cond**-0.5 * torch.randn(self.n_cond, self.n_cond)
        K_cond_init.fill_diagonal_(0.0)
        self.K_cond = nn.Parameter(K_cond_init)
        self.K_drive = nn.Parameter(
            init_k_scale
            * float(init_drive_scale)
            * self.n_cond**-0.5
            * torch.randn(self.num_classes, self.n, self.n_cond)
        )

        if crossbar == "reflash":
            # Learned initial crossbar states x0 = sigmoid(sym(x0_logits)) in
            # (0, 1). Zero init gives x0 = 0.5 everywhere, the notebooks'
            # initial memristor state; logits are symmetrized so the crossbar
            # starts (and therefore stays) symmetric.
            self.x0_logits = nn.Parameter(torch.zeros(self.n, self.n))
        elif crossbar == "persistent":
            # Persistent chains at the neutral 0.5; a persistent buffer so
            # checkpoints carry the grown crossbar (it *is* the coupling
            # table in this mode).
            self.register_buffer("x_persist", torch.full((int(n_chains), self.n, self.n), 0.5))
        else:
            # Device mode: the conductances are the weight store. Trainable
            # (gradient pulses program the devices through the rollout) and
            # never re-flashed; commit_crossbar adds each training run's
            # batch-mean physics drift on top of the programmed values.
            self.x_weights = nn.Parameter(torch.full((self.n, self.n), 0.5))
        # J entries span [G_off, G_on] siemens; dividing the gain init by G_on
        # normalizes the max coupling to the n^-0.5 scale of the standard
        # learned K for any device preset (physical or dimensionless units).
        self.coupling_gain = nn.Parameter(
            torch.tensor(self.n**-0.5 / self.memristor_params.G_on)
        )
        self.register_buffer("_offdiag_mask", 1.0 - torch.eye(self.n), persistent=False)
        # Runtime multiplier on the switching rate (disturb annealing); the
        # training loop sets it to the current LR-schedule multiplier via
        # `eta_scale.fill_()`. Non-persistent: resume recomputes it.
        self.register_buffer("eta_scale", torch.ones(()), persistent=False)
        self.signed_coupling = bool(signed_coupling)
        if self.signed_coupling:
            # +1 on the upper triangle (plus devices), -1 on the lower
            # (minus devices, driven by the inverted pair voltage), 0 diag.
            sign = torch.triu(torch.ones(self.n, self.n), diagonal=1)
            sign = sign - sign.transpose(-1, -2)
            self.register_buffer("_pair_sign", sign, persistent=False)

    @property
    def state_dim(self) -> int:
        """Total state dimension: main + cond phases + flattened crossbar."""
        return self.n + self.n_cond + self.n * self.n

    def initial_crossbar(self) -> Tensor:
        """Learned symmetric initial memristor states, shaped `(n, n)` in (0, 1)."""
        if self.crossbar != "reflash":
            raise RuntimeError(
                "initial_crossbar() only exists in reflash mode; persistent "
                "chains live in the x_persist buffer."
            )
        if self.signed_coupling:
            # Upper and lower triangles are independent devices (plus/minus).
            return torch.sigmoid(self.x0_logits)
        logits = 0.5 * (self.x0_logits + self.x0_logits.transpose(-1, -2))
        return torch.sigmoid(logits)

    def sample_initial_state(
        self,
        num_samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Random phases in [-pi, pi) joined with the learned crossbar init."""
        phases = (
            torch.rand(
                num_samples,
                self.n + self.n_cond,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * (2.0 * torch.pi)
            - torch.pi
        )
        if self.crossbar == "persistent":
            # One chain per lane; oversized batches (evals) wrap around. The
            # buffer never requires grad, so runs read it as a constant.
            idx = torch.arange(num_samples, device=self.x_persist.device) % self.x_persist.shape[0]
            x0 = self.x_persist[idx].to(device=device, dtype=dtype).reshape(num_samples, -1)
        elif self.crossbar == "device":
            # The devices' current states, in range; NOT detached, so
            # gradients flow through the rollout back into the stored
            # weights (the programming path). Unsigned coupling keeps the
            # crossbar symmetric; signed treats the triangles as the
            # independent plus/minus devices.
            if self.signed_coupling:
                # clone() is a graph barrier: downstream ops must not save
                # the parameter tensor itself, because commit_crossbar
                # writes into it in-place before backward runs.
                x_dev = self.x_weights.clone()
            else:
                x_dev = 0.5 * (self.x_weights + self.x_weights.transpose(-1, -2))
            x0 = x_dev.clamp(0.0, 1.0).to(device=device, dtype=dtype)
            x0 = x0.reshape(1, -1).expand(num_samples, -1)
        else:
            x0 = self.initial_crossbar().to(device=device, dtype=dtype)
            x0 = x0.reshape(1, -1).expand(num_samples, -1)
        return torch.cat([phases, x0], dim=1)

    @torch.no_grad()
    def commit_crossbar(self, final_state: Tensor) -> None:
        """Write a training run's final crossbar states back into storage.

        No-op in reflash mode. The generator calls this after each training
        run; evaluation reads without writing.

        Persistent mode: each lane's final crossbar overwrites its own chain
        (lanes beyond the chain count are dropped — only oversized eval
        batches wrap, and those do not commit).

        Device mode: the batch-mean final crossbar overwrites the stored
        weights, so the physics drift of each training run stays written into
        the devices; the optimizer's gradient pulse then lands on top.
        """
        if self.crossbar == "persistent":
            batch = min(final_state.shape[0], self.x_persist.shape[0])
            x = final_state[:batch, self.n + self.n_cond :].reshape(batch, self.n, self.n)
            # Non-inplace clamp: `x` can alias `final_state`'s memory.
            self.x_persist[:batch] = x.clamp(0.0, 1.0)
        elif self.crossbar == "device":
            batch = final_state.shape[0]
            x = final_state[:, self.n + self.n_cond :].reshape(batch, self.n, self.n)
            self.x_weights.copy_(x.mean(dim=0).clamp(0.0, 1.0))

    def coupling_matrix(self, x: Tensor) -> Tensor:
        """Coupling from crossbar states shaped `(..., n, n)`.

        Unsigned: `J_ij = coupling_gain * (G_off + x_ij (G_on - G_off))`
        off-diagonal, zero diagonal — the conductance map of the solver's
        `n_osc_mem_rhs`.

        Signed (differential pair): for i < j,
        `J_ij = J_ji = coupling_gain * (G(x_ij) - G(x_ji))` — plus device
        minus minus device, symmetric and sign-indefinite.
        """
        p = self.memristor_params
        conductance = p.G_off + x * (p.G_on - p.G_off)
        if self.signed_coupling:
            upper = torch.triu(conductance - conductance.transpose(-1, -2), diagonal=1)
            return (upper + upper.transpose(-1, -2)) * self.coupling_gain
        return conductance * self._offdiag_mask * self.coupling_gain

    def forward(self, state: Tensor, _time: Tensor, drive: Tensor) -> Tensor:
        """Compute dstate/dt for concatenated (main, cond, crossbar) state.

        Args:
            state: `(batch, n + n_cond + n*n)` — main phases, cond phases,
                flattened memristor crossbar states.
            _time: Unused (required by torchdiffeq signature).
            drive: Per-sample drive matrix `(batch, n, n_cond)`.
        """
        batch = state.shape[0]
        theta_main = state[:, : self.n]
        theta_cond = state[:, self.n : self.n + self.n_cond]
        x = state[:, self.n + self.n_cond :].reshape(batch, self.n, self.n)

        J = self.coupling_matrix(x)

        sin_m = torch.sin(theta_main)
        cos_m = torch.cos(theta_main)
        # Per-sample coupling, so the matmul is batched (J is symmetric).
        weighted_sin = torch.einsum("bij,bj->bi", J, sin_m)
        weighted_cos = torch.einsum("bij,bj->bi", J, cos_m)
        main_vel = self.omega + cos_m * weighted_sin - sin_m * weighted_cos

        K_cond = self.K_cond - torch.diag_embed(self.K_cond.diagonal())
        cond_vel = _kuramoto_velocity(theta_cond, self.omega_cond, K_cond)

        sin_c = torch.sin(theta_cond)
        cos_c = torch.cos(theta_cond)
        drive_sin = torch.einsum("bnm,bm->bn", drive, sin_c)
        drive_cos = torch.einsum("bnm,bm->bn", drive, cos_c)
        main_vel = main_vel + cos_m * drive_sin - sin_m * drive_cos

        # V[b, i, j] = V_osc * cos(θ_i - θ_j), symmetric in (i, j).
        V = self.v_osc * (
            cos_m.unsqueeze(-1) * cos_m.unsqueeze(-2) + sin_m.unsqueeze(-1) * sin_m.unsqueeze(-2)
        )
        if self.signed_coupling:
            # Plus devices (upper triangle) see +V, minus devices (lower)
            # see -V, so anti-phase history strengthens the minus device.
            V = V * self._pair_sign
        dx = memristor_dxdt(V, x, self.memristor_params, self.eta_scale)

        return torch.cat([main_vel, cond_vel, dx.reshape(batch, -1)], dim=1)


def build_memristive_cifar10_model(
    *,
    n_oscillators: int = 1024,
    n_conditional_oscillators: int = 8,
    class_dropout_prob: float = 0.1,
    num_steps: int = 10,
    decoder_in_channels: int | None = None,
    relativization: Relativization = "ref_oscillator",
    encoding: Encoding = "sin_cos",
    solver: Solver = "euler",
    memristor_params: MemristorParams | None = None,
    v_osc: float = 0.18,
    init_drive_scale: float = 1.0,
    crossbar: Literal["reflash", "persistent", "device"] = "reflash",
    n_chains: int = 128,
    signed_coupling: bool = False,
) -> ConditionalImplicitKuramotoGenerator:
    """Build a CIFAR-10 generator with memristive main-block coupling.

    Mirrors `build_cifar10_model` but swaps in
    `MemristiveConditionalKuramotoDynamics`. The ODE state is
    O(batch * n_oscillators^2), so prefer small `n_oscillators` / batch sizes;
    the released-scale configs do not fit this dynamics.
    """
    feature_dim = (2 if encoding == "sin_cos" else 1) * int(n_oscillators)
    decoder_in_height = 4
    decoder_in_width = 4
    if decoder_in_channels is None:
        spatial_features = decoder_in_height * decoder_in_width
        if feature_dim % spatial_features != 0:
            raise ValueError(
                f"feature_dim={feature_dim} must be divisible by "
                f"{spatial_features} when decoder_in_channels is not set."
            )
        decoder_in_channels = feature_dim // spatial_features

    dynamics = MemristiveConditionalKuramotoDynamics(
        n_oscillators=int(n_oscillators),
        n_conditional_oscillators=int(n_conditional_oscillators),
        num_classes=10,
        memristor_params=memristor_params,
        v_osc=float(v_osc),
        init_drive_scale=float(init_drive_scale),
        crossbar=crossbar,
        n_chains=int(n_chains),
        signed_coupling=bool(signed_coupling),
    )
    dynamics = torch.compile(dynamics)
    readout = ReadoutTransform(
        encoding=encoding,
        relativization=relativization,
    )
    decoder = ResizeConvDecoder(
        feature_dim=feature_dim,
        output_dim=3 * 32 * 32,
        in_channels=int(decoder_in_channels),
        in_height=decoder_in_height,
        in_width=decoder_in_width,
        out_channels=3,
        num_upsamples=3,
        final_activation="tanh",
        init_output_gain=0.5,
    )
    decoder = torch.compile(decoder)
    return ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics,
        readout=readout,
        decoder=decoder,
        class_dropout_prob=float(class_dropout_prob),
        integration_time=1.0,
        num_steps=int(num_steps),
        solver=solver,
    )
