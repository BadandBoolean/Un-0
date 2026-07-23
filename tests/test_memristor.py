from __future__ import annotations

import math

import torch

from un0.memristor import (
    KNOWM_W_SDC,
    TAOX_SANDIA,
    MemristiveConditionalKuramotoDynamics,
    MemristorParams,
    memristor_dxdt,
    voltage_switching,
)
from un0.model import (
    ConditionalImplicitKuramotoGenerator,
    ReadoutTransform,
    ResizeConvDecoder,
)


def _tiny_dynamics(**overrides) -> MemristiveConditionalKuramotoDynamics:
    kwargs = {
        "n_oscillators": 4,
        "n_conditional_oscillators": 2,
        "num_classes": 3,
    }
    kwargs.update(overrides)
    return MemristiveConditionalKuramotoDynamics(**kwargs)


def test_voltage_switching_thresholds() -> None:
    """g(V) is zero inside (-Vn, Vp), positive above Vp, negative below -Vn."""
    params = MemristorParams()
    V = torch.tensor([0.0, params.Vp * 0.5, -params.Vn * 0.5, params.Vp + 0.3, -params.Vn - 0.3])
    g = voltage_switching(V, params.Ap, params.An, params.Vp, params.Vn)

    assert torch.all(g[:3] == 0.0)
    assert g[3] > 0.0
    assert g[4] < 0.0


def test_memristor_dxdt_matches_reference_formula() -> None:
    """dx/dt reproduces the sanitychecks YakopcicMemristor value pointwise."""
    params = MemristorParams()
    V, x = 0.18, 0.3
    g = params.Ap * (math.exp(V) - math.exp(params.Vp))
    drive = params.eta * g - params.lambda_decay * x
    stp = 1.0 if drive < 0 else 0.0
    expected = drive * (1.0 - (x - stp) ** (2 * params.p))

    dx = memristor_dxdt(torch.tensor(V), torch.tensor(x), params)

    assert torch.isclose(dx, torch.tensor(expected))


def test_memristor_dxdt_set_reset_and_decay() -> None:
    """Above-threshold voltage grows the state, reversed shrinks it, V=0 decays."""
    params = MemristorParams(eta=1.0)  # strong switching so SET beats decay
    x = torch.full((3,), 0.5)
    V = torch.tensor([params.Vp + 0.3, -params.Vn - 0.3, 0.0])
    dx = memristor_dxdt(V, x, params)

    assert dx[0] > 0.0
    assert dx[1] < 0.0
    assert dx[2] < 0.0  # passive decay toward OFF


def test_biolek_window_blocks_motion_toward_nearest_boundary() -> None:
    """At x=1 SET stalls but decay proceeds; at x=0 decay stalls."""
    params = MemristorParams(eta=1.0)
    set_v = torch.tensor(params.Vp + 0.3)

    at_top_set = memristor_dxdt(set_v, torch.tensor(1.0), params)
    at_top_decay = memristor_dxdt(torch.tensor(0.0), torch.tensor(1.0), params)
    at_bottom_decay = memristor_dxdt(torch.tensor(0.0), torch.tensor(0.0), params)

    assert torch.isclose(at_top_set, torch.tensor(0.0))  # f = 1 - 1^{2p} = 0
    assert at_top_decay < 0.0  # decrease allowed at the top boundary
    assert at_bottom_decay == 0.0  # decrease blocked at the bottom boundary


def test_coupling_matrix_is_conductance_with_zero_diagonal() -> None:
    """J = gain * (G_off + x (G_on - G_off)) off-diagonal, zero on-diagonal."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics()
    x = torch.rand(2, 4, 4)
    x = 0.5 * (x + x.transpose(-1, -2))

    J = dynamics.coupling_matrix(x)

    assert J.shape == (2, 4, 4)
    assert torch.allclose(J, J.transpose(-1, -2))
    assert torch.all(J.diagonal(dim1=-2, dim2=-1) == 0.0)
    p = dynamics.memristor_params
    expected_01 = dynamics.coupling_gain * (p.G_off + x[0, 0, 1] * (p.G_on - p.G_off))
    assert torch.isclose(J[0, 0, 1], expected_01)
    off_diag = J[~torch.eye(4, dtype=torch.bool).expand(2, 4, 4)]
    assert torch.all(off_diag > 0.0)  # unsigned (ferromagnetic) coupling


def test_sample_initial_state_layout() -> None:
    """Phases are uniform in [-pi, pi); the crossbar starts at the notebook 0.5."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics()
    state = dynamics.sample_initial_state(3, device=torch.device("cpu"), dtype=torch.float32)

    assert state.shape == (3, dynamics.state_dim)
    n_phases = dynamics.n + dynamics.n_cond
    phases = state[:, :n_phases]
    assert torch.all(phases >= -torch.pi)
    assert torch.all(phases < torch.pi)
    x0 = state[:, n_phases:].reshape(3, 4, 4)
    # Zero-initialized logits -> sigmoid = 0.5, matching the notebooks' init.
    assert torch.allclose(x0, torch.full_like(x0, 0.5))


def test_initial_crossbar_is_symmetric_for_arbitrary_logits() -> None:
    """Logit symmetrization keeps the learned initial crossbar symmetric."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics()
    with torch.no_grad():
        dynamics.x0_logits.copy_(torch.randn(4, 4))

    x0 = dynamics.initial_crossbar()

    assert torch.allclose(x0, x0.transpose(-1, -2))


def test_forward_preserves_crossbar_symmetry_and_hebbian_sign() -> None:
    """dx is symmetric, positive for in-phase pairs, negative for anti-phase."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics()
    batch = 1
    # Oscillators 0, 1 in phase; oscillator 2 anti-phase; 3 in quadrature.
    theta_main = torch.tensor([[0.0, 0.0, torch.pi, torch.pi / 2]])
    theta_cond = torch.zeros(batch, dynamics.n_cond)
    x = torch.full((batch, dynamics.n, dynamics.n), 0.5)
    state = torch.cat([theta_main, theta_cond, x.reshape(batch, -1)], dim=1)
    drive = torch.zeros(batch, dynamics.n, dynamics.n_cond)

    dstate = dynamics(state, torch.tensor(0.0), drive)

    assert dstate.shape == (batch, dynamics.state_dim)
    dx = dstate[:, dynamics.n + dynamics.n_cond :].reshape(batch, 4, 4)
    assert torch.allclose(dx, dx.transpose(-1, -2))
    # In-phase pair (0,1): V = V_osc > Vp with default eta the SET term is
    # small, so just check it beats the pure-decay quadrature pair (0,3).
    assert dx[0, 0, 1] > dx[0, 0, 3]
    # Anti-phase pair (0,2): V = -V_osc < -Vn -> RESET pushes x down.
    assert dx[0, 0, 2] < 0.0


def test_memristive_generator_end_to_end_gradients() -> None:
    """A tiny memristive generator produces flat images and grads for x0/gain."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics()
    readout = ReadoutTransform(encoding="sin_cos", relativization="ref_oscillator")
    decoder = ResizeConvDecoder(
        feature_dim=8,
        output_dim=16,
        in_channels=2,
        in_height=2,
        in_width=2,
        out_channels=1,
        num_upsamples=1,
    )
    model = ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics,
        readout=readout,
        decoder=decoder,
        num_steps=2,
    )

    class_id = torch.tensor([0, 1, 2])
    samples = model(class_id)
    loss = samples.square().mean()
    loss.backward()

    assert samples.shape == (3, 16)
    assert dynamics.x0_logits.grad is not None
    assert torch.any(dynamics.x0_logits.grad != 0.0)
    assert dynamics.coupling_gain.grad is not None
    assert dynamics.omega.grad is not None
    assert dynamics.K_drive.grad is not None


def _tiny_generator(dynamics: MemristiveConditionalKuramotoDynamics, **kwargs):
    readout = ReadoutTransform(encoding="sin_cos", relativization="ref_oscillator")
    decoder = ResizeConvDecoder(
        feature_dim=8,
        output_dim=16,
        in_channels=2,
        in_height=2,
        in_width=2,
        out_channels=1,
        num_upsamples=1,
    )
    return ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics,
        readout=readout,
        decoder=decoder,
        **kwargs,
    )


def test_persistent_mode_has_chains_instead_of_learned_table() -> None:
    """Persistent mode drops x0_logits and carries one 0.5-crossbar per chain."""
    dynamics = _tiny_dynamics(crossbar="persistent", n_chains=2)

    assert not hasattr(dynamics, "x0_logits")
    assert dynamics.x_persist.shape == (2, 4, 4)
    assert torch.all(dynamics.x_persist == 0.5)
    assert "x_persist" in dynamics.state_dict()  # checkpoints carry the chains


def test_persistent_sample_initial_state_reads_chains_and_wraps() -> None:
    """Lanes read their own chain; oversized batches wrap around the chains."""
    dynamics = _tiny_dynamics(crossbar="persistent", n_chains=2)
    with torch.no_grad():
        dynamics.x_persist[1] = 0.9

    state = dynamics.sample_initial_state(3, device=torch.device("cpu"), dtype=torch.float32)

    n_phases = dynamics.n + dynamics.n_cond
    x0 = state[:, n_phases:].reshape(3, 4, 4)
    assert torch.all(x0[0] == 0.5)  # chain 0
    assert torch.all(x0[1] == 0.9)  # chain 1
    assert torch.all(x0[2] == 0.5)  # wraps to chain 0
    assert not state.requires_grad


def test_commit_crossbar_writes_final_states_and_noops_in_reflash() -> None:
    """commit_crossbar stores each lane's final crossbar; reflash ignores it."""
    dynamics = _tiny_dynamics(crossbar="persistent", n_chains=2)
    final = torch.zeros(2, dynamics.state_dim)
    final[:, dynamics.n + dynamics.n_cond :] = 0.25
    final[0, dynamics.n + dynamics.n_cond] = 7.0  # out of range -> clamped

    dynamics.commit_crossbar(final)

    assert torch.all(dynamics.x_persist[1] == 0.25)
    assert dynamics.x_persist[0, 0, 0] == 1.0  # clamped, not 7.0
    assert final[0, dynamics.n + dynamics.n_cond] == 7.0  # input not mutated

    reflash = _tiny_dynamics()
    reflash.commit_crossbar(final)  # must not raise or create state


def test_generator_commits_chains_only_in_training_mode() -> None:
    """Training runs write back to the chains; eval sampling leaves them alone."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics(crossbar="persistent", n_chains=3)
    model = _tiny_generator(dynamics, num_steps=2)

    model.train()
    model(torch.tensor([0, 1, 2]))
    after_train = dynamics.x_persist.clone()
    # The run's decay/switching moved the crossbar off its 0.5 start.
    assert not torch.all(after_train == 0.5)

    model.eval()
    model.sample(torch.tensor([0, 1, 2]))
    assert torch.equal(dynamics.x_persist, after_train)


def test_persistent_generator_backprop_reaches_gain_without_x0() -> None:
    """With the stored table gone, gradients still reach the remaining knobs."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics(crossbar="persistent", n_chains=3)
    model = _tiny_generator(dynamics, num_steps=2)

    samples = model(torch.tensor([0, 1, 2]))
    samples.square().mean().backward()

    assert dynamics.coupling_gain.grad is not None
    assert dynamics.omega.grad is not None
    assert dynamics.K_drive.grad is not None


def test_eta_scale_anneals_the_switching_term_only() -> None:
    """eta_scale scales voltage-driven switching linearly and spares decay."""
    params = MemristorParams(eta=1.0, lambda_decay=0.0)
    V = torch.tensor(params.Vp + 0.3)
    x = torch.tensor(0.5)

    full = memristor_dxdt(V, x, params, eta_scale=1.0)
    half = memristor_dxdt(V, x, params, eta_scale=0.5)
    off = memristor_dxdt(V, x, params, eta_scale=0.0)

    assert torch.isclose(half, 0.5 * full)
    assert off == 0.0

    decay_params = MemristorParams(eta=1.0, lambda_decay=0.1)
    decay_only = memristor_dxdt(torch.tensor(0.0), x, decay_params, eta_scale=0.0)
    assert decay_only < 0.0  # decay unaffected by the anneal


def test_dynamics_eta_scale_buffer_freezes_crossbar_at_zero() -> None:
    """With eta_scale=0 and no decay, the crossbar state is inert in forward."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics(memristor_params=MemristorParams(lambda_decay=0.0))
    dynamics.eta_scale.fill_(0.0)
    state = torch.cat(
        [torch.zeros(1, dynamics.n + dynamics.n_cond), torch.full((1, dynamics.n**2), 0.5)],
        dim=1,
    )
    dstate = dynamics(state, torch.tensor(0.0), torch.zeros(1, dynamics.n, dynamics.n_cond))

    dx = dstate[:, dynamics.n + dynamics.n_cond :]
    assert torch.all(dx == 0.0)


def test_device_mode_stores_weights_in_one_trainable_crossbar() -> None:
    """Device mode has a single trainable crossbar and no table or chains."""
    dynamics = _tiny_dynamics(crossbar="device")

    assert not hasattr(dynamics, "x0_logits")
    assert not hasattr(dynamics, "x_persist")
    assert dynamics.x_weights.shape == (4, 4)
    assert dynamics.x_weights.requires_grad
    assert torch.all(dynamics.x_weights == 0.5)


def test_device_mode_gradients_program_the_weights() -> None:
    """Backprop through the rollout reaches the device weights directly."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics(crossbar="device")
    model = _tiny_generator(dynamics, num_steps=2)

    samples = model(torch.tensor([0, 1, 2]))
    samples.square().mean().backward()

    assert dynamics.x_weights.grad is not None
    assert torch.any(dynamics.x_weights.grad != 0.0)


def test_device_mode_training_run_drift_stays_written() -> None:
    """A training run's batch-mean drift persists in the weights; eval reads only."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics(crossbar="device")
    model = _tiny_generator(dynamics, num_steps=2)

    model.train()
    model(torch.tensor([0, 1, 2]))
    after_train = dynamics.x_weights.detach().clone()
    assert not torch.all(after_train == 0.5)  # drift committed

    model.eval()
    model.sample(torch.tensor([0, 1, 2]))
    assert torch.equal(dynamics.x_weights.detach(), after_train)


def test_init_drive_scale_multiplies_only_the_class_drive() -> None:
    """init_drive_scale scales K_drive init 10x but leaves K_cond untouched."""
    torch.manual_seed(0)
    base = _tiny_dynamics()
    torch.manual_seed(0)
    scaled = _tiny_dynamics(init_drive_scale=10.0)

    assert torch.allclose(scaled.K_drive, 10.0 * base.K_drive)
    assert torch.allclose(scaled.K_cond, base.K_cond)


def test_signed_coupling_matrix_is_symmetric_and_sign_indefinite() -> None:
    """Signed J is the differential-pair difference, mirrored, zero diagonal."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics(signed_coupling=True)
    x = torch.rand(1, 4, 4)

    J = dynamics.coupling_matrix(x)

    assert torch.allclose(J, J.transpose(-1, -2))
    assert torch.all(J.diagonal(dim1=-2, dim2=-1) == 0.0)
    p = dynamics.memristor_params
    expected_01 = dynamics.coupling_gain * (x[0, 0, 1] - x[0, 1, 0]) * (p.G_on - p.G_off)
    assert torch.isclose(J[0, 0, 1], expected_01)
    x_neg = torch.zeros(1, 4, 4)
    x_neg[0, 1, 0] = 1.0  # minus device fully ON -> repulsive pair (0,1)
    assert dynamics.coupling_matrix(x_neg)[0, 0, 1] < 0.0


def test_signed_coupling_hebbian_writes_both_directions() -> None:
    """In-phase pairs grow the plus device; anti-phase pairs grow the minus."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics(
        signed_coupling=True,
        memristor_params=MemristorParams(eta=1.0, lambda_decay=0.0),
    )
    # Oscillators 0, 1 in phase; oscillator 2 anti-phase to both.
    theta_main = torch.tensor([[0.0, 0.0, torch.pi, torch.pi / 2]])
    theta_cond = torch.zeros(1, dynamics.n_cond)
    x = torch.full((1, dynamics.n, dynamics.n), 0.5)
    state = torch.cat([theta_main, theta_cond, x.reshape(1, -1)], dim=1)
    drive = torch.zeros(1, dynamics.n, dynamics.n_cond)

    dx = dynamics(state, torch.tensor(0.0), drive)[:, dynamics.n + dynamics.n_cond :]
    dx = dx.reshape(1, 4, 4)

    assert dx[0, 0, 1] > 0.0  # in-phase: plus device strengthens
    assert dx[0, 1, 0] < 0.0  # ... and its minus device weakens
    assert dx[0, 0, 2] < 0.0  # anti-phase: plus device weakens
    assert dx[0, 2, 0] > 0.0  # ... and its minus device strengthens


def test_signed_reflash_keeps_independent_triangles() -> None:
    """Signed mode does not symmetrize x0_logits (triangles are two devices)."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics(signed_coupling=True)
    with torch.no_grad():
        dynamics.x0_logits.copy_(torch.randn(4, 4))

    x0 = dynamics.initial_crossbar()

    assert not torch.allclose(x0, x0.transpose(-1, -2))


def test_signed_device_generator_end_to_end_gradients() -> None:
    """Signed device-mode generator trains: grads reach the device weights."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics(signed_coupling=True, crossbar="device")
    model = _tiny_generator(dynamics, num_steps=2)

    samples = model(torch.tensor([0, 1, 2]))
    samples.square().mean().backward()

    assert samples.shape == (3, 16)
    assert dynamics.x_weights.grad is not None
    assert torch.any(dynamics.x_weights.grad != 0.0)


def test_knowm_preset_matches_datasheet_values() -> None:
    """The preset carries the measured datasheet physics, unengineered."""
    assert KNOWM_W_SDC.Vp == 0.26  # measured forward threshold
    assert KNOWM_W_SDC.Vn == 0.11  # measured reverse threshold
    assert KNOWM_W_SDC.G_on == 2e-5  # LRS ~50 kOhm
    assert KNOWM_W_SDC.G_off == 1e-6  # HRS ~1 MOhm
    assert KNOWM_W_SDC.lambda_decay == 0.0  # Figure 7: flat retention
    # No published rate asymmetry for this device: An must not be engineered
    # away from Ap in the preset itself.
    assert KNOWM_W_SDC.An == KNOWM_W_SDC.Ap == 4000.0


def test_taox_preset_matches_paper_extraction() -> None:
    """TaOx preset: fitted thresholds/rates; SET-dominant at v_osc=1.5."""
    assert TAOX_SANDIA.Vp == 0.91
    assert TAOX_SANDIA.Vn == 1.425
    assert TAOX_SANDIA.Ap == 72.5
    assert TAOX_SANDIA.An == 21.0
    assert TAOX_SANDIA.G_on == 2.021e-3
    assert TAOX_SANDIA.lambda_decay == 0.0  # nonvolatile TaOx

    p = TAOX_SANDIA
    g = voltage_switching(torch.tensor([1.5, -1.5]), p.Ap, p.An, p.Vp, p.Vn)
    assert g[0] > 0.0  # SET fires above 0.91
    assert g[1] < 0.0  # RESET fires below -1.425
    assert float(g[0]) > 10 * abs(float(g[1]))  # SET-dominant, ~21:1

    # In the write-only band (0.91, 1.425): SET fires, RESET cannot.
    g2 = voltage_switching(torch.tensor([1.2, -1.2]), p.Ap, p.An, p.Vp, p.Vn)
    assert g2[0] > 0.0
    assert g2[1] == 0.0


def test_knowm_preset_thresholds_and_gain_normalization() -> None:
    """Knowm W+SDC preset: datasheet thresholds; gain init absorbs G_on units."""
    assert KNOWM_W_SDC.Vp == 0.26
    assert KNOWM_W_SDC.Vn == 0.11
    assert KNOWM_W_SDC.G_on == 2e-5

    dynamics = _tiny_dynamics(memristor_params=KNOWM_W_SDC)
    p = dynamics.memristor_params
    # Max coupling J = gain * G_on should sit at the n^-0.5 scale regardless
    # of whether conductances are physical siemens or dimensionless.
    max_j = float(dynamics.coupling_gain) * p.G_on
    assert abs(max_j - dynamics.n**-0.5) < 1e-6

    # At the preset's v_osc = 0.35: aligned pairs SET, anti-phase pairs RESET,
    # and RESET is the stronger of the two (the device's real asymmetry).
    g = voltage_switching(torch.tensor([0.35, -0.35]), p.Ap, p.An, p.Vp, p.Vn)
    assert g[0] > 0.0
    assert g[1] < 0.0
    assert abs(float(g[1])) > float(g[0])


def test_memristive_generator_num_steps_zero_decodes_initial_state() -> None:
    """num_steps=0 skips the ODE and decodes the learned initial phases."""
    torch.manual_seed(0)
    dynamics = _tiny_dynamics()
    readout = ReadoutTransform(encoding="sin_cos", relativization="ref_oscillator")
    decoder = ResizeConvDecoder(
        feature_dim=8,
        output_dim=16,
        in_channels=2,
        in_height=2,
        in_width=2,
        out_channels=1,
        num_upsamples=1,
    )
    model = ConditionalImplicitKuramotoGenerator(
        dynamics=dynamics,
        readout=readout,
        decoder=decoder,
        num_steps=0,
    )

    samples = model(torch.tensor([0, 1]))

    assert samples.shape == (2, 16)
