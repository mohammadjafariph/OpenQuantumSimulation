import numpy as np
import pytest

import openquantumsim as oqs


def test_state_fidelity_conventions() -> None:
    up = np.array([1.0, 0.0], dtype=np.complex128)
    down = np.array([0.0, 1.0], dtype=np.complex128)
    plus = (up + down) / np.sqrt(2.0)
    mixed = 0.5 * np.eye(2, dtype=np.complex128)

    assert oqs.fidelity(up, up) == pytest.approx(1.0)
    assert oqs.fidelity(up, down) == pytest.approx(0.0)
    assert oqs.fidelity(up, plus) == pytest.approx(0.5)
    assert oqs.fidelity(up, plus, squared=False) == pytest.approx(np.sqrt(0.5))
    assert oqs.fidelity(mixed, up) == pytest.approx(0.5)
    assert oqs.infidelity(up, plus) == pytest.approx(0.5)


def test_state_distances_and_bures_quantities() -> None:
    up = oqs.ket2dm(np.array([1.0, 0.0], dtype=np.complex128))
    down = oqs.ket2dm(np.array([0.0, 1.0], dtype=np.complex128))

    norm_input = np.diag([1.0, -2.0]).astype(np.complex128)
    assert oqs.trace_norm(norm_input) == pytest.approx(3.0)
    assert oqs.trace_distance(up, up) == pytest.approx(0.0)
    assert oqs.trace_distance(up, down) == pytest.approx(1.0)
    assert oqs.hilbert_schmidt_distance(up, down) == pytest.approx(np.sqrt(2.0))
    assert oqs.bures_angle(up, down) == pytest.approx(np.pi / 2)
    assert oqs.bures_distance(up, down) == pytest.approx(np.sqrt(2.0))


def test_entropies_purity_populations_and_coherence() -> None:
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho_plus = oqs.ket2dm(plus)
    mixed = 0.5 * np.eye(2, dtype=np.complex128)

    assert oqs.purity(plus) == pytest.approx(1.0)
    assert oqs.purity(mixed) == pytest.approx(0.5)
    assert oqs.linear_entropy(mixed) == pytest.approx(0.5)
    assert oqs.linear_entropy(mixed, normalized=True) == pytest.approx(1.0)
    assert oqs.renyi_entropy(mixed, alpha=2.0) == pytest.approx(1.0)
    assert oqs.participation_ratio(mixed) == pytest.approx(2.0)
    assert np.allclose(oqs.populations(rho_plus), [0.5, 0.5])
    assert oqs.l1_coherence(rho_plus) == pytest.approx(1.0)


def test_state_validation_normalization_and_bloch_vector() -> None:
    up = np.array([2.0, 0.0], dtype=np.complex128)
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho_plus = oqs.ket2dm(plus)

    assert np.allclose(oqs.normalize_state(up), [1.0, 0.0])
    assert oqs.is_hermitian(rho_plus)
    assert oqs.is_density_matrix(rho_plus)
    assert not oqs.is_density_matrix(2.0 * rho_plus)
    up_density = oqs.ket2dm(np.array([1.0, 0.0], dtype=np.complex128))
    assert np.allclose(oqs.bloch_vector(up_density), [0.0, 0.0, 1.0])
    assert np.allclose(oqs.bloch_vector(rho_plus), [1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="two-level"):
        oqs.bloch_vector(np.eye(3, dtype=np.complex128) / 3.0)


def test_state_metric_callback_builders() -> None:
    up = np.array([1.0, 0.0], dtype=np.complex128)
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)

    metrics = oqs.state_metrics(
        purity=True,
        entropy=True,
        linear_entropy=True,
        participation_ratio=True,
        population_indices=[0, 1],
        l1_coherence=True,
        bloch_vector=True,
        fidelity_to=up,
        trace_distance_to=up,
    )
    values = oqs.evaluate_state_observables([up, oqs.ket2dm(plus)], metrics)

    assert np.allclose(values["purity"].real, [1.0, 1.0])
    assert np.allclose(values["entropy"].real, [0.0, 0.0])
    assert np.allclose(values["linear_entropy"].real, [0.0, 0.0])
    assert np.allclose(values["participation_ratio"].real, [1.0, 1.0])
    assert np.allclose(values["population_0"].real, [1.0, 0.5])
    assert np.allclose(values["population_1"].real, [0.0, 0.5])
    assert np.allclose(values["l1_coherence"].real, [0.0, 1.0])
    assert np.allclose(values["bloch_x"].real, [0.0, 1.0])
    assert np.allclose(values["bloch_y"].real, [0.0, 0.0])
    assert np.allclose(values["bloch_z"].real, [1.0, 0.0])
    assert np.allclose(values["fidelity"].real, [1.0, 0.5])
    assert np.allclose(values["trace_distance"].real, [0.0, np.sqrt(0.5)])


def test_named_state_observable_helpers() -> None:
    up = np.array([1.0, 0.0], dtype=np.complex128)
    mixed = 0.5 * np.eye(2, dtype=np.complex128)

    callbacks = {}
    callbacks.update(oqs.fidelity_observable(up, name="fid_up"))
    callbacks.update(oqs.trace_distance_observable(up, name="dist_up"))
    callbacks.update(oqs.purity_observable(name="p"))
    callbacks.update(oqs.entropy_observable(name="s"))
    callbacks.update(oqs.linear_entropy_observable(normalized=True, name="slin"))
    callbacks.update(oqs.participation_ratio_observable(name="pr"))
    callbacks.update(oqs.l1_coherence_observable(name="coh"))
    callbacks.update(oqs.population_observable(0, name="p0"))

    values = oqs.evaluate_state_observables([mixed], callbacks)

    assert values["fid_up"][0].real == pytest.approx(0.5)
    assert values["dist_up"][0].real == pytest.approx(0.5)
    assert values["p"][0].real == pytest.approx(0.5)
    assert values["s"][0].real == pytest.approx(1.0)
    assert values["slin"][0].real == pytest.approx(1.0)
    assert values["pr"][0].real == pytest.approx(2.0)
    assert values["coh"][0].real == pytest.approx(0.0)
    assert values["p0"][0].real == pytest.approx(0.5)
