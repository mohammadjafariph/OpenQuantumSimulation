import numpy as np

import openquantumsim as oqs


def test_phase_space_plot_helpers_return_axes() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)

    state = oqs.fock(oqs.FockSpace(4), 0)
    ax_w = oqs.plot_wigner(state, points=25, colorbar=False)
    ax_q = oqs.plot_q_function(state, points=25, colorbar=False)

    assert ax_w.get_xlabel() == "x"
    assert ax_w.get_ylabel() == "p"
    assert ax_q.get_title() == "Husimi Q function"


def test_result_plot_helpers_return_axes() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)

    result = oqs.Result(
        times=np.array([0.0, 1.0], dtype=np.float64),
        expect=[np.array([1.0, 0.5], dtype=np.complex128)],
        state_observables={"purity": np.array([1.0, 0.75], dtype=np.complex128)},
    )

    ax_expect = oqs.plot_expectations(result, labels=["population"])
    ax_observable = oqs.plot_state_observable(result, "purity")

    assert ax_expect.get_ylabel() == "expectation"
    assert ax_observable.get_ylabel() == "purity"


def test_density_matrix_plot_helper_return_axes() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)

    rho = oqs.ket2dm(np.array([1.0, 0.0], dtype=np.complex128))
    ax = oqs.plot_density_matrix(rho, colorbar=False)

    assert ax.get_xlabel() == "column"
    assert ax.get_ylabel() == "row"
