import pytest

import openquantumsim as oqs


def test_fock_and_spin_dimensions() -> None:
    cavity = oqs.FockSpace(5, label="cavity")
    atom = oqs.SpinSpace(0.5, label="atom")
    ensemble = oqs.DickeSpace(4, label="ensemble")
    space = cavity * atom * ensemble

    assert cavity.dim == 5
    assert atom.dim == 2
    assert ensemble.dim == 5
    assert ensemble.total_spin == 2
    assert space.dim == 50


def test_invalid_spin_rejected() -> None:
    with pytest.raises(ValueError):
        oqs.SpinSpace(0.25)


def test_invalid_dicke_space_rejected() -> None:
    with pytest.raises(ValueError):
        oqs.DickeSpace(-1)
