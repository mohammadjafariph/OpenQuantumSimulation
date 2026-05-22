from openquantumsim.solvers import _to_python_dict


class _NamedTupleLikeStats:
    nsteps = 12
    retcode = "Success"
    method = "ode"

    def __dir__(self) -> list[str]:
        return ["method", "nsteps", "retcode"]

    def __getitem__(self, _key: object) -> object:
        raise AssertionError("_to_python_dict should not probe missing fields")


def test_solver_stats_conversion_uses_present_fields_only() -> None:
    stats = _to_python_dict(_NamedTupleLikeStats())

    assert stats == {"nsteps": 12, "retcode": "Success", "method": "ode"}


def test_solver_stats_conversion_accepts_mappings() -> None:
    assert _to_python_dict({"retcode": "Success"}) == {"retcode": "Success"}
