# Solver health reporting: integrator return codes, Krylov convergence, and
# density-matrix invariants. Solvers never swallow a failure; they record it in
# `solver_stats` so the Python wrappers can raise or warn.

const SUCCESS_RETCODE = "Success"

# Positivity needs an eigenvalue decomposition (O(d^3)) per checked state, which
# costs more than the whole solve when done at every output time. It is
# therefore a spot check at a few evenly spaced times, always including the
# first and the last. Trace and Hermiticity are cheap and checked everywhere.
const POSITIVITY_SMALL_DIM = 64
const POSITIVITY_SAMPLES_SMALL = 8
const POSITIVITY_SAMPLES_LARGE = 4

function _ode_algorithm(name::AbstractString)
    key = lowercase(String(name))
    key in ("auto", "tsit5") && return Tsit5(), "tsit5"
    key == "vern7" && return Vern7(), "vern7"
    key == "vern9" && return Vern9(), "vern9"
    throw(ArgumentError(
        "unknown ode_solver: $(name); expected auto, tsit5, vern7, or vern9.",
    ))
end

_linear_rhs!(du, u, L, _t) = (mul!(du, L, u); nothing)

function _nan_state(n::Int)
    return fill(ComplexF64(NaN, NaN), n)
end

function _ode_report(;
    retcode::AbstractString = SUCCESS_RETCODE,
    nsteps::Integer = 0,
    nfev::Integer = 0,
    nreject::Integer = 0,
    ode_solver::AbstractString = "",
)
    return (
        retcode = String(retcode),
        success = retcode == SUCCESS_RETCODE,
        nsteps = Int(nsteps),
        nfev = Int(nfev),
        nreject = Int(nreject),
        ode_solver = String(ode_solver),
        krylov_converged = true,
        krylov_max_residual = 0.0,
    )
end

function _krylov_report(;
    converged::Bool,
    max_residual::Real,
    nsteps::Integer,
    numops::Integer,
)
    return (
        retcode = converged ? SUCCESS_RETCODE : "KrylovNotConverged",
        success = converged,
        nsteps = Int(nsteps),
        nfev = Int(numops),
        nreject = 0,
        ode_solver = "",
        krylov_converged = converged,
        krylov_max_residual = Float64(max_residual),
    )
end

"""
Integrate `du/dt = rhs!(du, u, params, t)` and return one state per entry of
`times` (repeated times are allowed) together with a health report.

If the integrator stops early the missing states are filled with `NaN`, and
the report carries the failing return code; nothing is silently truncated.
"""
function _solve_ode_at_times(
    rhs!,
    u0::Vector{ComplexF64},
    times::Vector{Float64},
    params;
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    ode_solver::AbstractString = "auto",
)
    algorithm, algorithm_name = _ode_algorithm(ode_solver)
    n_times = length(times)
    if n_times == 0 || times[end] == times[1]
        return [copy(u0) for _ in 1:n_times], _ode_report(; ode_solver = algorithm_name)
    end

    save_times = unique(times)
    problem = ODEProblem(rhs!, u0, (times[1], times[end]), params)
    sol = solve(
        problem,
        algorithm;
        saveat = save_times,
        reltol = Float64(rtol),
        abstol = Float64(atol),
    )

    saved = sol.u
    states = Vector{Vector{ComplexF64}}(undef, n_times)
    for idx in 1:n_times
        save_idx = searchsortedfirst(save_times, times[idx])
        states[idx] = save_idx <= length(saved) ?
            # The integrator stores a separate copy per saved time, so the
            # saved vector can be used without copying it again.
            convert(Vector{ComplexF64}, saved[save_idx]) :
            _nan_state(length(u0))
    end

    retcode = string(sol.retcode)
    if retcode == SUCCESS_RETCODE && length(saved) != length(save_times)
        retcode = "IncompleteSolution"
    end
    report = _ode_report(;
        retcode = retcode,
        nsteps = sol.stats.naccept,
        nfev = sol.stats.nf,
        nreject = sol.stats.nreject,
        ode_solver = algorithm_name,
    )
    return states, report
end

"""
Propagate `u0` through `exp(L * dt)` between consecutive `times` with Krylov
exponentiation, starting from `start_time`. Returns the states and a report
with the worst residual and whether every step converged.
"""
function _solve_krylov_at_times(
    L,
    u0::Vector{ComplexF64},
    times::Vector{Float64},
    start_time::Float64;
    tol::Real = 1e-10,
    krylov_dim::Integer = 30,
)
    states = Vector{Vector{ComplexF64}}(undef, length(times))
    current = copy(u0)
    current_time = start_time
    converged = true
    max_residual = 0.0
    numops = 0
    nsteps = 0
    for idx in eachindex(times)
        dt = times[idx] - current_time
        if dt != 0
            step = krylov_expmv_info(L, dt, current; krylov_dim = krylov_dim, tol = tol)
            current = step.vector
            converged &= step.converged
            max_residual = max(max_residual, step.residual)
            numops += step.numops
            nsteps += 1
        end
        states[idx] = copy(current)
        current_time = times[idx]
    end
    report = _krylov_report(;
        converged = converged,
        max_residual = max_residual,
        nsteps = nsteps,
        numops = numops,
    )
    return states, report
end

function _positivity_sample_indices(n_times::Int, d::Int)
    n_samples = d <= POSITIVITY_SMALL_DIM ? POSITIVITY_SAMPLES_SMALL : POSITIVITY_SAMPLES_LARGE
    n_times <= n_samples && return collect(1:n_times)
    return unique(round.(Int, range(1, n_times; length = n_samples)))
end

"""
Invariants of a density-matrix time series stored as vectorised states:
the largest change of the trace relative to the first state, the largest
anti-Hermitian element, and the most negative eigenvalue of the Hermitian
part among the spot-checked states (0 when they are all positive to within
`POSITIVITY_MARGIN`; see `_negative_eigenvalue`). Non-finite states (from a
failed integration) are skipped.
"""
function _density_health(sol_u, d::Integer)
    n_times = length(sol_u)
    trace_drift = 0.0
    hermiticity_drift = 0.0
    min_eigenvalue = NaN
    n_finite = 0
    n_times == 0 && return (
        trace_drift = trace_drift,
        hermiticity_drift = hermiticity_drift,
        min_eigenvalue = min_eigenvalue,
        positivity_samples = 0,
        n_nonfinite_states = 0,
    )

    reference_trace = tr(reshape(sol_u[1], d, d))
    for idx in 1:n_times
        rho = reshape(sol_u[idx], d, d)
        all(isfinite, rho) || continue
        n_finite += 1
        trace_drift = max(trace_drift, abs(tr(rho) - reference_trace))
        # Compare squared magnitudes and take one square root at the end;
        # abs() on every element pair would dominate the check.
        worst = 0.0
        @inbounds for col in 1:d, row in 1:col
            worst = max(worst, abs2(rho[row, col] - conj(rho[col, row])))
        end
        hermiticity_drift = max(hermiticity_drift, sqrt(worst))
    end

    samples = 0
    for idx in _positivity_sample_indices(n_times, Int(d))
        rho = reshape(sol_u[idx], d, d)
        all(isfinite, rho) || continue
        smallest = _negative_eigenvalue(rho)
        min_eigenvalue = samples == 0 ? smallest : min(min_eigenvalue, smallest)
        samples += 1
    end

    return (
        trace_drift = trace_drift,
        hermiticity_drift = hermiticity_drift,
        min_eigenvalue = min_eigenvalue,
        positivity_samples = samples,
        n_nonfinite_states = n_times - n_finite,
    )
end

# A state counts as positive when rho + margin * trace * I has a Cholesky
# factorisation. That proves every eigenvalue is above -margin * trace at a
# fraction of the cost of computing the smallest one.
const POSITIVITY_MARGIN = 1e-9

"""
Return the smallest eigenvalue of the Hermitian part of `rho` if it is
negative beyond `POSITIVITY_MARGIN` (relative to the trace), and 0 otherwise.

The eigenvalue decomposition, which costs several times more than a Cholesky
factorisation, is only computed for a state that fails the Cholesky test.
"""
function _negative_eigenvalue(rho::AbstractMatrix)::Float64
    hermitian = Matrix{ComplexF64}(undef, size(rho))
    @inbounds for col in axes(rho, 2), row in axes(rho, 1)
        hermitian[row, col] = (rho[row, col] + conj(rho[col, row])) / 2
    end
    scale = max(1.0, abs(real(tr(hermitian))))
    shifted = copy(hermitian)
    @inbounds for idx in axes(shifted, 1)
        shifted[idx, idx] += POSITIVITY_MARGIN * scale
    end
    issuccess(cholesky!(Hermitian(shifted); check = false)) && return 0.0
    return min(0.0, eigmin(Hermitian(hermitian)))
end
