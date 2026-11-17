function _as_complex_matrix(matrix)::Matrix{ComplexF64}
    return Matrix{ComplexF64}(ComplexF64.(matrix))
end

function _as_complex_operator(matrix)::SparseMatrixCSC{ComplexF64, Int64}
    return dropzeros!(sparse(ComplexF64.(matrix)))
end

function _as_complex_matrices(matrices)::Vector{Matrix{ComplexF64}}
    converted = Matrix{ComplexF64}[]
    for matrix in matrices
        push!(converted, _as_complex_matrix(matrix))
    end
    return converted
end

function _as_complex_operators(matrices)::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    converted = SparseMatrixCSC{ComplexF64, Int64}[]
    for matrix in matrices
        push!(converted, _as_complex_operator(matrix))
    end
    return converted
end

function _validate_mesolve_inputs(
    H::AbstractMatrix{ComplexF64},
    rho0::Matrix{ComplexF64},
    times::Vector{Float64},
)
    d = size(H, 1)
    size(H, 2) == d || throw(DimensionMismatch("Hamiltonian must be square."))
    size(rho0) == (d, d) || throw(DimensionMismatch("rho0 must have the same dimension as H."))
    length(times) > 0 || throw(ArgumentError("tlist must contain at least one time."))
    all(diff(times) .>= 0) || throw(ArgumentError("tlist must be sorted in ascending order."))
    return d
end

function mesolve(
    H,
    rho0,
    tlist,
    c_ops = Matrix{ComplexF64}[],
    e_ops = Matrix{ComplexF64}[],
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    save_states::Bool = false,
    method::AbstractString = "auto",
    krylov_dim::Integer = 30,
    compute_entropy::Bool = true,
    ode_solver::AbstractString = "auto",
)
    Hc = _as_complex_operator(H)
    rho0c = _as_complex_matrix(rho0)
    times = Float64.(collect(tlist))
    d = _validate_mesolve_inputs(Hc, rho0c, times)

    collapse_ops = _as_complex_operators(c_ops)
    expectation_ops = _as_complex_operators(e_ops)
    for c in collapse_ops
        size(c) == (d, d) || throw(DimensionMismatch("collapse operators must match H."))
    end
    for op in expectation_ops
        size(op) == (d, d) || throw(DimensionMismatch("expectation operators must match H."))
    end

    L = liouvillian_operator(Hc, collapse_ops)
    u0 = vec(copy(rho0c))
    requested_method, method_name, ode_solver_name = _resolve_method(method, ode_solver)

    sol_u, report, elapsed = _solve_liouvillian(
        L,
        u0,
        times;
        rtol = Float64(rtol),
        atol = Float64(atol),
        method_name = method_name,
        krylov_dim = Int(krylov_dim),
        ode_solver = ode_solver_name,
    )

    return _collect_solution(
        sol_u,
        times,
        d,
        expectation_ops;
        save_states = save_states,
        compute_entropy = compute_entropy,
        elapsed = elapsed,
        report = report,
        method_name = method_name,
        requested_method = requested_method,
        krylov_dim = Int(krylov_dim),
    )
end

"""
Normalise the `method` / `ode_solver` pair. `method` picks the strategy
(`"ode"` or `"krylov"`); `ode_solver` picks the integrator used by `"ode"`.
`method = "tsit5"` is kept as an alias for `method = "ode"` with Tsit5.
"""
function _resolve_method(method::AbstractString, ode_solver::AbstractString)
    requested_method = lowercase(String(method))
    solver_name = lowercase(String(ode_solver))
    if requested_method == "tsit5"
        solver_name in ("auto", "tsit5") || throw(ArgumentError(
            "method = \"tsit5\" conflicts with ode_solver = \"$(solver_name)\".",
        ))
        return requested_method, "ode", "tsit5"
    end
    method_name = requested_method == "auto" ? "ode" : requested_method
    method_name in ("ode", "krylov") ||
        throw(ArgumentError("unknown solver method: $(method_name)"))
    return requested_method, method_name, solver_name
end

function _solve_liouvillian(
    L,
    u0::Vector{ComplexF64},
    times::Vector{Float64};
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    method_name::AbstractString = "ode",
    krylov_dim::Integer = 30,
    ode_solver::AbstractString = "auto",
)
    local sol_u, report
    elapsed = @elapsed begin
        L = _matvec_operator(L)
        if method_name == "krylov"
            sol_u, report = _solve_krylov_at_times(
                L,
                u0,
                times,
                times[1];
                tol = max(Float64(rtol), Float64(atol)),
                krylov_dim = krylov_dim,
            )
        elseif method_name == "ode"
            sol_u, report = _solve_ode_at_times(
                _linear_rhs!,
                u0,
                times,
                L;
                rtol = rtol,
                atol = atol,
                ode_solver = ode_solver,
            )
        else
            throw(ArgumentError("unknown solver method: $(method_name)"))
        end
    end
    return sol_u, report, elapsed
end

function _collect_solution(
    sol_u,
    times::Vector{Float64},
    d::Integer,
    expectation_ops::Vector{<:AbstractMatrix{ComplexF64}};
    save_states::Bool = false,
    compute_entropy::Bool = true,
    elapsed::Real = 0.0,
    report = _ode_report(),
    method_name::AbstractString = "ode",
    requested_method::AbstractString = "auto",
    krylov_dim::Integer = 30,
)
    n_times = length(times)
    n_expect = length(expectation_ops)
    expect_values = zeros(ComplexF64, n_expect, n_times)
    entropy_values = compute_entropy ? zeros(Float64, n_times) : Float64[]
    states = Array{ComplexF64}(undef, d, d, save_states ? n_times : 0)

    for idx in 1:n_times
        rho = reshape(sol_u[idx], d, d)
        finite = all(isfinite, rho)
        if compute_entropy
            entropy_values[idx] = finite ?
                von_neumann_entropy(Hermitian((rho .+ rho') ./ 2)) :
                NaN
        end
        for op_idx in 1:n_expect
            expect_values[op_idx, idx] = expect(expectation_ops[op_idx], rho)
        end
        if save_states
            states[:, :, idx] .= rho
        end
    end

    health = _density_health(sol_u, d)
    return (
        times = times,
        states = states,
        expect = expect_values,
        entropy = entropy_values,
        solver_stats = (
            nsteps = report.nsteps,
            nfev = report.nfev,
            nreject = report.nreject,
            wall_time = elapsed,
            retcode = report.retcode,
            success = report.success,
            method = method_name,
            requested_method = requested_method,
            ode_solver = report.ode_solver,
            krylov_dim = Int(krylov_dim),
            krylov_converged = report.krylov_converged,
            krylov_max_residual = report.krylov_max_residual,
            compute_entropy = compute_entropy,
            trace_drift = health.trace_drift,
            hermiticity_drift = health.hermiticity_drift,
            min_eigenvalue = health.min_eigenvalue,
            positivity_samples = health.positivity_samples,
            n_nonfinite_states = health.n_nonfinite_states,
        ),
    )
end
