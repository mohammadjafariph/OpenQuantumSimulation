function _validate_correlation_times(times::Vector{Float64}, name::AbstractString)
    length(times) > 0 || throw(ArgumentError("$(name) must contain at least one time."))
    all(times .>= 0) || throw(ArgumentError("$(name) must be non-negative."))
    all(diff(times) .>= 0) || throw(ArgumentError("$(name) must be sorted in ascending order."))
    return nothing
end

function _validate_operator_shape(op::Matrix{ComplexF64}, d::Int, name::AbstractString)
    size(op) == (d, d) || throw(DimensionMismatch("$(name) must match H."))
    return nothing
end

function _propagate_liouville_from_zero(
    L,
    u0::Vector{ComplexF64},
    times::Vector{Float64};
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    method::AbstractString = "auto",
    krylov_dim::Integer = 30,
    ode_solver::AbstractString = "auto",
)
    # "auto" integrates with an ODE solver: it was 2.5-3.6x faster than
    # Krylov propagation for typical delay grids at the same accuracy.
    method_name = lowercase(String(method))
    method_name == "auto" && (method_name = "ode")

    if method_name == "krylov"
        return _solve_krylov_at_times(
            L,
            u0,
            times,
            0.0;
            tol = max(Float64(rtol), Float64(atol)),
            krylov_dim = krylov_dim,
        )
    elseif method_name in ("ode", "tsit5")
        # Integrate from tau = 0 even when the first requested delay is later.
        grid = times[1] == 0 ? times : vcat(0.0, times)
        states, report = _solve_ode_at_times(
            _linear_rhs!,
            u0,
            grid,
            L;
            rtol = rtol,
            atol = atol,
            ode_solver = method_name == "tsit5" ? "tsit5" : ode_solver,
        )
        return (times[1] == 0 ? states : states[2:end]), report
    end

    throw(ArgumentError("unknown correlation propagation method: $(method)"))
end

function _correlation_stats(reports, elapsed::Real, nsteps::Integer, method, krylov_dim)
    failed = [report.retcode for report in reports if !report.success]
    return (
        nsteps = Int(nsteps),
        nfev = sum(report.nfev for report in reports; init = 0),
        wall_time = elapsed,
        retcode = isempty(failed) ? SUCCESS_RETCODE : join(unique(failed), ","),
        success = isempty(failed),
        method = lowercase(String(method)),
        krylov_dim = Int(krylov_dim),
        krylov_converged = all(report.krylov_converged for report in reports),
        krylov_max_residual = maximum(
            report.krylov_max_residual for report in reports;
            init = 0.0,
        ),
        quantum_regression = true,
    )
end

function correlation_2op_1t(
    H,
    rho0,
    taulist,
    a_op,
    b_op,
    c_ops = Matrix{ComplexF64}[];
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    method::AbstractString = "auto",
    krylov_dim::Integer = 30,
    ode_solver::AbstractString = "auto",
)
    Hc = _as_complex_matrix(H)
    rho0c = _as_complex_matrix(rho0)
    taus = Float64.(collect(taulist))
    d = _validate_mesolve_inputs(Hc, rho0c, [0.0])
    _validate_correlation_times(taus, "taulist")

    Ac = _as_complex_matrix(a_op)
    Bc = _as_complex_matrix(b_op)
    _validate_operator_shape(Ac, d, "a_op")
    _validate_operator_shape(Bc, d, "b_op")

    collapse_ops = _as_complex_matrices(c_ops)
    for c in collapse_ops
        _validate_operator_shape(c, d, "collapse operator")
    end

    L = liouvillian_operator(Hc, collapse_ops)
    u0 = vec(Bc * rho0c)
    local sol_u, report
    elapsed = @elapsed begin
        sol_u, report = _propagate_liouville_from_zero(
            L,
            u0,
            taus;
            rtol = rtol,
            atol = atol,
            method = method,
            krylov_dim = krylov_dim,
            ode_solver = ode_solver,
        )
    end

    values = zeros(ComplexF64, length(taus))
    for idx in eachindex(taus)
        values[idx] = tr(Ac * reshape(sol_u[idx], d, d))
    end

    return (
        taulist = taus,
        correlations = values,
        solver_stats = _correlation_stats([report], elapsed, length(taus), method, krylov_dim),
    )
end

function correlation_2op_2t(
    H,
    rho0,
    tlist,
    taulist,
    a_op,
    b_op,
    c_ops = Matrix{ComplexF64}[];
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    method::AbstractString = "auto",
    krylov_dim::Integer = 30,
    ode_solver::AbstractString = "auto",
)
    Hc = _as_complex_matrix(H)
    rho0c = _as_complex_matrix(rho0)
    times = Float64.(collect(tlist))
    taus = Float64.(collect(taulist))
    d = _validate_mesolve_inputs(Hc, rho0c, times)
    _validate_correlation_times(taus, "taulist")

    Ac = _as_complex_matrix(a_op)
    Bc = _as_complex_matrix(b_op)
    _validate_operator_shape(Ac, d, "a_op")
    _validate_operator_shape(Bc, d, "b_op")

    collapse_ops = _as_complex_matrices(c_ops)
    for c in collapse_ops
        _validate_operator_shape(c, d, "collapse operator")
    end

    L = liouvillian_operator(Hc, collapse_ops)
    values = zeros(ComplexF64, length(times), length(taus))
    reports = []
    elapsed = @elapsed begin
        state_result = mesolve(
            Hc,
            rho0c,
            times,
            collapse_ops,
            Matrix{ComplexF64}[],
            rtol,
            atol,
            true,
            method,
            krylov_dim,
            false,
            ode_solver,
        )
        stats = state_result.solver_stats
        push!(reports, (
            retcode = stats.retcode,
            success = stats.success,
            nfev = stats.nfev,
            krylov_converged = stats.krylov_converged,
            krylov_max_residual = stats.krylov_max_residual,
        ))

        for tidx in eachindex(times)
            rho_t = state_result.states[:, :, tidx]
            u0 = vec(Bc * rho_t)
            sol_u, report = _propagate_liouville_from_zero(
                L,
                u0,
                taus;
                rtol = rtol,
                atol = atol,
                method = method,
                krylov_dim = krylov_dim,
                ode_solver = ode_solver,
            )
            push!(reports, report)
            for tauidx in eachindex(taus)
                values[tidx, tauidx] = tr(Ac * reshape(sol_u[tauidx], d, d))
            end
        end
    end

    return (
        tlist = times,
        taulist = taus,
        correlations = values,
        solver_stats = _correlation_stats(
            reports,
            elapsed,
            length(times) * length(taus),
            method,
            krylov_dim,
        ),
    )
end
