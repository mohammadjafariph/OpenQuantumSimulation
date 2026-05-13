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
    method::AbstractString = "krylov",
    krylov_dim::Integer = 30,
)
    method_name = lowercase(String(method))

    if method_name in ("krylov", "auto")
        sol_u = Vector{Vector{ComplexF64}}(undef, length(times))
        current = copy(u0)
        current_time = 0.0
        for idx in eachindex(times)
            dt = times[idx] - current_time
            if dt != 0
                current = krylov_expmv(
                    L,
                    dt,
                    current;
                    krylov_dim = krylov_dim,
                    tol = max(Float64(rtol), Float64(atol)),
                )
            end
            sol_u[idx] = copy(current)
            current_time = times[idx]
        end
        return sol_u, "Success"
    elseif method_name in ("ode", "tsit5")
        if times[end] == 0
            return [copy(u0) for _ in times], "Success"
        end
        rhs!(du, u, _p, _t) = (mul!(du, L, u); nothing)
        problem = ODEProblem(rhs!, u0, (0.0, times[end]))
        sol = solve(
            problem,
            Tsit5();
            saveat = times,
            reltol = Float64(rtol),
            abstol = Float64(atol),
        )
        return Vector{Vector{ComplexF64}}(sol.u), string(sol.retcode)
    end

    throw(ArgumentError("unknown correlation propagation method: $(method)"))
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
    method::AbstractString = "krylov",
    krylov_dim::Integer = 30,
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

    L = liouvillian(Hc, collapse_ops)
    u0 = vec(Bc * rho0c)
    elapsed = @elapsed begin
        sol_u, retcode = _propagate_liouville_from_zero(
            L,
            u0,
            taus;
            rtol = rtol,
            atol = atol,
            method = method,
            krylov_dim = krylov_dim,
        )
    end

    values = zeros(ComplexF64, length(taus))
    for idx in eachindex(taus)
        values[idx] = tr(Ac * reshape(sol_u[idx], d, d))
    end

    return (
        taulist = taus,
        correlations = values,
        solver_stats = (
            nsteps = length(taus),
            nfev = 0,
            wall_time = elapsed,
            retcode = retcode,
            method = lowercase(String(method)),
            krylov_dim = Int(krylov_dim),
            quantum_regression = true,
        ),
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
    method::AbstractString = "krylov",
    krylov_dim::Integer = 30,
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

    L = liouvillian(Hc, collapse_ops)
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
        )

        values = zeros(ComplexF64, length(times), length(taus))
        retcodes = String[state_result.solver_stats.retcode]
        for tidx in eachindex(times)
            rho_t = state_result.states[:, :, tidx]
            u0 = vec(Bc * rho_t)
            sol_u, retcode = _propagate_liouville_from_zero(
                L,
                u0,
                taus;
                rtol = rtol,
                atol = atol,
                method = method,
                krylov_dim = krylov_dim,
            )
            push!(retcodes, retcode)
            for tauidx in eachindex(taus)
                values[tidx, tauidx] = tr(Ac * reshape(sol_u[tauidx], d, d))
            end
        end
    end

    success = all(retcode == "Success" for retcode in retcodes)
    return (
        tlist = times,
        taulist = taus,
        correlations = values,
        solver_stats = (
            nsteps = length(times) * length(taus),
            nfev = 0,
            wall_time = elapsed,
            retcode = success ? "Success" : join(retcodes, ","),
            method = lowercase(String(method)),
            krylov_dim = Int(krylov_dim),
            quantum_regression = true,
        ),
    )
end
