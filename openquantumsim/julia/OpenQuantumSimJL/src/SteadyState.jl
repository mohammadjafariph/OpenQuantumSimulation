function _trace_constraint_row(d::Int)
    n = d * d
    diagonal_indices = [idx + (idx - 1) * d for idx in 1:d]
    return sparse(ones(Int, d), diagonal_indices, ones(ComplexF64, d), 1, n)
end

function _steady_state_system(L, d::Int)
    trace_row = _trace_constraint_row(d)
    A = vcat(trace_row, L[2:end, :])
    b = zeros(ComplexF64, d * d)
    b[1] = 1
    return A, b
end

function _normalize_density_matrix(rho::Matrix{ComplexF64})::Matrix{ComplexF64}
    rho_h = Matrix{ComplexF64}((rho .+ rho') ./ 2)
    rho_trace = tr(rho_h)
    abs(rho_trace) > 0 ||
        throw(ArgumentError("steady-state solve returned a zero-trace density matrix."))
    rho_h ./= rho_trace
    return rho_h
end

# `method = "auto"` uses the sparse direct solve up to this many unknowns
# (d^2), where it is fast and always accurate, and GMRES above it.
const STEADYSTATE_DIRECT_MAX_UNKNOWNS = 16384

"""
Solve for the Lindblad steady state and report how well the answer satisfies
`L rho = 0`.

`method` is `"auto"`, `"direct"` (sparse LU) or `"iterative-gmres"`. GMRES
is not preconditioned and can stagnate; when it does, `converged` is `false`.

Returns `(rho, solver_stats)`. `residual` is `norm(L * vec(rho))` for the
returned (Hermitian, unit-trace) state and `relative_residual` divides it by
the 1-norm of `L`, so it can be compared across problems. For the iterative
method `converged` is the GMRES convergence flag; for the direct method it
is `true` whenever the factorisation returned a finite state.
"""
function steadystate_info(
    H,
    c_ops = Matrix{ComplexF64}[];
    method::AbstractString = "auto",
    rtol::Real = 1e-10,
    krylov_dim::Integer = 30,
)
    Hc = _as_complex_operator(H)
    d = size(Hc, 1)
    size(Hc, 2) == d || throw(DimensionMismatch("Hamiltonian must be square."))

    collapse_ops = _as_complex_operators(c_ops)
    for c in collapse_ops
        size(c) == (d, d) || throw(DimensionMismatch("collapse operators must match H."))
    end

    L = liouvillian(Hc, collapse_ops)
    A, b = _steady_state_system(L, d)
    requested_method = lowercase(String(method))
    method_name = requested_method
    if method_name == "auto"
        method_name = d * d <= STEADYSTATE_DIRECT_MAX_UNKNOWNS ? "direct" : "iterative-gmres"
    end
    krylov_dim > 0 || throw(ArgumentError("krylov_dim must be positive."))

    converged = true
    linear_residual = 0.0
    numops = 0
    local solution
    elapsed = @elapsed begin
        if method_name in ("direct", "sparse-direct", "lu")
            solution = A \ b
            method_name = "direct"
        elseif method_name in ("iterative-gmres", "gmres", "krylov")
            solution, info = linsolve(
                A,
                b;
                krylovdim = Int(krylov_dim),
                tol = Float64(rtol),
                verbosity = 0,
            )
            converged = info.converged > 0
            linear_residual = Float64(info.normres)
            numops = Int(info.numops)
            method_name = "iterative-gmres"
        else
            throw(ArgumentError("unknown steadystate method: $(method)"))
        end
    end

    raw = reshape(Vector{ComplexF64}(solution), d, d)
    finite = all(isfinite, raw)
    rho = finite ? _normalize_density_matrix(raw) : raw
    residual = finite ? norm(L * vec(rho)) : NaN
    scale = opnorm(L, 1)
    return (
        rho = rho,
        solver_stats = (
            method = method_name,
            requested_method = requested_method,
            wall_time = elapsed,
            retcode = !finite ? "NonFiniteSolution" :
                converged ? SUCCESS_RETCODE : "GMRESNotConverged",
            success = finite && converged,
            converged = finite && converged,
            residual = residual,
            relative_residual = scale > 0 ? residual / scale : residual,
            linear_residual = linear_residual,
            nfev = numops,
            krylov_dim = Int(krylov_dim),
            min_eigenvalue = finite ? eigmin(Hermitian(rho)) : NaN,
            trace_drift = finite ? abs(tr(rho) - 1) : NaN,
        ),
    )
end

function steadystate(
    H,
    c_ops = Matrix{ComplexF64}[];
    method::AbstractString = "auto",
    rtol::Real = 1e-10,
    krylov_dim::Integer = 30,
)
    return steadystate_info(
        H,
        c_ops;
        method = method,
        rtol = rtol,
        krylov_dim = krylov_dim,
    ).rho
end
