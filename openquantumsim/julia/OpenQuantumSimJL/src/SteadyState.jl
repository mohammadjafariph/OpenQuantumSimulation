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

function steadystate(
    H,
    c_ops = Matrix{ComplexF64}[];
    method::AbstractString = "iterative-gmres",
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
    method_name = lowercase(String(method))
    krylov_dim > 0 || throw(ArgumentError("krylov_dim must be positive."))

    solution = if method_name in ("direct", "sparse-direct", "lu")
        A \ b
    elseif method_name in ("iterative-gmres", "gmres", "krylov")
        values, _info = linsolve(
            A,
            b;
            krylovdim = Int(krylov_dim),
            tol = Float64(rtol),
        )
        values
    else
        throw(ArgumentError("unknown steadystate method: $(method)"))
    end

    rho = reshape(Vector{ComplexF64}(solution), d, d)
    return _normalize_density_matrix(rho)
end
