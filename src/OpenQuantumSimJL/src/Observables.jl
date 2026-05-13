function partial_trace_A(psi::AbstractVector{ComplexF64}, dA::Int, dB::Int)
    length(psi) == dA * dB || throw(DimensionMismatch("psi length does not match dA*dB."))
    M = reshape(psi, dB, dA)
    return M' * M
end

function partial_traces(psi::AbstractVector{ComplexF64}, dA::Int, dB::Int)
    length(psi) == dA * dB || throw(DimensionMismatch("psi length does not match dA*dB."))
    M = reshape(psi, dB, dA)
    rhoA = M' * M
    rhoB = M * M'
    return rhoA, rhoB
end

function von_neumann_entropy(rho::Hermitian)
    lambda = eigvals(rho)
    lambda .= max.(lambda, 0.0)
    total = sum(lambda)
    total > 0 && (lambda ./= total)
    return -sum(v * log2(v) for v in lambda if v > 1e-15; init=0.0)
end

function von_neumann_entropy(rho::AbstractMatrix)
    return von_neumann_entropy(Hermitian((rho .+ rho') ./ 2))
end

function purity(rho::AbstractMatrix)::Float64
    return real(tr(rho * rho))
end

function expect(op::AbstractMatrix, state::AbstractVector{ComplexF64})
    return dot(state, op, state)
end

function expect(op::AbstractMatrix, rho::AbstractMatrix)
    size(op) == size(rho) || throw(DimensionMismatch("operator and density matrix sizes differ."))
    d = size(rho, 1)
    size(rho, 2) == d || throw(DimensionMismatch("density matrix must be square."))
    value = zero(ComplexF64)
    @inbounds for col in 1:d
        for row in 1:d
            value += op[row, col] * rho[col, row]
        end
    end
    return value
end

function expect(op::SparseMatrixCSC{ComplexF64}, rho::AbstractMatrix)
    size(op) == size(rho) || throw(DimensionMismatch("operator and density matrix sizes differ."))
    value = zero(ComplexF64)
    rows = rowvals(op)
    vals = nonzeros(op)
    @inbounds for col in 1:size(op, 2)
        for ptr in nzrange(op, col)
            row = rows[ptr]
            value += vals[ptr] * rho[col, row]
        end
    end
    return value
end

function precompute_F(N::Int, k::Int)::Matrix{Float64}
    logC(n, kk) = (kk < 0 || kk > n) ? -Inf :
        loggamma(n + 1) - loggamma(kk + 1) - loggamma(n - kk + 1)
    F = zeros(Float64, N + 1, k + 1)
    log_denom = logC(N, k)
    for r in 0:N, p in 0:k
        lv = logC(r, p) + logC(N - r, k - p) - log_denom
        F[r + 1, p + 1] = isfinite(lv) ? exp(0.5 * lv) : 0.0
    end
    return F
end

function krdm(rho::AbstractMatrix, F::Matrix{Float64})
    rho_k = Hermitian(F' * rho * F)
    tr_rho_k = real(tr(rho_k))
    return tr_rho_k > 0 ? Hermitian(rho_k.data ./ tr_rho_k) : rho_k
end
