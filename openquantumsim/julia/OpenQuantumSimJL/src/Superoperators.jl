# Fast construction and application of Lindblad superoperators.
#
# The Liouvillian acts on column-stacked density matrices,
#     vec(A X B) = (transpose(B) ⊗ A) vec(X),
# and for H and collapse operators c_k it is
#     L = -i I⊗H + i Hᵀ⊗I + Σ_k [ conj(c_k)⊗c_k - ½ I⊗C - ½ Cᵀ⊗I ],
# with C = Σ_k c_k' c_k. Grouping terms gives three Kronecker products in total:
#     L = -i I⊗(H - i/2 C) + i (H + i/2 C)ᵀ⊗I + Σ_k conj(c_k)⊗c_k.
#
# Two choices make large problems fast:
#   * L is assembled once from (row, column, value) triplets. Adding sparse
#     matrices term by term allocates a full copy of L for every term.
#   * L is applied row by row (a gather), which reads memory in order and can
#     be split across threads. Julia's sparse matrices are stored by column,
#     so this needs the transpose of L, which is assembled directly by
#     swapping the row and column triplets.

"""
A sparse operator applied row by row: `y = L * x` with `L = transpose(Lt)`.

`mul!` uses every Julia thread when the operator is large enough for the
split to pay off. The result does not depend on the thread count: every
output element is computed by one thread with the same summation order.
"""
struct RowMajorOperator
    Lt::SparseMatrixCSC{ComplexF64, Int64}
end

# Below this many rows a single thread is faster than splitting the work.
const ROW_MAJOR_THREADING_MIN_ROWS = 16_384

Base.size(A::RowMajorOperator) = reverse(size(A.Lt))
Base.size(A::RowMajorOperator, dim::Integer) = size(A)[dim]
Base.eltype(::Type{RowMajorOperator}) = ComplexF64
Base.eltype(::RowMajorOperator) = ComplexF64

"""Return the operator as an ordinary (column-stored) sparse matrix."""
SparseArrays.sparse(A::RowMajorOperator) = sparse(transpose(A.Lt))

@inline function _row_major_rows!(y, A::RowMajorOperator, x, rows)
    Lt = A.Lt
    stored_rows = rowvals(Lt)
    values = nonzeros(Lt)
    @inbounds for j in rows
        total = zero(ComplexF64)
        for k in nzrange(Lt, j)
            total += values[k] * x[stored_rows[k]]
        end
        y[j] = total
    end
    return y
end

function LinearAlgebra.mul!(
    y::AbstractVector{ComplexF64},
    A::RowMajorOperator,
    x::AbstractVector{ComplexF64},
)
    n = size(A.Lt, 2)
    length(y) == n || throw(DimensionMismatch("output length does not match the operator."))
    length(x) == size(A.Lt, 1) ||
        throw(DimensionMismatch("input length does not match the operator."))
    workers = Threads.nthreads()
    if workers == 1 || n < ROW_MAJOR_THREADING_MIN_ROWS
        return _row_major_rows!(y, A, x, 1:n)
    end
    chunk = cld(n, workers)
    Threads.@threads :static for worker in 1:workers
        first_row = (worker - 1) * chunk + 1
        last_row = min(worker * chunk, n)
        first_row <= last_row && _row_major_rows!(y, A, x, first_row:last_row)
    end
    return y
end

@inline function _row_major_rows!(y, A::RowMajorOperator, x, rows, alpha, beta)
    Lt = A.Lt
    stored_rows = rowvals(Lt)
    values = nonzeros(Lt)
    @inbounds for j in rows
        total = zero(ComplexF64)
        for k in nzrange(Lt, j)
            total += values[k] * x[stored_rows[k]]
        end
        y[j] = iszero(beta) ? alpha * total : alpha * total + beta * y[j]
    end
    return y
end

"""`y = alpha * A * x + beta * y`, split across threads like `mul!(y, A, x)`."""
function LinearAlgebra.mul!(
    y::AbstractVector{ComplexF64},
    A::RowMajorOperator,
    x::AbstractVector{ComplexF64},
    alpha::Number,
    beta::Number,
)
    n = size(A.Lt, 2)
    length(y) == n || throw(DimensionMismatch("output length does not match the operator."))
    length(x) == size(A.Lt, 1) ||
        throw(DimensionMismatch("input length does not match the operator."))
    a, b = ComplexF64(alpha), ComplexF64(beta)
    workers = Threads.nthreads()
    if workers == 1 || n < ROW_MAJOR_THREADING_MIN_ROWS
        return _row_major_rows!(y, A, x, 1:n, a, b)
    end
    chunk = cld(n, workers)
    Threads.@threads :static for worker in 1:workers
        first_row = (worker - 1) * chunk + 1
        last_row = min(worker * chunk, n)
        first_row <= last_row && _row_major_rows!(y, A, x, first_row:last_row, a, b)
    end
    return y
end

function Base.:*(A::RowMajorOperator, x::AbstractVector)
    y = Vector{ComplexF64}(undef, size(A, 1))
    return mul!(y, A, Vector{ComplexF64}(x))
end

# KrylovKit applies a linear map either as a matrix or by calling it.
(A::RowMajorOperator)(x) = A * x

"""Use the row-by-row form for sparse generators; leave anything else alone."""
_matvec_operator(L::SparseMatrixCSC) = RowMajorOperator(sparse(transpose(L)))
_matvec_operator(L) = L

_sparse_complex(A::SparseMatrixCSC{ComplexF64}) = A
_sparse_complex(A) = SparseMatrixCSC{ComplexF64, Int64}(sparse(ComplexF64.(A)))

"""Append the entries of `scale * kron(A, B)` to the triplet arrays."""
function _append_kron!(
    rows::Vector{Int64},
    cols::Vector{Int64},
    vals::Vector{ComplexF64},
    A::SparseMatrixCSC{ComplexF64},
    B::SparseMatrixCSC{ComplexF64},
    scale::ComplexF64,
)
    rows_B, cols_B = size(B)
    stored_A, values_A = rowvals(A), nonzeros(A)
    stored_B, values_B = rowvals(B), nonzeros(B)
    @inbounds for col_a in 1:size(A, 2), k_a in nzrange(A, col_a)
        a = scale * values_A[k_a]
        iszero(a) && continue
        row_offset = (stored_A[k_a] - 1) * rows_B
        col_offset = (col_a - 1) * cols_B
        for col_b in 1:cols_B, k_b in nzrange(B, col_b)
            push!(rows, row_offset + stored_B[k_b])
            push!(cols, col_offset + col_b)
            push!(vals, a * values_B[k_b])
        end
    end
    return nothing
end

function _liouvillian_triplets(H, c_ops)
    Hs = _sparse_complex(H)
    d = size(Hs, 1)
    size(Hs, 2) == d || throw(DimensionMismatch("Hamiltonian must be square."))
    collapse = [_sparse_complex(c) for c in c_ops]
    for c in collapse
        size(c) == (d, d) || throw(DimensionMismatch("collapse operators must match H."))
    end

    decay = spzeros(ComplexF64, d, d)
    for c in collapse
        decay += c' * c
    end
    left = Hs - 0.5im * decay
    right = SparseMatrixCSC{ComplexF64, Int64}(transpose(Hs + 0.5im * decay))
    identity = SparseMatrixCSC{ComplexF64, Int64}(I, d, d)

    expected = 2 * d * (nnz(left) + nnz(right)) + sum((nnz(c)^2 for c in collapse); init = 0)
    rows = sizehint!(Int64[], expected)
    cols = sizehint!(Int64[], expected)
    vals = sizehint!(ComplexF64[], expected)
    _append_kron!(rows, cols, vals, identity, left, ComplexF64(0, -1))
    _append_kron!(rows, cols, vals, right, identity, ComplexF64(0, 1))
    for c in collapse
        _append_kron!(rows, cols, vals, conj(c), c, ComplexF64(1))
    end
    return rows, cols, vals, d * d
end

"""
Sparse Lindblad generator `L` acting on column-stacked density matrices.
"""
function liouvillian(H::AbstractMatrix, c_ops::Vector)
    rows, cols, vals, n = _liouvillian_triplets(H, c_ops)
    return sparse(rows, cols, vals, n, n)
end

"""
The Lindblad generator in the form used to propagate states: assembled
directly as its transpose and applied row by row.
"""
function liouvillian_operator(H::AbstractMatrix, c_ops::Vector)
    rows, cols, vals, n = _liouvillian_triplets(H, c_ops)
    return RowMajorOperator(sparse(cols, rows, vals, n, n))
end
