const SparseOp = SparseMatrixCSC{ComplexF64, Int64}

_dim(space_or_dim::Integer)::Int = Int(space_or_dim)
_dim(space::AbstractHilbertSpace)::Int = dim(space)

function sparse_from_csc(nrows::Integer, ncols::Integer, colptr, rowval, nzval)::SparseOp
    return SparseMatrixCSC{ComplexF64, Int64}(
        Int(nrows),
        Int(ncols),
        Int64.(collect(colptr)),
        Int64.(collect(rowval)),
        ComplexF64.(collect(nzval)),
    )
end

function eye(space_or_dim)::SparseOp
    d = _dim(space_or_dim)
    return spdiagm(0 => ones(ComplexF64, d))
end

function destroy(space_or_dim)::SparseOp
    d = _dim(space_or_dim)
    d > 0 || throw(ArgumentError("dimension must be positive."))
    vals = ComplexF64.(sqrt.(1:(d - 1)))
    return spdiagm(d, d, 1 => vals)
end

create(space_or_dim)::SparseOp = destroy(space_or_dim)'

function num(space_or_dim)::SparseOp
    d = _dim(space_or_dim)
    return spdiagm(0 => ComplexF64.(0:(d - 1)))
end

function _check_qubit(space_or_dim)
    d = isnothing(space_or_dim) ? 2 : _dim(space_or_dim)
    d == 2 || throw(ArgumentError("Pauli operators require dimension 2."))
    return d
end

function sigmax(space_or_dim=nothing)::SparseOp
    _check_qubit(space_or_dim)
    return sparse(ComplexF64[0 1; 1 0])
end

function sigmay(space_or_dim=nothing)::SparseOp
    _check_qubit(space_or_dim)
    return sparse(ComplexF64[0 -im; im 0])
end

function sigmaz(space_or_dim=nothing)::SparseOp
    _check_qubit(space_or_dim)
    return sparse(ComplexF64[1 0; 0 -1])
end

function sigmam(space_or_dim=nothing)::SparseOp
    _check_qubit(space_or_dim)
    return sparse(ComplexF64[0 0; 1 0])
end

sigmap(space_or_dim=nothing)::SparseOp = sigmam(space_or_dim)'

function collective_lowering(space_or_dim)::SparseOp
    d = _dim(space_or_dim)
    d > 0 || throw(ArgumentError("dimension must be positive."))
    spin = (d - 1) / 2
    m_values = spin .- collect(0:(d - 2))
    coeffs = ComplexF64.(sqrt.(spin * (spin + 1) .- m_values .* (m_values .- 1)))
    return spdiagm(d, d, -1 => coeffs)
end

collective_raising(space_or_dim)::SparseOp = collective_lowering(space_or_dim)'

function collective_x(space_or_dim)::SparseOp
    jm = collective_lowering(space_or_dim)
    return (jm + jm') / 2
end

function collective_z(space_or_dim)::SparseOp
    d = _dim(space_or_dim)
    spin = (d - 1) / 2
    return spdiagm(0 => ComplexF64.(spin .- collect(0:(d - 1))))
end

function collective_excitation(space_or_dim)::SparseOp
    d = _dim(space_or_dim)
    return spdiagm(0 => ComplexF64.(collect((d - 1):-1:0)))
end

dicke_jm(space_or_dim)::SparseOp = collective_lowering(space_or_dim)
dicke_jp(space_or_dim)::SparseOp = collective_raising(space_or_dim)
dicke_jx(space_or_dim)::SparseOp = collective_x(space_or_dim)
dicke_jz(space_or_dim)::SparseOp = collective_z(space_or_dim)
dicke_excitation(space_or_dim)::SparseOp = collective_excitation(space_or_dim)

function tensor(ops::SparseOp...)::SparseOp
    length(ops) > 0 || throw(ArgumentError("tensor requires at least one operator."))
    return kron(ops...)
end
