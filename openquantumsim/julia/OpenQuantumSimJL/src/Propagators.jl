function expv_taylor!(
    out::AbstractVector{ComplexF64},
    A,
    psi::AbstractVector{ComplexF64},
    alpha::ComplexF64,
    buf1::AbstractVector{ComplexF64},
    buf2::AbstractVector{ComplexF64},
    order::Int = 5,
)
    copyto!(out, psi)
    copyto!(buf1, psi)
    for k in 1:order
        mul!(buf2, A, buf1)
        buf2 .*= alpha / k
        out .+= buf2
        buf1, buf2 = buf2, buf1
    end
    return out
end

function krylov_expmv(
    A,
    dt::Real,
    vector::AbstractVector{ComplexF64};
    krylov_dim::Integer = 30,
    tol::Real = 1e-10,
)::Vector{ComplexF64}
    krylov_dim > 0 || throw(ArgumentError("krylov_dim must be positive."))
    if dt == 0
        return copy(vector)
    end
    evolved, _info = exponentiate(
        A,
        Float64(dt),
        vector;
        krylovdim = Int(krylov_dim),
        tol = Float64(tol),
    )
    return Vector{ComplexF64}(evolved)
end
