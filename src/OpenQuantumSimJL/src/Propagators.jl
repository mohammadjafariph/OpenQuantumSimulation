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

"""
Apply `exp(A * dt)` to `vector` and report whether the Krylov iteration
converged, its residual estimate, and the number of operator applications.
"""
function krylov_expmv_info(
    A,
    dt::Real,
    vector::AbstractVector{ComplexF64};
    krylov_dim::Integer = 30,
    tol::Real = 1e-10,
)
    krylov_dim > 0 || throw(ArgumentError("krylov_dim must be positive."))
    if dt == 0
        return (
            vector = Vector{ComplexF64}(vector),
            converged = true,
            residual = 0.0,
            numops = 0,
        )
    end
    evolved, info = exponentiate(
        A,
        Float64(dt),
        vector;
        krylovdim = Int(krylov_dim),
        tol = Float64(tol),
        # Non-convergence is reported through the returned flag instead of a
        # log message, so callers decide whether to raise or warn.
        verbosity = 0,
    )
    return (
        vector = Vector{ComplexF64}(evolved),
        converged = info.converged > 0,
        residual = Float64(info.normres),
        numops = Int(info.numops),
    )
end

function krylov_expmv(
    A,
    dt::Real,
    vector::AbstractVector{ComplexF64};
    krylov_dim::Integer = 30,
    tol::Real = 1e-10,
)::Vector{ComplexF64}
    return krylov_expmv_info(A, dt, vector; krylov_dim = krylov_dim, tol = tol).vector
end
