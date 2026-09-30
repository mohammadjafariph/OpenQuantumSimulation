function _br_spectrum_value(spectrum, omega::Real)::Float64
    value = spectrum(Float64(omega))
    value isa Real && return Float64(value)
    for (pkgid, mod) in Base.loaded_modules
        if string(pkgid.name) == "PythonCall" && isdefined(mod, :pyconvert)
            return getproperty(mod, :pyconvert)(Float64, value)
        end
    end
    return Float64(value)
end

function _br_channel_groups(energies::Vector{Float64})::Vector{Vector{Tuple{Int,Int}}}
    """Group eigenvalue-transition pairs by their Bohr frequency."""
    groups = Vector{Tuple{Int,Int}}[]
    omegas = Float64[]
    tol = 1e-10 * max(1.0, maximum(abs, energies) + abs(minimum(energies)))
    for m in eachindex(energies), n in eachindex(energies)
        omega = energies[m] - energies[n]
        placed = false
        for idx in eachindex(omegas)
            if abs(omega - omegas[idx]) <= tol
                push!(groups[idx], (m, n))
                placed = true
                break
            end
        end
        if !placed
            push!(groups, [(m, n)])
            push!(omegas, omega)
        end
    end
    return groups
end

function br_liouvillian(H::AbstractMatrix{ComplexF64}, a_ops, spectra)
    """Secular Bloch-Redfield generator for `(operator, spectrum)` couplings.

    Each spectrum maps a Bohr frequency to the dissipator weight gamma;
    a constant gamma g is equivalent to a Lindblad collapse operator
    sqrt(g) * A. The secular approximation groups transitions with equal
    Bohr frequencies into single channels.
    """
    d = size(H, 1)
    size(H, 2) == d || throw(DimensionMismatch("Hamiltonian must be square."))
    H_dense = Matrix{ComplexF64}(H)
    norm(H_dense - H_dense') <= 1e-10 * max(1.0, norm(H_dense)) ||
        throw(ArgumentError("Hamiltonian must be Hermitian."))

    factorization = eigen(Hermitian(H_dense))
    energies = Float64.(factorization.values)
    U = Matrix{ComplexF64}(factorization.vectors)

    identity = Matrix{ComplexF64}(I, d, d)
    length(a_ops) == length(spectra) ||
        throw(DimensionMismatch("a_ops and spectra must have equal length."))
    L = zeros(ComplexF64, d * d, d * d)
    for (a_index, a_raw) in enumerate(a_ops)
        spectrum = spectra[a_index]
        A = Matrix{ComplexF64}(a_raw)
        size(A) == (d, d) ||
            throw(DimensionMismatch("coupling operators must match H."))
        B = U' * A * U
        for group in _br_channel_groups(energies)
            channel = zeros(ComplexF64, d, d)
            for (m, n) in group
                channel[m, n] = B[m, n]
            end
            gamma = _br_spectrum_value(spectrum, energies[group[1][1]] - energies[group[1][2]])
            iszero(gamma) && continue
            cdgc = channel' * channel
            L += gamma .* (
                kron(conj(channel), channel) .-
                0.5 .* kron(identity, cdgc) .-
                0.5 .* kron(transpose(cdgc), identity)
            )
        end
    end

    # Transform the eigenbasis generator to the computational basis:
    # vec(U rho U') = (conj(U) (x) U) vec(rho).
    T = kron(conj(U), U)
    L = T * L * T'
    L += -1im .* (kron(identity, H) .- kron(transpose(H), identity))
    return L
end

function brmesolve(
    H,
    rho0,
    tlist,
    a_ops,
    spectra,
    e_ops = Matrix{ComplexF64}[],
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    save_states::Bool = false,
    method::AbstractString = "ode",
    krylov_dim::Integer = 30,
    compute_entropy::Bool = true,
)
    Hc = _as_complex_operator(H)
    rho0c = _as_complex_matrix(rho0)
    times = Float64.(collect(tlist))
    d = _validate_mesolve_inputs(Hc, rho0c, times)

    expectation_ops = _as_complex_operators(e_ops)
    for op in expectation_ops
        size(op) == (d, d) || throw(DimensionMismatch("expectation operators must match H."))
    end

    requested_method = lowercase(String(method))
    method_name = requested_method == "auto" ? "ode" : requested_method

    L = br_liouvillian(Hc, a_ops, spectra)
    u0 = vec(copy(rho0c))
    sol_u, retcode, elapsed = _solve_liouvillian(
        L,
        u0,
        times;
        rtol = Float64(rtol),
        atol = Float64(atol),
        method_name = method_name,
        krylov_dim = Int(krylov_dim),
    )

    return _collect_solution(
        sol_u,
        times,
        d,
        expectation_ops;
        save_states = save_states,
        compute_entropy = compute_entropy,
        elapsed = elapsed,
        retcode = retcode,
        method_name = method_name,
        requested_method = requested_method,
        krylov_dim = Int(krylov_dim),
    )
end
