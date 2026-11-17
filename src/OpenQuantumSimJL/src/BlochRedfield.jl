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

"""
Group the transition pairs `(m, n)` by their Bohr frequency `E_m - E_n`.

The frequencies are sorted once, and a group collects every frequency within
`tol` of the group's first one, so this costs O(d^2 log d) instead of
comparing each pair with every group found so far.
"""
function _br_channel_groups(energies::Vector{Float64})::Vector{Vector{Tuple{Int,Int}}}
    d = length(energies)
    pairs = Tuple{Int,Int}[(m, n) for n in 1:d for m in 1:d]
    omegas = Float64[energies[m] - energies[n] for (m, n) in pairs]
    tol = 1e-10 * max(1.0, maximum(abs, energies) + abs(minimum(energies)))
    groups = Vector{Tuple{Int,Int}}[]
    group_start = 0.0
    for idx in sortperm(omegas)
        if isempty(groups) || omegas[idx] - group_start > tol
            push!(groups, Tuple{Int,Int}[])
            group_start = omegas[idx]
        end
        push!(groups[end], pairs[idx])
    end
    return groups
end

"""Energies and eigenvectors of a Hermitian Hamiltonian."""
function _br_eigenbasis(H::AbstractMatrix)
    H_dense = Matrix{ComplexF64}(H)
    norm(H_dense - H_dense') <= 1e-10 * max(1.0, norm(H_dense)) ||
        throw(ArgumentError("Hamiltonian must be Hermitian."))
    factorization = eigen(Hermitian(H_dense))
    return Float64.(factorization.values), Matrix{ComplexF64}(factorization.vectors)
end

"""
Secular Bloch-Redfield generator in the energy eigenbasis, as a sparse
operator applied row by row.

In the eigenbasis the coherent part is diagonal, `-i (E_m - E_n)` on
`rho_mn`, and each Bohr-frequency channel `A(omega)` is a sparse matrix with
one entry per transition in the group. The generator therefore has O(d^2)
entries for a nondegenerate spectrum instead of the d^4 of a dense matrix.
"""
function _br_eigenbasis_generator(energies::Vector{Float64}, U::Matrix{ComplexF64}, a_ops, spectra)
    d = length(energies)
    length(a_ops) == length(spectra) ||
        throw(DimensionMismatch("a_ops and spectra must have equal length."))
    rows = Int64[]
    cols = Int64[]
    vals = ComplexF64[]
    for n in 1:d, m in 1:d
        index = (n - 1) * d + m
        push!(rows, index)
        push!(cols, index)
        push!(vals, -1im * (energies[m] - energies[n]))
    end

    identity = SparseMatrixCSC{ComplexF64, Int64}(I, d, d)
    groups = _br_channel_groups(energies)
    for (a_index, a_raw) in enumerate(a_ops)
        A = Matrix{ComplexF64}(a_raw)
        size(A) == (d, d) || throw(DimensionMismatch("coupling operators must match H."))
        B = U' * A * U
        for group in groups
            m1, n1 = group[1]
            gamma = _br_spectrum_value(spectra[a_index], energies[m1] - energies[n1])
            iszero(gamma) && continue
            channel = dropzeros!(sparse(
                Int[m for (m, _) in group],
                Int[n for (_, n) in group],
                ComplexF64[B[m, n] for (m, n) in group],
                d,
                d,
            ))
            nnz(channel) == 0 && continue
            decay = channel' * channel
            _append_kron!(rows, cols, vals, conj(channel), channel, ComplexF64(gamma))
            _append_kron!(rows, cols, vals, identity, decay, ComplexF64(-gamma / 2))
            _append_kron!(
                rows,
                cols,
                vals,
                SparseMatrixCSC{ComplexF64, Int64}(transpose(decay)),
                identity,
                ComplexF64(-gamma / 2),
            )
        end
    end
    return RowMajorOperator(sparse(cols, rows, vals, d * d, d * d))
end

"""
Secular Bloch-Redfield generator for `(operator, spectrum)` couplings, in the
computational basis, as a dense matrix acting on column-stacked states.

Each spectrum maps a Bohr frequency to the dissipator weight gamma. A
constant gamma g matches a Lindblad collapse operator sqrt(g) * A only when A
connects a single Bohr frequency; otherwise the secular approximation drops
the cross terms between frequencies.
The secular approximation groups transitions with equal Bohr frequencies into
single channels. `brmesolve` does not use this matrix: it propagates in the
eigenbasis, which is far cheaper for anything but small systems.
"""
function br_liouvillian(H::AbstractMatrix{ComplexF64}, a_ops, spectra)
    size(H, 1) == size(H, 2) || throw(DimensionMismatch("Hamiltonian must be square."))
    energies, U = _br_eigenbasis(H)
    eigenbasis = Matrix(sparse(_br_eigenbasis_generator(energies, U, a_ops, spectra)))
    # vec(U rho U') = (conj(U) (x) U) vec(rho).
    T = kron(conj(U), U)
    return T * eigenbasis * T'
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
    ode_solver::AbstractString = "auto",
)
    Hc = _as_complex_operator(H)
    rho0c = _as_complex_matrix(rho0)
    times = Float64.(collect(tlist))
    d = _validate_mesolve_inputs(Hc, rho0c, times)

    expectation_ops = _as_complex_operators(e_ops)
    for op in expectation_ops
        size(op) == (d, d) || throw(DimensionMismatch("expectation operators must match H."))
    end

    requested_method, method_name, ode_solver_name = _resolve_method(method, ode_solver)

    # Propagate in the energy eigenbasis, where the secular generator is
    # sparse, and rotate only the outputs. Trace, Hermiticity, eigenvalues and
    # entropy do not depend on the basis, so the health checks and entropy are
    # computed on the eigenbasis states directly.
    energies, U = _br_eigenbasis(Hc)
    L = _br_eigenbasis_generator(energies, U, a_ops, spectra)
    u0 = vec(U' * rho0c * U)
    eigenbasis_e_ops = Matrix{ComplexF64}[U' * Matrix(op) * U for op in expectation_ops]
    sol_u, report, elapsed = _solve_liouvillian(
        L,
        u0,
        times;
        rtol = Float64(rtol),
        atol = Float64(atol),
        method_name = method_name,
        krylov_dim = Int(krylov_dim),
        ode_solver = ode_solver_name,
    )

    collected = _collect_solution(
        sol_u,
        times,
        d,
        eigenbasis_e_ops;
        save_states = save_states,
        compute_entropy = compute_entropy,
        elapsed = elapsed,
        report = report,
        method_name = method_name,
        requested_method = requested_method,
        krylov_dim = Int(krylov_dim),
    )
    states = collected.states
    for idx in axes(states, 3)
        states[:, :, idx] .= U * states[:, :, idx] * U'
    end
    return collected
end
