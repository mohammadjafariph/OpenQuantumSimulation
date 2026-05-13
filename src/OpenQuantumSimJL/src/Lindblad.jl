function liouvillian(H::AbstractMatrix{ComplexF64}, c_ops::Vector)
    d = size(H, 1)
    size(H, 2) == d || throw(DimensionMismatch("Hamiltonian must be square."))
    I_d = eye(d)
    L = -1im * (kron(I_d, H) - kron(transpose(H), I_d))
    for c in c_ops
        cdgc = c' * c
        L += kron(conj(c), c)
        L -= 0.5 * kron(I_d, cdgc)
        L -= 0.5 * kron(transpose(cdgc), I_d)
    end
    return sparse(L)
end

function _as_complex_matrix(matrix)::Matrix{ComplexF64}
    return Matrix{ComplexF64}(ComplexF64.(matrix))
end

function _as_complex_operator(matrix)::SparseMatrixCSC{ComplexF64, Int64}
    return dropzeros!(sparse(ComplexF64.(matrix)))
end

function _as_complex_matrices(matrices)::Vector{Matrix{ComplexF64}}
    converted = Matrix{ComplexF64}[]
    for matrix in matrices
        push!(converted, _as_complex_matrix(matrix))
    end
    return converted
end

function _as_complex_operators(matrices)::Vector{SparseMatrixCSC{ComplexF64, Int64}}
    converted = SparseMatrixCSC{ComplexF64, Int64}[]
    for matrix in matrices
        push!(converted, _as_complex_operator(matrix))
    end
    return converted
end

function _validate_mesolve_inputs(
    H::AbstractMatrix{ComplexF64},
    rho0::Matrix{ComplexF64},
    times::Vector{Float64},
)
    d = size(H, 1)
    size(H, 2) == d || throw(DimensionMismatch("Hamiltonian must be square."))
    size(rho0) == (d, d) || throw(DimensionMismatch("rho0 must have the same dimension as H."))
    length(times) > 0 || throw(ArgumentError("tlist must contain at least one time."))
    all(diff(times) .>= 0) || throw(ArgumentError("tlist must be sorted in ascending order."))
    return d
end

function mesolve(
    H,
    rho0,
    tlist,
    c_ops = Matrix{ComplexF64}[],
    e_ops = Matrix{ComplexF64}[],
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    save_states::Bool = false,
    method::AbstractString = "auto",
    krylov_dim::Integer = 30,
    compute_entropy::Bool = true,
)
    Hc = _as_complex_operator(H)
    rho0c = _as_complex_matrix(rho0)
    times = Float64.(collect(tlist))
    d = _validate_mesolve_inputs(Hc, rho0c, times)

    collapse_ops = _as_complex_operators(c_ops)
    expectation_ops = _as_complex_operators(e_ops)
    for c in collapse_ops
        size(c) == (d, d) || throw(DimensionMismatch("collapse operators must match H."))
    end
    for op in expectation_ops
        size(op) == (d, d) || throw(DimensionMismatch("expectation operators must match H."))
    end

    L = liouvillian(Hc, collapse_ops)
    u0 = vec(copy(rho0c))
    requested_method = lowercase(String(method))
    method_name = requested_method == "auto" ? "ode" : requested_method

    elapsed = @elapsed begin
        if length(times) == 1
            sol_u = [u0]
            retcode = "Success"
        elseif method_name == "krylov"
            sol_u = Vector{Vector{ComplexF64}}(undef, length(times))
            sol_u[1] = u0
            for idx in 2:length(times)
                dt = times[idx] - times[idx - 1]
                sol_u[idx] = krylov_expmv(
                    L,
                    dt,
                    sol_u[idx - 1];
                    krylov_dim = krylov_dim,
                    tol = max(Float64(rtol), Float64(atol)),
                )
            end
            retcode = "Success"
        elseif method_name in ("ode", "tsit5")
            rhs!(du, u, _p, _t) = (mul!(du, L, u); nothing)
            problem = ODEProblem(rhs!, u0, (times[1], times[end]))
            sol = solve(
                problem,
                Tsit5();
                saveat = times,
                reltol = Float64(rtol),
                abstol = Float64(atol),
            )
            sol_u = sol.u
            retcode = string(sol.retcode)
        else
            throw(ArgumentError("unknown mesolve method: $(method)"))
        end
    end

    n_times = length(times)
    n_expect = length(expectation_ops)
    expect_values = zeros(ComplexF64, n_expect, n_times)
    entropy_values = compute_entropy ? zeros(Float64, n_times) : Float64[]
    states = Array{ComplexF64}(undef, d, d, save_states ? n_times : 0)

    for idx in 1:n_times
        rho = reshape(sol_u[idx], d, d)
        if compute_entropy
            rho_h = Hermitian((rho .+ rho') ./ 2)
            entropy_values[idx] = von_neumann_entropy(rho_h)
        end
        for op_idx in 1:n_expect
            expect_values[op_idx, idx] = expect(expectation_ops[op_idx], rho)
        end
        if save_states
            states[:, :, idx] .= rho
        end
    end

    return (
        times = times,
        states = states,
        expect = expect_values,
        entropy = entropy_values,
        solver_stats = (
            nsteps = length(sol_u) - 1,
            nfev = 0,
            wall_time = elapsed,
            retcode = retcode,
            method = method_name,
            requested_method = requested_method,
            krylov_dim = Int(krylov_dim),
            compute_entropy = compute_entropy,
        ),
    )
end
