struct InterpolatedCoefficient
    times::Vector{Float64}
    values::Vector{ComplexF64}

    function InterpolatedCoefficient(times, values)
        time_values = Float64.(collect(times))
        coeff_values = ComplexF64.(collect(values))
        length(time_values) > 0 || throw(ArgumentError("times and values must not be empty."))
        length(time_values) == length(coeff_values) ||
            throw(DimensionMismatch("times and values must have the same length."))
        if length(time_values) > 1
            all(diff(time_values) .> 0) ||
                throw(ArgumentError("times must be sorted in strictly ascending order."))
        end
        return new(time_values, coeff_values)
    end
end

function (coefficient::InterpolatedCoefficient)(t::Real)::ComplexF64
    length(coefficient.times) == 1 && return coefficient.values[1]
    t_value = Float64(t)
    t_value <= coefficient.times[1] && return coefficient.values[1]
    t_value >= coefficient.times[end] && return coefficient.values[end]

    idx = searchsortedlast(coefficient.times, t_value)
    t0 = coefficient.times[idx]
    t1 = coefficient.times[idx + 1]
    weight = (t_value - t0) / (t1 - t0)
    return (1 - weight) * coefficient.values[idx] + weight * coefficient.values[idx + 1]
end

function _coefficient_value(coefficient, t::Real)::ComplexF64
    coefficient isa Number && return ComplexF64(coefficient)
    value = coefficient(Float64(t))
    value isa Number && return ComplexF64(value)
    return pyconvert(ComplexF64, value)
end

function _hamiltonian_at!(
    out::Matrix{ComplexF64},
    base::Matrix{ComplexF64},
    term_ops::Vector{Matrix{ComplexF64}},
    coefficients,
    t::Real,
)
    copyto!(out, base)
    for idx in eachindex(term_ops)
        value = _coefficient_value(coefficients[idx], t)
        out .+= value .* term_ops[idx]
    end
    return out
end

function _lindblad_rhs_time_dependent!(
    du,
    u,
    params,
    t,
)
    H0, term_ops, coefficients, collapse_ops, cdgc_ops, Hbuf, leftbuf, rightbuf, d = params
    _hamiltonian_at!(Hbuf, H0, term_ops, coefficients, t)

    rho = reshape(u, d, d)
    drho = reshape(du, d, d)

    mul!(leftbuf, Hbuf, rho)
    mul!(rightbuf, rho, Hbuf)
    drho .= -1im .* (leftbuf .- rightbuf)

    for idx in eachindex(collapse_ops)
        c = collapse_ops[idx]
        cdgc = cdgc_ops[idx]

        mul!(leftbuf, c, rho)
        mul!(rightbuf, leftbuf, c')
        drho .+= rightbuf

        mul!(leftbuf, cdgc, rho)
        drho .-= 0.5 .* leftbuf

        mul!(leftbuf, rho, cdgc)
        drho .-= 0.5 .* leftbuf
    end

    return nothing
end

function mesolve_time_dependent(
    H0,
    term_ops,
    coefficients,
    rho0,
    tlist,
    c_ops = Matrix{ComplexF64}[],
    e_ops = Matrix{ComplexF64}[],
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    save_states::Bool = false,
    compute_entropy::Bool = true,
)
    H0c = _as_complex_matrix(H0)
    rho0c = _as_complex_matrix(rho0)
    times = Float64.(collect(tlist))
    d = _validate_mesolve_inputs(H0c, rho0c, times)

    h_terms = _as_complex_matrices(term_ops)
    coeffs = collect(coefficients)
    length(h_terms) == length(coeffs) ||
        throw(DimensionMismatch("term operators and coefficients must have the same length."))
    for op in h_terms
        size(op) == (d, d) || throw(DimensionMismatch("Hamiltonian terms must match H0."))
    end

    collapse_ops = _as_complex_matrices(c_ops)
    expectation_ops = _as_complex_matrices(e_ops)
    for c in collapse_ops
        size(c) == (d, d) || throw(DimensionMismatch("collapse operators must match H0."))
    end
    for op in expectation_ops
        size(op) == (d, d) || throw(DimensionMismatch("expectation operators must match H0."))
    end

    cdgc_ops = [c' * c for c in collapse_ops]
    u0 = vec(copy(rho0c))

    Hbuf = similar(H0c)
    leftbuf = similar(H0c)
    rightbuf = similar(H0c)
    params = (H0c, h_terms, coeffs, collapse_ops, cdgc_ops, Hbuf, leftbuf, rightbuf, d)

    elapsed = @elapsed begin
        if length(times) == 1
            sol_u = [u0]
            retcode = "Success"
            nfev = 0
        else
            problem = ODEProblem(
                _lindblad_rhs_time_dependent!,
                u0,
                (times[1], times[end]),
                params,
            )
            sol = solve(
                problem,
                Tsit5();
                saveat = times,
                reltol = Float64(rtol),
                abstol = Float64(atol),
            )
            sol_u = sol.u
            retcode = string(sol.retcode)
            nfev = 0
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
            nfev = nfev,
            wall_time = elapsed,
            retcode = retcode,
            method = "time-dependent-ode",
            time_dependent = true,
            compute_entropy = compute_entropy,
        ),
    )
end
