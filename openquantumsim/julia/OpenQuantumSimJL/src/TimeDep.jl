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

function _as_complex64(value)::ComplexF64
    value isa Number && return ComplexF64(value)
    for (pkgid, mod) in Base.loaded_modules
        if string(pkgid.name) == "PythonCall" && isdefined(mod, :pyconvert)
            return getproperty(mod, :pyconvert)(ComplexF64, value)
        end
    end
    return ComplexF64(value)
end

function _coefficient_value(coefficient, t::Real)::ComplexF64
    coefficient isa Number && return ComplexF64(coefficient)
    value = coefficient(Float64(t))
    return _as_complex64(value)
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

# For large problems whose operators are mostly zero, the master equation is
# integrated as a sparse superoperator sum L0 + sum_i f_i(t) K_i, applied
# row by row; each right-hand side then costs O(nnz) instead of the O(d^3)
# dense matrix products used below the thresholds.
const TIMEDEP_SUPEROPERATOR_MIN_DIM = 32
const TIMEDEP_SUPEROPERATOR_MAX_DENSITY = 0.1

struct _TimeDependentGenerator{C}
    static::RowMajorOperator
    terms::Vector{RowMajorOperator}
    coefficients::C
end

function _lindblad_rhs_superoperator!(du, u, generator::_TimeDependentGenerator, t)
    mul!(du, generator.static, u)
    for idx in eachindex(generator.terms)
        value = _coefficient_value(generator.coefficients[idx], t)
        iszero(value) && continue
        mul!(du, generator.terms[idx], u, value, true)
    end
    return nothing
end

function _use_superoperator_form(d::Integer, operators)
    d >= TIMEDEP_SUPEROPERATOR_MIN_DIM || return false
    limit = TIMEDEP_SUPEROPERATOR_MAX_DENSITY * d^2
    return all(count(!iszero, op) <= limit for op in operators)
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
    ode_solver::AbstractString = "auto",
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

    u0 = vec(copy(rho0c))
    superoperator = _use_superoperator_form(d, vcat([H0c], h_terms, collapse_ops))
    if superoperator
        no_collapse = SparseMatrixCSC{ComplexF64, Int64}[]
        rhs! = _lindblad_rhs_superoperator!
        params = _TimeDependentGenerator(
            liouvillian_operator(sparse(H0c), [sparse(c) for c in collapse_ops]),
            [liouvillian_operator(sparse(op), no_collapse) for op in h_terms],
            coeffs,
        )
    else
        cdgc_ops = [c' * c for c in collapse_ops]
        rhs! = _lindblad_rhs_time_dependent!
        params = (
            H0c,
            h_terms,
            coeffs,
            collapse_ops,
            cdgc_ops,
            similar(H0c),
            similar(H0c),
            similar(H0c),
            d,
        )
    end

    local sol_u, report
    elapsed = @elapsed begin
        sol_u, report = _solve_ode_at_times(
            rhs!,
            u0,
            times,
            params;
            rtol = rtol,
            atol = atol,
            ode_solver = ode_solver,
        )
    end

    collected = _collect_solution(
        sol_u,
        times,
        d,
        expectation_ops;
        save_states = save_states,
        compute_entropy = compute_entropy,
        elapsed = elapsed,
        report = report,
        method_name = "time-dependent-ode",
        requested_method = "time-dependent-ode",
    )
    return merge(
        collected,
        (
            solver_stats = merge(
                collected.solver_stats,
                (time_dependent = true, superoperator = superoperator),
            ),
        ),
    )
end
