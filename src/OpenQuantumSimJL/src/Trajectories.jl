@inline function sample3(w1::Float64, w2::Float64, w3::Float64)::Int
    r = rand() * (w1 + w2 + w3)
    r < w1 && return 1
    r < w1 + w2 && return 2
    return 3
end

mutable struct OQSRng
    state::UInt64
end

function OQSRng(seed::Integer)
    state = UInt64(seed)
    state == 0 && (state = 0x9e3779b97f4a7c15)
    return OQSRng(state)
end

@inline function _rand!(rng::OQSRng)::Float64
    rng.state = rng.state * 0x5851f42d4c957f2d + 0x14057b7ef767814f
    return Float64(rng.state >> 11) * (1.0 / 9007199254740992.0)
end

function _as_complex_vector(vector)::Vector{ComplexF64}
    return Vector{ComplexF64}(ComplexF64.(vector))
end

function _validate_mcsolve_inputs(
    H::Matrix{ComplexF64},
    psi0::Vector{ComplexF64},
    times::Vector{Float64},
    n_traj::Int,
    max_step::Float64,
    n_jobs::Int = 1,
)
    d = size(H, 1)
    size(H, 2) == d || throw(DimensionMismatch("Hamiltonian must be square."))
    length(psi0) == d || throw(DimensionMismatch("psi0 length must match H."))
    length(times) > 0 || throw(ArgumentError("tlist must contain at least one time."))
    all(diff(times) .>= 0) || throw(ArgumentError("tlist must be sorted in ascending order."))
    n_traj > 0 || throw(ArgumentError("n_traj must be positive."))
    max_step > 0 || throw(ArgumentError("max_step must be positive."))
    (n_jobs == -1 || n_jobs > 0) || throw(ArgumentError("n_jobs must be -1 or positive."))
    return d
end

function _normalize_or_error!(psi::Vector{ComplexF64})
    norm_psi = norm(psi)
    norm_psi > 0 || throw(ArgumentError("state norm became zero during trajectory propagation."))
    psi ./= norm_psi
    return psi
end

function _sample_weighted(rng::OQSRng, weights::Vector{Float64})::Int
    total = sum(weights)
    total > 0 || throw(ArgumentError("cannot sample jump with zero total weight."))
    threshold = _rand!(rng) * total
    accum = 0.0
    for idx in eachindex(weights)
        accum += weights[idx]
        threshold <= accum && return idx
    end
    return length(weights)
end

function _record_expectations!(
    out::Matrix{ComplexF64},
    e_ops::Vector{Matrix{ComplexF64}},
    psi::Vector{ComplexF64},
    time_idx::Int,
)
    for op_idx in eachindex(e_ops)
        out[op_idx, time_idx] += expect(e_ops[op_idx], psi)
    end
    return out
end

function _effective_hamiltonian(
    H::Matrix{ComplexF64},
    c_ops::Vector{Matrix{ComplexF64}},
)
    h_eff = copy(H)
    for c in c_ops
        h_eff .-= 0.5im .* (c' * c)
    end
    return h_eff
end

function _mcwf_step_plans(
    h_eff::Matrix{ComplexF64},
    times::Vector{Float64},
    max_step::Float64,
)
    plans = Vector{NamedTuple{(:n_substeps, :dt, :propagator), Tuple{Int, Float64, Matrix{ComplexF64}}}}(
        undef,
        max(length(times) - 1, 0),
    )
    for time_idx in 2:length(times)
        delta_t = times[time_idx] - times[time_idx - 1]
        n_substeps = max(1, Int(ceil(delta_t / max_step)))
        dt = delta_t / n_substeps
        plans[time_idx - 1] = (
            n_substeps = n_substeps,
            dt = dt,
            propagator = Matrix{ComplexF64}(exp(-1im * h_eff * dt)),
        )
    end
    return plans
end

function _mcwf_step!(
    psi::Vector{ComplexF64},
    propagator::Matrix{ComplexF64},
    c_ops::Vector{Matrix{ComplexF64}},
    dt::Float64,
    rng::OQSRng,
)
    if isempty(c_ops)
        psi .= propagator * psi
        return _normalize_or_error!(psi)
    end

    jump_states = Vector{Vector{ComplexF64}}(undef, length(c_ops))
    weights = zeros(Float64, length(c_ops))
    for op_idx in eachindex(c_ops)
        jump_states[op_idx] = c_ops[op_idx] * psi
        weights[op_idx] = max(real(dot(jump_states[op_idx], jump_states[op_idx])), 0.0)
    end

    total_rate = sum(weights)
    if total_rate > 0 && _rand!(rng) <= min(dt * total_rate, 1.0)
        jump_idx = _sample_weighted(rng, weights)
        psi .= jump_states[jump_idx]
        return _normalize_or_error!(psi)
    end

    psi .= propagator * psi
    return _normalize_or_error!(psi)
end

function _single_trajectory_expectations(
    psi0::Vector{ComplexF64},
    times::Vector{Float64},
    c_ops::Vector{Matrix{ComplexF64}},
    e_ops::Vector{Matrix{ComplexF64}},
    rng::OQSRng,
    step_plans,
    save_states::Bool = false,
)
    d = length(psi0)
    psi = copy(psi0)
    _normalize_or_error!(psi)
    expect_values = zeros(ComplexF64, length(e_ops), length(times))
    states = Array{ComplexF64}(undef, d, save_states ? length(times) : 0)
    _record_expectations!(expect_values, e_ops, psi, 1)
    save_states && (states[:, 1] .= psi)

    for time_idx in 2:length(times)
        plan = step_plans[time_idx - 1]
        for _ in 1:plan.n_substeps
            _mcwf_step!(psi, plan.propagator, c_ops, plan.dt, rng)
        end
        _record_expectations!(expect_values, e_ops, psi, time_idx)
        save_states && (states[:, time_idx] .= psi)
    end

    return expect_values, psi, states
end

const MCSOLVE_CHECKPOINT_FORMAT = "openquantumsim.mcsolve.checkpoint"
const MCSOLVE_CHECKPOINT_VERSION = "1"

function _hash_array!(io::IO, name::AbstractString, array::AbstractArray)
    write(io, name)
    write(io, "\n")
    write(io, string(eltype(array)))
    write(io, "\n")
    write(io, string(size(array)))
    write(io, "\n")
    write(io, reinterpret(UInt8, vec(array)))
    write(io, "\n")
    return io
end

function _mcsolve_config_hash(
    H::Matrix{ComplexF64},
    psi0::Vector{ComplexF64},
    times::Vector{Float64},
    c_ops::Vector{Matrix{ComplexF64}},
    e_ops::Vector{Matrix{ComplexF64}},
    max_step::Float64,
)
    io = IOBuffer()
    write(io, "mcsolve-v1\n")
    write(io, "max_step=$(max_step)\n")
    _hash_array!(io, "H", H)
    _hash_array!(io, "psi0", psi0)
    _hash_array!(io, "times", times)
    write(io, "c_ops=$(length(c_ops))\n")
    for (idx, c) in pairs(c_ops)
        _hash_array!(io, "c_ops[$idx]", c)
    end
    write(io, "e_ops=$(length(e_ops))\n")
    for (idx, op) in pairs(e_ops)
        _hash_array!(io, "e_ops[$idx]", op)
    end
    return bytes2hex(sha256(take!(io)))
end

function _empty_mcsolve_sums(n_ops::Int, n_times::Int)
    return (
        expect_sum = zeros(ComplexF64, n_ops, n_times),
        expect_abs2_sum = zeros(Float64, n_ops, n_times),
    )
end

function _init_mcsolve_checkpoint(
    checkpoint_path::String,
    times::Vector{Float64},
    n_traj::Int,
    seed::Integer,
    max_step::Float64,
    n_ops::Int,
    config_hash::String,
)
    sums = _empty_mcsolve_sums(n_ops, length(times))
    if isempty(checkpoint_path) || !isfile(checkpoint_path)
        base_seed = seed == 0 ? UInt64(time_ns()) : UInt64(seed)
        return (
            completed = 0,
            base_seed = base_seed,
            expect_sum = sums.expect_sum,
            expect_abs2_sum = sums.expect_abs2_sum,
            resumed = false,
            previous_target_n_traj = 0,
        )
    end

    h5open(checkpoint_path, "r") do file
        attrs(file)["format"] == MCSOLVE_CHECKPOINT_FORMAT ||
            throw(ArgumentError("checkpoint file is not an mcsolve checkpoint."))
        attrs(file)["format_version"] == MCSOLVE_CHECKPOINT_VERSION ||
            throw(ArgumentError("unsupported mcsolve checkpoint version."))
        attrs(file)["config_hash"] == config_hash ||
            throw(ArgumentError("checkpoint does not match this mcsolve configuration."))

        stored_times = Vector{Float64}(read(file["times"]))
        stored_times == times ||
            throw(ArgumentError("checkpoint time grid does not match this run."))

        completed = Int(attrs(file)["completed"])
        completed <= n_traj ||
            throw(ArgumentError("checkpoint has more completed trajectories than requested."))
        Int(attrs(file)["n_ops"]) == n_ops ||
            throw(ArgumentError("checkpoint expectation-operator count does not match."))
        Int(attrs(file)["n_times"]) == length(times) ||
            throw(ArgumentError("checkpoint time count does not match."))
        Int(attrs(file)["seed"]) == Int(seed) ||
            throw(ArgumentError("checkpoint seed does not match this run."))
        Float64(attrs(file)["max_step"]) == max_step ||
            throw(ArgumentError("checkpoint max_step does not match this run."))

        expect_sum = Matrix{ComplexF64}(read(file["expect_sum"]))
        expect_abs2_sum = Matrix{Float64}(read(file["expect_abs2_sum"]))
        size(expect_sum) == (n_ops, length(times)) ||
            throw(ArgumentError("checkpoint expectation sum has the wrong shape."))
        size(expect_abs2_sum) == (n_ops, length(times)) ||
            throw(ArgumentError("checkpoint second-moment sum has the wrong shape."))

        return (
            completed = completed,
            base_seed = parse(UInt64, String(attrs(file)["base_seed"])),
            expect_sum = expect_sum,
            expect_abs2_sum = expect_abs2_sum,
            resumed = completed > 0,
            previous_target_n_traj = Int(attrs(file)["n_traj"]),
        )
    end
end

function _write_mcsolve_checkpoint(
    checkpoint_path::String,
    times::Vector{Float64},
    expect_sum::Matrix{ComplexF64},
    expect_abs2_sum::Matrix{Float64},
    completed::Int,
    n_traj::Int,
    seed::Integer,
    base_seed::UInt64,
    max_step::Float64,
    n_jobs::Int,
    n_workers::Int,
    threaded::Bool,
    config_hash::String,
)
    mkpath(dirname(checkpoint_path))
    tmp_path = string(checkpoint_path, ".tmp")
    h5open(tmp_path, "w") do file
        attrs(file)["format"] = MCSOLVE_CHECKPOINT_FORMAT
        attrs(file)["format_version"] = MCSOLVE_CHECKPOINT_VERSION
        attrs(file)["config_hash"] = config_hash
        attrs(file)["completed"] = completed
        attrs(file)["n_traj"] = n_traj
        attrs(file)["seed"] = Int(seed)
        attrs(file)["base_seed"] = string(base_seed)
        attrs(file)["max_step"] = max_step
        attrs(file)["n_jobs_requested"] = n_jobs
        attrs(file)["n_workers"] = n_workers
        attrs(file)["threaded"] = Int(threaded)
        attrs(file)["n_ops"] = size(expect_sum, 1)
        attrs(file)["n_times"] = length(times)
        write(file, "times", times)
        write(file, "expect_sum", expect_sum)
        write(file, "expect_abs2_sum", expect_abs2_sum)
    end
    mv(tmp_path, checkpoint_path; force = true)
    return nothing
end

function _format_duration(seconds::Float64)
    if !isfinite(seconds)
        return "unknown"
    end
    seconds = max(seconds, 0.0)
    if seconds < 60
        return "$(round(seconds; digits = 1))s"
    end
    if seconds < 3600
        minutes = floor(Int, seconds / 60)
        remaining = round(seconds - 60 * minutes; digits = 1)
        return "$(minutes)m $(remaining)s"
    end
    hours = floor(Int, seconds / 3600)
    minutes = floor(Int, (seconds - 3600 * hours) / 60)
    return "$(hours)h $(minutes)m"
end

function _report_mcsolve_progress(
    completed::Int,
    start_completed::Int,
    n_traj::Int,
    start_time::Float64,
    checkpoint_path::String,
    wrote_checkpoint::Bool,
)
    elapsed = max(time() - start_time, eps(Float64))
    new_completed = max(completed - start_completed, 0)
    rate = new_completed / elapsed
    eta = rate > 0 ? (n_traj - completed) / rate : Inf
    percent = 100 * completed / n_traj
    checkpoint_note = wrote_checkpoint ? " checkpoint=$(checkpoint_path)" : ""
    println(
        "mcsolve progress: $(completed)/$(n_traj) " *
        "($(round(percent; digits = 1))%) " *
        "rate=$(round(rate; digits = 2)) traj/s " *
        "elapsed=$(_format_duration(elapsed)) " *
        "eta=$(_format_duration(eta))" *
        checkpoint_note,
    )
    flush(stdout)
    return nothing
end

function single_trajectory(
    H,
    psi0,
    tlist,
    c_ops = Matrix{ComplexF64}[],
    e_ops = Matrix{ComplexF64}[],
    seed::Integer = 0,
    max_step::Real = 1e-2,
    save_states::Bool = false,
)
    Hc = _as_complex_matrix(H)
    psi0c = _as_complex_vector(psi0)
    times = Float64.(collect(tlist))
    max_step_f = Float64(max_step)
    _validate_mcsolve_inputs(Hc, psi0c, times, 1, max_step_f)

    collapse_ops = _as_complex_matrices(c_ops)
    expectation_ops = _as_complex_matrices(e_ops)
    d = size(Hc, 1)
    for c in collapse_ops
        size(c) == (d, d) || throw(DimensionMismatch("collapse operators must match H."))
    end
    for op in expectation_ops
        size(op) == (d, d) || throw(DimensionMismatch("expectation operators must match H."))
    end

    rng = OQSRng(seed)
    h_eff = _effective_hamiltonian(Hc, collapse_ops)
    step_plans = _mcwf_step_plans(h_eff, times, max_step_f)
    expect_values, final_state, states = _single_trajectory_expectations(
        psi0c,
        times,
        collapse_ops,
        expectation_ops,
        rng,
        step_plans,
        save_states,
    )

    return (
        times = times,
        final_state = final_state,
        states = states,
        expect = expect_values,
    )
end

function mcsolve(
    H,
    psi0,
    tlist,
    c_ops = Matrix{ComplexF64}[],
    e_ops = Matrix{ComplexF64}[],
    n_traj::Integer = 500,
    seed::Integer = 0,
    max_step::Real = 1e-2,
    n_jobs::Integer = 1,
    checkpoint_file::AbstractString = "",
    checkpoint_every::Integer = 100,
    progress::Bool = false,
)
    Hc = _as_complex_matrix(H)
    psi0c = _as_complex_vector(psi0)
    times = Float64.(collect(tlist))
    n_traj_i = Int(n_traj)
    max_step_f = Float64(max_step)
    n_jobs_i = Int(n_jobs)
    checkpoint_path = String(strip(String(checkpoint_file)))
    checkpoint_every_i = Int(checkpoint_every)
    checkpoint_every_i > 0 ||
        throw(ArgumentError("checkpoint_every must be positive."))
    progress_enabled = Bool(progress)
    d = _validate_mcsolve_inputs(Hc, psi0c, times, n_traj_i, max_step_f, n_jobs_i)

    collapse_ops = _as_complex_matrices(c_ops)
    expectation_ops = _as_complex_matrices(e_ops)
    for c in collapse_ops
        size(c) == (d, d) || throw(DimensionMismatch("collapse operators must match H."))
    end
    for op in expectation_ops
        size(op) == (d, d) || throw(DimensionMismatch("expectation operators must match H."))
    end

    n_ops = length(expectation_ops)
    n_times = length(times)
    config_hash = _mcsolve_config_hash(
        Hc,
        psi0c,
        times,
        collapse_ops,
        expectation_ops,
        max_step_f,
    )
    h_eff = _effective_hamiltonian(Hc, collapse_ops)
    step_plans = _mcwf_step_plans(h_eff, times, max_step_f)
    use_threads = n_jobs_i != 1 && Threads.nthreads() > 1
    workers_used = use_threads ? Threads.nthreads() : 1
    checkpoint_state = _init_mcsolve_checkpoint(
        checkpoint_path,
        times,
        n_traj_i,
        seed,
        max_step_f,
        n_ops,
        config_hash,
    )
    completed = checkpoint_state.completed
    start_completed = completed
    base_seed = checkpoint_state.base_seed
    expect_sum = checkpoint_state.expect_sum
    expect_abs2_sum = checkpoint_state.expect_abs2_sum

    run_start_time = time()
    if progress_enabled && completed >= n_traj_i
        _report_mcsolve_progress(
            completed,
            start_completed,
            n_traj_i,
            run_start_time,
            checkpoint_path,
            false,
        )
    end

    elapsed = @elapsed begin
        while completed < n_traj_i
            next_traj = completed + 1
            batch_end = isempty(checkpoint_path) && !progress_enabled ?
                n_traj_i :
                min(n_traj_i, completed + checkpoint_every_i)
            batch_count = batch_end - next_traj + 1
            trajectory_expect = Array{ComplexF64}(
                undef,
                n_ops,
                n_times,
                batch_count,
            )

            if use_threads
                Threads.@threads for local_idx in 1:batch_count
                    traj_idx = next_traj + local_idx - 1
                    rng = OQSRng(base_seed + UInt64(traj_idx))
                    expect_values, _, _ = _single_trajectory_expectations(
                        psi0c,
                        times,
                        collapse_ops,
                        expectation_ops,
                        rng,
                        step_plans,
                    )
                    trajectory_expect[:, :, local_idx] .= expect_values
                end
            else
                for local_idx in 1:batch_count
                    traj_idx = next_traj + local_idx - 1
                    rng = OQSRng(base_seed + UInt64(traj_idx))
                    expect_values, _, _ = _single_trajectory_expectations(
                        psi0c,
                        times,
                        collapse_ops,
                        expectation_ops,
                        rng,
                        step_plans,
                    )
                    trajectory_expect[:, :, local_idx] .= expect_values
                end
            end

            for local_idx in 1:batch_count
                values = @view trajectory_expect[:, :, local_idx]
                expect_sum .+= values
                expect_abs2_sum .+= abs2.(values)
            end
            completed = batch_end

            if !isempty(checkpoint_path)
                _write_mcsolve_checkpoint(
                    checkpoint_path,
                    times,
                    expect_sum,
                    expect_abs2_sum,
                    completed,
                    n_traj_i,
                    seed,
                    base_seed,
                    max_step_f,
                    n_jobs_i,
                    workers_used,
                    use_threads,
                    config_hash,
                )
            end

            if progress_enabled
                _report_mcsolve_progress(
                    completed,
                    start_completed,
                    n_traj_i,
                    run_start_time,
                    checkpoint_path,
                    !isempty(checkpoint_path),
                )
            end
        end
    end

    expect_mean = expect_sum ./ n_traj_i
    expect_second_moment = expect_abs2_sum ./ n_traj_i
    expect_variance = max.(expect_second_moment .- abs2.(expect_mean), 0.0)
    expect_std = sqrt.(expect_variance)
    expect_stderr = expect_std ./ sqrt(n_traj_i)
    return (
        times = times,
        states = Array{ComplexF64}(undef, d, d, 0),
        expect = expect_mean,
        expect_std = expect_std,
        expect_stderr = expect_stderr,
        entropy = zeros(Float64, length(times)),
        solver_stats = (
            nsteps = max(length(times) - 1, 0),
            nfev = 0,
            wall_time = elapsed,
            retcode = "Success",
            n_traj = n_traj_i,
            max_step = max_step_f,
            n_jobs_requested = n_jobs_i,
            n_workers = workers_used,
            threaded = use_threads,
            checkpoint_file = checkpoint_path,
            checkpoint_every = checkpoint_every_i,
            checkpoint_completed = completed,
            checkpoint_start_completed = start_completed,
            checkpoint_previous_target_n_traj = checkpoint_state.previous_target_n_traj,
            progress = progress_enabled,
            resumed = checkpoint_state.resumed,
        ),
    )
end
