# Monte Carlo wave-function (quantum-jump) solver.
#
# Each trajectory follows the standard waiting-time algorithm:
#   1. draw a threshold r uniformly from [0, 1);
#   2. integrate d|psi>/dt = -i H_eff |psi> with H_eff = H - (i/2) sum_k c_k' c_k,
#      without renormalising, until the squared norm has decayed to r;
#   3. at that instant apply collapse operator c_k with probability proportional
#      to |c_k psi|^2, normalise, draw a new threshold, and continue.
# The no-jump evolution uses an adaptive Runge-Kutta integrator and the jump
# time is located by root-finding on the integrator's interpolant, so the
# accuracy is set by `rtol`/`atol` and not by a hand-picked time step.

# Random numbers: every trajectory owns an independent Xoshiro256++ stream
# (Julia's standard generator). The stream state is the SHA-256 digest of the
# run seed and the trajectory index, so
#   * results are reproducible from `seed` alone, on any thread count,
#   * resuming from a checkpoint continues with the same streams, and
#   * the seeding does not depend on the Julia version's own seed hashing.
const MCSOLVE_RNG_NAME = "xoshiro256++/sha256-stream-v1"
const _MCSOLVE_STREAM_DOMAIN = "openquantumsim.mcsolve.stream.v1"

# Upper bound on integrator steps per trajectory. Reaching it is reported as a
# failed trajectory instead of looping forever.
const MCWF_MAX_STEPS = 10_000_000

function _trajectory_rng(base_seed::UInt64, traj_idx::Integer)::Xoshiro
    traj_idx >= 1 || throw(ArgumentError("trajectory index must be positive."))
    message = IOBuffer()
    write(message, _MCSOLVE_STREAM_DOMAIN)
    write(message, htol(base_seed))
    write(message, htol(UInt64(traj_idx)))
    words = ltoh.(reinterpret(UInt64, sha256(take!(message))))
    # An all-zero state is the one invalid Xoshiro state.
    all(iszero, words) && return Xoshiro(0x9e3779b97f4a7c15, 0, 0, 0)
    return Xoshiro(words[1], words[2], words[3], words[4])
end

"""
Resolve the user-facing `seed` to the base seed of a run. `seed = 0` asks for
a fresh seed from the operating system; the value actually used is reported
in `solver_stats.base_seed` so the run can be repeated.
"""
function _resolve_base_seed(seed::Integer)::UInt64
    seed < 0 && throw(ArgumentError("seed must be non-negative."))
    seed == 0 && return rand(RandomDevice(), UInt64(1):UInt64(typemax(Int64)))
    return UInt64(seed)
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
    max_step >= 0 || throw(ArgumentError("max_step must be positive, or 0 for no limit."))
    (n_jobs == -1 || n_jobs > 0) || throw(ArgumentError("n_jobs must be -1 or positive."))
    norm(psi0) > 0 || throw(ArgumentError("psi0 must have nonzero norm."))
    return d
end

function _sample_weighted(rng::AbstractRNG, weights::Vector{Float64})::Int
    total = sum(weights)
    total > 0 || throw(ArgumentError("cannot sample jump with zero total weight."))
    threshold = rand(rng) * total
    accum = 0.0
    for idx in eachindex(weights)
        accum += weights[idx]
        threshold <= accum && return idx
    end
    return length(weights)
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

mutable struct _JumpState
    rng::Xoshiro
    threshold::Float64
    n_jumps::Int
end

struct _MCWFParams
    generator::Matrix{ComplexF64}    # -i H_eff
    c_ops::Vector{Matrix{ComplexF64}}
    jump::_JumpState
    weights::Vector{Float64}
    buffer::Vector{ComplexF64}
end

_mcwf_rhs!(du, u, p::_MCWFParams, _t) = (mul!(du, p.generator, u); nothing)

# Zero when the squared norm of the unnormalised state reaches the threshold.
_mcwf_condition(u, _t, integrator) = real(dot(u, u)) - integrator.p.jump.threshold

function _mcwf_jump!(integrator)
    params = integrator.p
    psi = integrator.u
    total = 0.0
    for idx in eachindex(params.c_ops)
        mul!(params.buffer, params.c_ops[idx], psi)
        weight = real(dot(params.buffer, params.buffer))
        params.weights[idx] = weight
        total += weight
    end
    if total > 0
        channel = _sample_weighted(params.jump.rng, params.weights)
        mul!(params.buffer, params.c_ops[channel], psi)
        psi .= params.buffer ./ norm(params.buffer)
        params.jump.n_jumps += 1
    else
        # The norm cannot decay without a nonzero jump rate; reaching this
        # branch means the crossing came from integration error alone.
        psi ./= norm(psi)
    end
    params.jump.threshold = rand(params.jump.rng)
    return nothing
end

function _mcwf_integrator(
    H::Matrix{ComplexF64},
    c_ops::Vector{Matrix{ComplexF64}},
    psi0::Vector{ComplexF64},
    times::Vector{Float64};
    rtol::Float64,
    atol::Float64,
    max_step::Float64,
    ode_solver::AbstractString,
)
    algorithm, _ = _ode_algorithm(ode_solver)
    d = length(psi0)
    params = _MCWFParams(
        -1im .* _effective_hamiltonian(H, c_ops),
        c_ops,
        _JumpState(Xoshiro(1, 2, 3, 4), 1.0, 0),
        zeros(Float64, length(c_ops)),
        zeros(ComplexF64, d),
    )
    problem = ODEProblem(_mcwf_rhs!, copy(psi0), (times[1], times[end]), params)
    callback = isempty(c_ops) ? nothing :
        ContinuousCallback(_mcwf_condition, _mcwf_jump!; save_positions = (false, false))
    step_limit = max_step > 0 ? (dtmax = max_step,) : NamedTuple()
    return init(
        problem,
        algorithm;
        reltol = rtol,
        abstol = atol,
        tstops = unique(times),
        save_everystep = false,
        save_start = false,
        save_end = false,
        callback = callback,
        maxiters = MCWF_MAX_STEPS,
        step_limit...,
    )
end

"""
Propagate one trajectory and write its expectation values (and optionally
its normalised states) at every entry of `times`.

Returns `(ok, retcode, n_jumps, nfev, nsteps)`. When the integrator stops
early the remaining columns are filled with `NaN` and `ok` is `false`.
`integrator` may be `nothing` when all output times coincide.
"""
function _run_trajectory!(
    expect_out::AbstractMatrix{ComplexF64},
    states_out::AbstractMatrix{ComplexF64},
    integrator,
    psi0::Vector{ComplexF64},
    times::Vector{Float64},
    e_ops::Vector{Matrix{ComplexF64}},
    rng::Xoshiro,
)
    save_states = size(states_out, 2) > 0
    if integrator === nothing
        scale = 1 / norm(psi0)
        for idx in eachindex(times)
            for op_idx in eachindex(e_ops)
                expect_out[op_idx, idx] = dot(psi0, e_ops[op_idx], psi0) * scale^2
            end
            save_states && (states_out[:, idx] .= psi0 .* scale)
        end
        return true, SUCCESS_RETCODE, 0, 0, 0
    end

    jump = integrator.p.jump
    jump.rng = rng
    jump.threshold = rand(rng)
    jump.n_jumps = 0
    # Reset the step-size history too: a trajectory must not depend on which
    # trajectory the same integrator ran before it, or the result would
    # change with the number of threads.
    reinit!(integrator, psi0; reset_dt = true)
    nfev_start = integrator.stats.nf
    nsteps_start = integrator.stats.naccept

    ok = true
    for idx in eachindex(times)
        target = times[idx]
        while ok && integrator.t < target
            previous_time = integrator.t
            step!(integrator)
            # A step that does not advance means the integrator gave up.
            ok = integrator.t > previous_time
        end
        if ok
            psi = integrator.u
            norm_squared = real(dot(psi, psi))
            for op_idx in eachindex(e_ops)
                expect_out[op_idx, idx] = dot(psi, e_ops[op_idx], psi) / norm_squared
            end
            save_states && (states_out[:, idx] .= psi ./ sqrt(norm_squared))
        else
            expect_out[:, idx] .= ComplexF64(NaN, NaN)
            save_states && (states_out[:, idx] .= ComplexF64(NaN, NaN))
        end
    end

    retcode = SUCCESS_RETCODE
    if !ok
        retcode = string(integrator.sol.retcode)
        retcode in (SUCCESS_RETCODE, "Default") && (retcode = "Stalled")
    end
    return (
        ok,
        retcode,
        jump.n_jumps,
        Int(integrator.stats.nf - nfev_start),
        Int(integrator.stats.naccept - nsteps_start),
    )
end

mutable struct _TrajectoryTally
    n_failed::Int
    n_jumps::Int
    nfev::Int
    nsteps::Int
    first_failure::String
end

_TrajectoryTally() = _TrajectoryTally(0, 0, 0, 0, "")

function _add!(total::_TrajectoryTally, part::_TrajectoryTally)
    total.n_failed += part.n_failed
    total.n_jumps += part.n_jumps
    total.nfev += part.nfev
    total.nsteps += part.nsteps
    isempty(total.first_failure) && (total.first_failure = part.first_failure)
    return total
end

"""
Run the trajectories `local_indices` of the current batch on one integrator.
Trajectory `first_traj + local_idx - 1` always uses the random stream derived
from its own index, so the result does not depend on how work is split.
"""
function _run_trajectory_chunk!(
    trajectory_expect::Array{ComplexF64, 3},
    integrator,
    local_indices,
    first_traj::Int,
    base_seed::UInt64,
    psi0::Vector{ComplexF64},
    times::Vector{Float64},
    e_ops::Vector{Matrix{ComplexF64}},
)
    tally = _TrajectoryTally()
    no_states = Matrix{ComplexF64}(undef, length(psi0), 0)
    for local_idx in local_indices
        rng = _trajectory_rng(base_seed, first_traj + local_idx - 1)
        ok, retcode, n_jumps, nfev, nsteps = _run_trajectory!(
            view(trajectory_expect, :, :, local_idx),
            no_states,
            integrator,
            psi0,
            times,
            e_ops,
            rng,
        )
        tally.n_jumps += n_jumps
        tally.nfev += nfev
        tally.nsteps += nsteps
        if !ok
            tally.n_failed += 1
            isempty(tally.first_failure) && (tally.first_failure = retcode)
        end
    end
    return tally
end

function _mcsolve_worker_count(n_jobs::Int)::Int
    available = Threads.nthreads()
    n_jobs == -1 && return available
    return min(n_jobs, available)
end

const MCSOLVE_CHECKPOINT_FORMAT = "openquantumsim.mcsolve.checkpoint"
# Version 2: per-trajectory Xoshiro streams and the waiting-time stepper.
# Version 1 checkpoints were written with a different generator and jump rule
# and must not be continued.
const MCSOLVE_CHECKPOINT_VERSION = "2"

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
    rtol::Float64,
    atol::Float64,
    ode_solver::AbstractString,
)
    io = IOBuffer()
    write(io, "mcsolve-v2\n")
    write(io, "rng=$(MCSOLVE_RNG_NAME)\n")
    write(io, "max_step=$(max_step)\n")
    write(io, "rtol=$(rtol)\n")
    write(io, "atol=$(atol)\n")
    write(io, "ode_solver=$(ode_solver)\n")
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
        base_seed = _resolve_base_seed(seed)
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
        stored_version = string(attrs(file)["format_version"])
        stored_version == MCSOLVE_CHECKPOINT_VERSION || throw(ArgumentError(
            "mcsolve checkpoint has format version $(stored_version), but this " *
            "release writes version $(MCSOLVE_CHECKPOINT_VERSION). Checkpoints " *
            "from older releases used a different random number generator and " *
            "jump rule and cannot be resumed; delete the file or choose a new " *
            "checkpoint_file.",
        ))
        attrs(file)["config_hash"] == config_hash || throw(ArgumentError(
            "checkpoint does not match this mcsolve configuration (system, " *
            "time grid, tolerances, or solver options differ).",
        ))

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
        attrs(file)["rng"] = MCSOLVE_RNG_NAME
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

function _prepare_trajectory_operators(H, c_ops, e_ops)
    Hc = _as_complex_matrix(H)
    d = size(Hc, 1)
    collapse_ops = _as_complex_matrices(c_ops)
    expectation_ops = _as_complex_matrices(e_ops)
    for c in collapse_ops
        size(c) == (d, d) || throw(DimensionMismatch("collapse operators must match H."))
    end
    for op in expectation_ops
        size(op) == (d, d) || throw(DimensionMismatch("expectation operators must match H."))
    end
    return Hc, collapse_ops, expectation_ops
end

"""
Run one quantum-jump trajectory.

`max_step = 0` (the default) leaves the adaptive step unbounded; a positive
value caps it. `seed = 0` draws a fresh seed.
"""
function single_trajectory(
    H,
    psi0,
    tlist,
    c_ops = Matrix{ComplexF64}[],
    e_ops = Matrix{ComplexF64}[],
    seed::Integer = 0,
    max_step::Real = 0.0,
    save_states::Bool = false,
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    ode_solver::AbstractString = "auto",
)
    Hc, collapse_ops, expectation_ops = _prepare_trajectory_operators(H, c_ops, e_ops)
    psi0c = _as_complex_vector(psi0)
    times = Float64.(collect(tlist))
    max_step_f = Float64(max_step)
    d = _validate_mcsolve_inputs(Hc, psi0c, times, 1, max_step_f)
    psi0c ./= norm(psi0c)
    _, ode_solver_name = _ode_algorithm(ode_solver)

    base_seed = _resolve_base_seed(seed)
    rng = _trajectory_rng(base_seed, 1)
    integrator = times[end] == times[1] ? nothing : _mcwf_integrator(
        Hc,
        collapse_ops,
        psi0c,
        times;
        rtol = Float64(rtol),
        atol = Float64(atol),
        max_step = max_step_f,
        ode_solver = ode_solver_name,
    )
    expect_values = zeros(ComplexF64, length(expectation_ops), length(times))
    # States are always tracked so the final state can be returned.
    states = Matrix{ComplexF64}(undef, d, length(times))
    local outcome
    elapsed = @elapsed begin
        outcome = _run_trajectory!(
            expect_values,
            states,
            integrator,
            psi0c,
            times,
            expectation_ops,
            rng,
        )
    end
    ok, retcode, n_jumps, nfev, nsteps = outcome

    return (
        times = times,
        final_state = states[:, end],
        states = save_states ? states : Matrix{ComplexF64}(undef, d, 0),
        expect = expect_values,
        solver_stats = (
            nsteps = nsteps,
            nfev = nfev,
            wall_time = elapsed,
            retcode = retcode,
            success = ok,
            method = "waiting-time",
            ode_solver = ode_solver_name,
            n_traj = 1,
            n_failed_trajectories = ok ? 0 : 1,
            n_jumps = n_jumps,
            max_step = max_step_f,
            base_seed = Int(base_seed),
            rng = MCSOLVE_RNG_NAME,
        ),
    )
end

"""
Average `n_traj` quantum-jump trajectories.

`n_jobs = 1` runs serially, `n_jobs = -1` uses every Julia thread, and any
other positive value uses at most that many threads. The result is identical
for every choice because each trajectory has its own random stream.
"""
function mcsolve(
    H,
    psi0,
    tlist,
    c_ops = Matrix{ComplexF64}[],
    e_ops = Matrix{ComplexF64}[],
    n_traj::Integer = 500,
    seed::Integer = 0,
    max_step::Real = 0.0,
    n_jobs::Integer = 1,
    checkpoint_file::AbstractString = "",
    checkpoint_every::Integer = 100,
    progress::Bool = false,
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
    ode_solver::AbstractString = "auto",
)
    Hc, collapse_ops, expectation_ops = _prepare_trajectory_operators(H, c_ops, e_ops)
    psi0c = _as_complex_vector(psi0)
    times = Float64.(collect(tlist))
    n_traj_i = Int(n_traj)
    max_step_f = Float64(max_step)
    rtol_f = Float64(rtol)
    atol_f = Float64(atol)
    n_jobs_i = Int(n_jobs)
    checkpoint_path = String(strip(String(checkpoint_file)))
    checkpoint_every_i = Int(checkpoint_every)
    checkpoint_every_i > 0 ||
        throw(ArgumentError("checkpoint_every must be positive."))
    progress_enabled = Bool(progress)
    d = _validate_mcsolve_inputs(Hc, psi0c, times, n_traj_i, max_step_f, n_jobs_i)
    psi0c ./= norm(psi0c)
    _, ode_solver_name = _ode_algorithm(ode_solver)

    n_ops = length(expectation_ops)
    n_times = length(times)
    config_hash = _mcsolve_config_hash(
        Hc,
        psi0c,
        times,
        collapse_ops,
        expectation_ops,
        max_step_f,
        rtol_f,
        atol_f,
        ode_solver_name,
    )
    workers_used = _mcsolve_worker_count(n_jobs_i)
    use_threads = workers_used > 1
    integrators = [
        times[end] == times[1] ? nothing : _mcwf_integrator(
            Hc,
            collapse_ops,
            psi0c,
            times;
            rtol = rtol_f,
            atol = atol_f,
            max_step = max_step_f,
            ode_solver = ode_solver_name,
        ) for _ in 1:workers_used
    ]
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
    tally = _TrajectoryTally()

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
                tasks = map(1:workers_used) do worker
                    Threads.@spawn _run_trajectory_chunk!(
                        trajectory_expect,
                        integrators[worker],
                        worker:workers_used:batch_count,
                        next_traj,
                        base_seed,
                        psi0c,
                        times,
                        expectation_ops,
                    )
                end
                for task in tasks
                    _add!(tally, fetch(task))
                end
            else
                _add!(tally, _run_trajectory_chunk!(
                    trajectory_expect,
                    integrators[1],
                    1:batch_count,
                    next_traj,
                    base_seed,
                    psi0c,
                    times,
                    expectation_ops,
                ))
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
    # A resumed run inherits any failed trajectory through the stored sums.
    finite = all(isfinite, expect_sum)
    success = tally.n_failed == 0 && finite
    retcode = success ? SUCCESS_RETCODE :
        !isempty(tally.first_failure) ? tally.first_failure : "NonFiniteCheckpoint"
    return (
        times = times,
        states = Array{ComplexF64}(undef, d, d, 0),
        expect = expect_mean,
        expect_std = expect_std,
        expect_stderr = expect_stderr,
        entropy = zeros(Float64, length(times)),
        solver_stats = (
            nsteps = tally.nsteps,
            nfev = tally.nfev,
            wall_time = elapsed,
            retcode = retcode,
            success = success,
            method = "waiting-time",
            ode_solver = ode_solver_name,
            n_traj = n_traj_i,
            n_failed_trajectories = tally.n_failed,
            n_jumps = tally.n_jumps,
            max_step = max_step_f,
            base_seed = Int(base_seed),
            rng = MCSOLVE_RNG_NAME,
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
