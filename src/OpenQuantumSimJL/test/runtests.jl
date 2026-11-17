using Test
using HDF5
using LinearAlgebra
using OpenQuantumSimJL
using Random
using SparseArrays

@testset "Hilbert spaces" begin
    f = FockSpace(4; label="cavity")
    s = SpinSpace(0.5; label="atom")
    d = DickeSpace(4; label="ensemble")
    c = tensor_space(f, s, d)
    @test dim(f) == 4
    @test dim(s) == 2
    @test dim(d) == 5
    @test d.n_spins == 4
    @test dim(c) == 40
end

@testset "Operators" begin
    a = destroy(4)
    @test size(a) == (4, 4)
    @test a[1, 2] == 1
    @test a[2, 3] == sqrt(2)
    @test sigmax()[1, 2] == 1
    @test sigmam()[2, 1] == 1
    d = DickeSpace(4)
    jm = collective_lowering(d)
    @test size(jm) == (5, 5)
    @test jm[2, 1] ≈ 2
    @test jm[3, 2] ≈ sqrt(6)
    @test collective_raising(d) ≈ jm'
    @test diag(collective_z(d)) ≈ ComplexF64[2, 1, 0, -1, -2]
    @test diag(collective_excitation(d)) ≈ ComplexF64[4, 3, 2, 1, 0]
end

@testset "Observables" begin
    psi = ComplexF64[1, 0, 0, 0]
    rhoA, rhoB = partial_traces(psi, 2, 2)
    @test rhoA ≈ [1 0; 0 0]
    @test rhoB ≈ [1 0; 0 0]
    @test von_neumann_entropy(rhoA) ≈ 0
end

@testset "mesolve qubit decay" begin
    gamma = 0.35
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    rho0 = ComplexF64[1 0; 0 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 6.0; length=61))

    result = mesolve(
        H,
        rho0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        1e-9,
        1e-11,
        false,
    )

    @test result.times ≈ times
    @test real.(result.expect[1, :]) ≈ exp.(-gamma .* times) rtol=2e-5 atol=2e-7
    @test maximum(abs.(imag.(result.expect[1, :]))) < 1e-12
    @test result.solver_stats.requested_method == "auto"
    @test result.solver_stats.method == "ode"
end

@testset "mesolve Krylov and ODE agree" begin
    gamma = 0.2
    omega = 0.3
    H = ComplexF64[0 omega / 2; omega / 2 0]
    sm = ComplexF64[0 0; 1 0]
    rho0 = ComplexF64[1 0; 0 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 1.0; length=11))

    krylov = mesolve(
        H,
        rho0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        1e-9,
        1e-11,
        false,
        "krylov",
        20,
    )
    ode = mesolve(
        H,
        rho0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        1e-9,
        1e-11,
        false,
        "ode",
        20,
    )

    @test real.(krylov.expect[1, :]) ≈ real.(ode.expect[1, :]) rtol=1e-7 atol=1e-9
    @test krylov.solver_stats.krylov_dim == 20
end

@testset "mesolve saved-state invariants" begin
    gamma = 0.2
    H = ComplexF64[0 0.15; 0.15 0]
    sm = ComplexF64[0 0; 1 0]
    rho0 = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 3.0; length=31))

    result = mesolve(
        H,
        rho0,
        times,
        [sqrt(gamma) * sm],
        Matrix{ComplexF64}[],
        1e-9,
        1e-11,
        true,
    )

    @test size(result.states) == (2, 2, length(times))
    @test result.states[:, :, 1] ≈ rho0
    for idx in axes(result.states, 3)
        rho = result.states[:, :, idx]
        @test tr(rho) ≈ 1 atol=2e-8
        @test rho ≈ rho' atol=2e-8
        @test minimum(eigvals(Hermitian((rho .+ rho') ./ 2))) > -2e-8
    end
end

@testset "mesolve Rabi oscillation" begin
    omega = 0.8
    H = ComplexF64[0 omega / 2; omega / 2 0]
    rho0 = ComplexF64[1 0; 0 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 8.0; length=81))

    result = mesolve(
        H,
        rho0,
        times,
        Matrix{ComplexF64}[],
        [excited_projector],
        1e-9,
        1e-11,
        false,
    )

    expected = cos.(0.5 .* omega .* times) .^ 2
    @test real.(result.expect[1, :]) ≈ expected rtol=2e-6 atol=2e-7
    @test maximum(abs.(imag.(result.expect[1, :]))) < 1e-12
end

@testset "time-dependent mesolve Rabi oscillation" begin
    omega = 0.8
    H0 = zeros(ComplexF64, 2, 2)
    Hdrive = ComplexF64[0 1; 1 0]
    rho0 = ComplexF64[1 0; 0 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 8.0; length=81))
    coefficient = InterpolatedCoefficient([0.0, 8.0], [omega / 2, omega / 2])

    result = mesolve_time_dependent(
        H0,
        [Hdrive],
        [coefficient],
        rho0,
        times,
        Matrix{ComplexF64}[],
        [excited_projector],
        1e-9,
        1e-11,
        false,
    )

    expected = cos.(0.5 .* omega .* times) .^ 2
    @test real.(result.expect[1, :]) ≈ expected rtol=2e-6 atol=2e-7
    @test maximum(abs.(imag.(result.expect[1, :]))) < 1e-12
    @test result.solver_stats.method == "time-dependent-ode"
    @test result.solver_stats.time_dependent
end

@testset "quantum regression correlations" begin
    gamma = 0.4
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    sp = sm'
    rho0 = ComplexF64[1 0; 0 0]
    taus = collect(range(0.0, 4.0; length=41))

    corr_1t = correlation_2op_1t(
        H,
        rho0,
        taus,
        sp,
        sm,
        [sqrt(gamma) * sm];
        rtol = 1e-9,
        atol = 1e-11,
    )

    @test real.(corr_1t.correlations) ≈ exp.(-0.5 .* gamma .* taus) rtol=2e-6 atol=2e-8
    @test maximum(abs.(imag.(corr_1t.correlations))) < 1e-12

    times = collect(range(0.0, 3.0; length=7))
    taus_2t = collect(range(0.0, 2.0; length=9))
    corr_2t = correlation_2op_2t(
        H,
        rho0,
        times,
        taus_2t,
        sp,
        sm,
        [sqrt(gamma) * sm];
        rtol = 1e-9,
        atol = 1e-11,
    )
    expected = [exp(-gamma * t) * exp(-0.5 * gamma * tau) for t in times, tau in taus_2t]

    @test real.(corr_2t.correlations) ≈ expected rtol=2e-6 atol=2e-8
    @test maximum(abs.(imag.(corr_2t.correlations))) < 1e-12
    @test corr_2t.solver_stats.quantum_regression
end

@testset "steadystate qubit decay" begin
    gamma = 0.35
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    expected = ComplexF64[0 0; 0 1]

    direct = steadystate(H, [sqrt(gamma) * sm]; method = "direct")
    iterative = steadystate(
        H,
        [sqrt(gamma) * sm];
        method = "iterative-gmres",
        rtol = 1e-12,
        krylov_dim = 10,
    )

    @test direct ≈ expected atol=1e-10
    @test iterative ≈ expected atol=1e-10
    @test tr(direct) ≈ 1
    @test direct ≈ direct'
end

@testset "single_trajectory saved ket states" begin
    H = zeros(ComplexF64, 2, 2)
    psi0 = ComplexF64[1, 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 0.2; length=3))

    result = single_trajectory(
        H,
        psi0,
        times,
        Matrix{ComplexF64}[],
        [excited_projector],
        909,
        0.05,
        true,
    )

    @test size(result.states) == (2, length(times))
    @test result.states[:, 1] ≈ psi0
    @test real.(result.expect[1, :]) ≈ ones(length(times))
    for idx in axes(result.states, 2)
        @test norm(result.states[:, idx]) ≈ 1
    end
end

@testset "mcsolve qubit decay" begin
    gamma = 0.35
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 4.0; length=41))

    result = mcsolve(
        H,
        psi0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        1000,
        2026,
        0.01,
    )

    expected = exp.(-gamma .* times)
    population = real.(result.expect[1, :])

    @test result.solver_stats.n_traj == 1000
    @test size(result.expect_std) == size(result.expect)
    @test size(result.expect_stderr) == size(result.expect)
    @test all(result.expect_std .>= 0)
    @test all(result.expect_stderr .>= 0)
    @test all(population .>= -1e-12)
    @test all(population .<= 1.0 + 1e-12)
    @test sum(abs.(population .- expected)) / length(times) < 0.03
    @test maximum(abs.(population .- expected)) < 0.08
end

@testset "mcsolve standard error scaling" begin
    gamma = 0.35
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 3.0; length=31))

    low = mcsolve(
        H,
        psi0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        250,
        44,
        0.01,
    )
    high = mcsolve(
        H,
        psi0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        1000,
        44,
        0.01,
    )

    mask = vec(low.expect_std[1, :] .> 0.05)
    ratios = high.expect_stderr[1, mask] ./ low.expect_stderr[1, mask]
    stderr_ratio = sort(ratios)[max(1, length(ratios) ÷ 2)]

    @test 0.35 < stderr_ratio < 0.65
    @test isapprox(stderr_ratio, sqrt(250 / 1000); atol=0.15)
end

@testset "mcsolve serial/thread request determinism" begin
    gamma = 0.35
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 2.0; length=21))

    serial = mcsolve(
        H,
        psi0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        250,
        77,
        0.01,
        1,
    )
    threaded = mcsolve(
        H,
        psi0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        250,
        77,
        0.01,
        -1,
    )

    @test threaded.solver_stats.n_jobs_requested == -1
    @test threaded.solver_stats.n_workers >= 1
    @test serial.expect ≈ threaded.expect
    @test serial.expect_std ≈ threaded.expect_std
    @test serial.expect_stderr ≈ threaded.expect_stderr
end

@testset "mcsolve checkpoint resume" begin
    gamma = 0.35
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 2.0; length=21))
    checkpoint_file = tempname() * ".h5"

    mcsolve(
        H,
        psi0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        30,
        505,
        0.01,
        1,
        checkpoint_file,
        10,
    )
    resumed = mcsolve(
        H,
        psi0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        80,
        505,
        0.01,
        1,
        checkpoint_file,
        10,
    )
    clean = mcsolve(
        H,
        psi0,
        times,
        [sqrt(gamma) * sm],
        [excited_projector],
        80,
        505,
        0.01,
        1,
    )

    @test isfile(checkpoint_file)
    @test resumed.solver_stats.resumed == true
    @test resumed.solver_stats.checkpoint_start_completed == 30
    @test resumed.solver_stats.checkpoint_completed == 80
    @test resumed.solver_stats.checkpoint_previous_target_n_traj == 30
    @test resumed.expect ≈ clean.expect
    @test resumed.expect_std ≈ clean.expect_std
    @test resumed.expect_stderr ≈ clean.expect_stderr
end

@testset "mcsolve progress reporting" begin
    gamma = 0.35
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 1.0; length=11))

    quiet_file = tempname()
    quiet = open(quiet_file, "w") do io
        redirect_stdout(io) do
            mcsolve(
                H,
                psi0,
                times,
                [sqrt(gamma) * sm],
                [excited_projector],
                25,
                606,
                0.01,
                1,
                "",
                10,
                false,
            )
        end
    end

    loud_file = tempname()
    loud = open(loud_file, "w") do io
        redirect_stdout(io) do
            mcsolve(
                H,
                psi0,
                times,
                [sqrt(gamma) * sm],
                [excited_projector],
                25,
                606,
                0.01,
                1,
                "",
                10,
                true,
            )
        end
    end

    quiet_output = read(quiet_file, String)
    loud_output = read(loud_file, String)
    @test !occursin("mcsolve progress:", quiet_output)
    @test occursin("mcsolve progress:", loud_output)
    @test occursin("25/25", loud_output)
    @test quiet.solver_stats.progress == false
    @test loud.solver_stats.progress == true
    @test quiet.expect ≈ loud.expect
    @test quiet.expect_std ≈ loud.expect_std
    @test quiet.expect_stderr ≈ loud.expect_stderr
end

@testset "Bloch-Redfield" begin
    # Zero-temperature amplitude damping: decay channels sit at negative
    # Bohr frequencies, so a one-sided spectrum must reproduce the Lindblad
    # solution exactly for a qubit.
    H = ComplexF64[0.5 0; 0 -0.5]
    rho0 = ComplexF64[1 0; 0 0]
    times = [0.0, 0.5, 1.5]
    kappa = 0.4
    lowering = ComplexF64[0 1; 0 0]
    spectrum(omega) = omega < 0 ? kappa : 0.0
    sm = ComplexF64[0 1; 0 0]

    br = brmesolve(H, rho0, times, [lowering], [spectrum], ComplexF64[], 1e-8, 1e-10, false, "ode", 30, false)
    lind = mesolve(H, rho0, times, [sqrt(kappa) * sm], ComplexF64[], 1e-8, 1e-10, false, "ode", 30, false)
    @test br.expect ≈ lind.expect atol = 1e-10

    # Constant white spectrum on sigma_z is pure dephasing; coherences decay
    # at rate 2*gamma, populations stay fixed.
    sz = ComplexF64[1 0; 0 -1]
    white(omega) = 0.3
    rho_plus = 0.5 * ComplexF64[1 1; 1 1]
    dep = brmesolve(H, rho_plus, times, [sz], [white], ComplexF64[], 1e-8, 1e-10, true, "ode", 30, false)
    for idx in 1:length(times)
        rho = dep.states[:, :, idx]
        @test real(rho[1, 1]) ≈ 0.5 atol = 1e-10
        # coherence magnitude decays at rate 2*gamma; the phase precesses
        @test abs(rho[1, 2]) ≈ 0.5 * exp(-0.6 * times[idx]) atol = 1e-9
    end

    # The generator is trace preserving: L vec(I) = 0.
    L = br_liouvillian(H, [lowering], [spectrum])
    d = 2
    identity_vec = vec(Matrix{ComplexF64}(I, d, d))
    @test norm(L * identity_vec) ≈ 0.0 atol = 1e-12

    # Non-Hermitian Hamiltonians are rejected.
    @test_throws ArgumentError br_liouvillian(
        ComplexF64[0 1; 0 0],
        [lowering],
        [spectrum],
    )
end

function _driven_cavity(n::Int; drive = 2.0, detuning = 1.0, kappa = 0.5)
    a = zeros(ComplexF64, n, n)
    for k in 1:(n - 1)
        a[k, k + 1] = sqrt(k)
    end
    H = drive * (a + a') + detuning * a' * a
    rho0 = zeros(ComplexF64, n, n)
    rho0[1, 1] = 1
    return H, [sqrt(kappa) * a], rho0
end

@testset "solver health: healthy run statistics" begin
    gamma = 0.3
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    rho0 = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 5.0; length=51))

    result = mesolve(H, rho0, times, [sqrt(gamma) * sm], [rho0], 1e-8, 1e-10, false)
    stats = result.solver_stats

    @test stats.success
    @test stats.retcode == "Success"
    @test stats.ode_solver == "tsit5"
    @test stats.nsteps > 0
    @test stats.nfev > stats.nsteps
    @test stats.krylov_converged
    @test stats.trace_drift < 1e-10
    @test stats.hermiticity_drift < 1e-10
    @test stats.min_eigenvalue > -1e-10
    @test stats.positivity_samples == OpenQuantumSimJL.POSITIVITY_SAMPLES_SMALL
    @test stats.n_nonfinite_states == 0
end

@testset "solver health: positivity spot check is bounded and includes the ends" begin
    sample = OpenQuantumSimJL._positivity_sample_indices
    small = OpenQuantumSimJL.POSITIVITY_SAMPLES_SMALL
    large = OpenQuantumSimJL.POSITIVITY_SAMPLES_LARGE

    @test sample(3, 2) == [1, 2, 3]
    @test length(sample(81, 2)) == small
    @test length(sample(81, 200)) == large
    for (n_times, d) in ((81, 2), (81, 200), (1000, 10))
        indices = sample(n_times, d)
        @test first(indices) == 1
        @test last(indices) == n_times
        @test issorted(indices) && allunique(indices)
    end

    # A negative eigenvalue at the final time is always caught.
    d = 2
    good = vec(ComplexF64[1 0; 0 0])
    bad = vec(ComplexF64[1.2 0; 0 -0.2])
    states = [good for _ in 1:50]
    push!(states, bad)
    @test OpenQuantumSimJL._density_health(states, d).min_eigenvalue ≈ -0.2

    # The Cholesky screen: positive and pure (rank-deficient) states report 0,
    # tiny negative rounding within the margin is tolerated, and anything
    # beyond it is reported exactly.
    negative = OpenQuantumSimJL._negative_eigenvalue
    psi = normalize(ComplexF64[1, 2im, -1])
    @test negative(Matrix(I, 3, 3) / 3) == 0.0
    @test negative(psi * psi') == 0.0
    @test negative(ComplexF64[1 0; 0 -1e-12]) == 0.0
    @test negative(ComplexF64[1.01 0; 0 -0.01]) ≈ -0.01
    @test negative(ComplexF64[0.5 0.6; 0.6 0.5]) ≈ -0.1
end

@testset "solver health: integrator failure is reported, not hidden" begin
    H0 = zeros(ComplexF64, 2, 2)
    Hdrive = ComplexF64[0 1; 1 0]
    rho0 = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 3.0; length=13))
    broken_drive(t) = t > 1.0 ? NaN : 0.4

    result = mesolve_time_dependent(
        H0,
        [Hdrive],
        [broken_drive],
        rho0,
        times,
        Matrix{ComplexF64}[],
        [rho0],
        1e-8,
        1e-10,
        true,
    )
    stats = result.solver_stats
    population = real.(result.expect[1, :])

    @test !stats.success
    @test stats.retcode != "Success"
    @test stats.n_nonfinite_states > 0
    @test size(result.expect, 2) == length(times)
    @test size(result.states, 3) == length(times)
    # Values before the failure are still correct; values after it are NaN.
    @test population[1:5] ≈ cos.(0.4 .* times[1:5]) .^ 2 rtol=1e-6
    @test all(isnan, population[6:end])
end

@testset "solver health: Krylov non-convergence is reported" begin
    H, c_ops, rho0 = _driven_cavity(12)
    projector = copy(rho0)

    starved = mesolve(H, rho0, [0.0, 50.0], c_ops, [projector], 1e-10, 1e-10, false, "krylov", 4)
    @test !starved.solver_stats.success
    @test !starved.solver_stats.krylov_converged
    @test starved.solver_stats.retcode == "KrylovNotConverged"
    @test starved.solver_stats.krylov_max_residual > 1.0

    healthy = mesolve(H, rho0, [0.0, 50.0], c_ops, [projector], 1e-10, 1e-10, false, "krylov", 30)
    reference = mesolve(H, rho0, [0.0, 50.0], c_ops, [projector], 1e-10, 1e-12, false, "ode", 30, false, "vern7")
    @test healthy.solver_stats.success
    @test healthy.solver_stats.krylov_converged
    @test healthy.solver_stats.nfev > 0
    @test healthy.expect ≈ reference.expect atol=1e-7

    info = krylov_expmv_info(
        liouvillian(OpenQuantumSimJL._as_complex_operator(H), OpenQuantumSimJL._as_complex_operators(c_ops)),
        50.0,
        vec(rho0);
        krylov_dim = 4,
        tol = 1e-10,
    )
    @test !info.converged
end

@testset "solver health: correlation functions report propagation failures" begin
    H, c_ops, rho0 = _driven_cavity(12)
    a = c_ops[1] ./ sqrt(0.5)

    starved = correlation_2op_1t(
        H, rho0, [0.0, 50.0], a, a', c_ops; method = "krylov", krylov_dim = 4,
    )
    @test !starved.solver_stats.success
    @test !starved.solver_stats.krylov_converged

    krylov = correlation_2op_1t(
        H, rho0, [0.0, 1.0, 2.0], a, a', c_ops; method = "krylov", krylov_dim = 30,
    )
    default = correlation_2op_1t(H, rho0, [0.0, 1.0, 2.0], a, a', c_ops)
    @test default.solver_stats.method == "auto"
    @test default.correlations ≈ krylov.correlations atol=1e-7
    ode = correlation_2op_1t(H, rho0, [0.0, 1.0, 2.0], a, a', c_ops; method = "ode", ode_solver = "vern7")
    @test krylov.solver_stats.success
    @test ode.solver_stats.success
    @test krylov.correlations ≈ ode.correlations atol=1e-7

    # A delay grid that does not start at zero is still propagated from zero.
    late = correlation_2op_1t(H, rho0, [1.0, 2.0], a, a', c_ops; method = "ode", ode_solver = "vern7")
    @test late.correlations ≈ ode.correlations[2:3] atol=1e-9
end

@testset "steadystate reports convergence" begin
    H, c_ops, _ = _driven_cavity(25)

    direct = steadystate_info(H, c_ops; method = "direct")
    @test direct.solver_stats.success
    @test direct.solver_stats.method == "direct"
    @test direct.solver_stats.relative_residual < 1e-12
    @test direct.solver_stats.min_eigenvalue > -1e-10
    @test tr(direct.rho) ≈ 1

    # Unpreconditioned GMRES stagnates on this problem. That must be visible.
    gmres = steadystate_info(H, c_ops; method = "iterative-gmres", rtol = 1e-10, krylov_dim = 30)
    @test !gmres.solver_stats.success
    @test !gmres.solver_stats.converged
    @test gmres.solver_stats.retcode == "GMRESNotConverged"
    @test gmres.solver_stats.relative_residual > 1e-6

    # The default picks the direct solve for systems this small.
    auto = steadystate_info(H, c_ops)
    @test auto.solver_stats.requested_method == "auto"
    @test auto.solver_stats.method == "direct"
    @test auto.rho ≈ direct.rho
    @test steadystate(H, c_ops) ≈ direct.rho

    @test_throws ArgumentError steadystate_info(H, c_ops; method = "magic")
end

@testset "ode_solver selection and repeated output times" begin
    gamma = 0.3
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    rho0 = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 5.0; length=51))
    expected = exp.(-gamma .* times)

    for solver in ("tsit5", "vern7", "vern9")
        result = mesolve(H, rho0, times, [sqrt(gamma) * sm], [rho0], 1e-10, 1e-12, false, "ode", 30, false, solver)
        @test result.solver_stats.ode_solver == solver
        @test real.(result.expect[1, :]) ≈ expected atol=1e-8
    end
    @test_throws ArgumentError mesolve(H, rho0, times, [sqrt(gamma) * sm], [rho0], 1e-8, 1e-10, false, "ode", 30, false, "euler")
    @test_throws ArgumentError mesolve(H, rho0, times, [sqrt(gamma) * sm], [rho0], 1e-8, 1e-10, false, "tsit5", 30, false, "vern7")
    @test_throws ArgumentError mesolve(H, rho0, times, [sqrt(gamma) * sm], [rho0], 1e-8, 1e-10, false, "magic")

    repeated = [0.0, 1.0, 1.0, 2.0]
    for method in ("ode", "krylov")
        result = mesolve(H, rho0, repeated, [sqrt(gamma) * sm], [rho0], 1e-9, 1e-11, false, method)
        @test result.solver_stats.success
        @test real.(result.expect[1, :]) ≈ exp.(-gamma .* repeated) atol=1e-7
    end

    single = mesolve(H, rho0, [0.7], [sqrt(gamma) * sm], [rho0], 1e-9, 1e-11, true)
    @test single.solver_stats.success
    @test single.states[:, :, 1] ≈ rho0
end

@testset "trajectory random streams" begin
    stream = OpenQuantumSimJL._trajectory_rng

    # Pinned values: the stream derivation must not change silently, because
    # that would change every seeded result and break checkpoint resumes.
    first_stream = stream(UInt64(2026), 1)
    @test rand(first_stream) == 0.31798548078062383
    @test rand(first_stream) == 0.88198100494944
    @test rand(stream(UInt64(2026), 2)) == 0.3388444168125838
    @test rand(stream(UInt64(1), 1), UInt64) == 0x278d16b7748d639f

    # Same (seed, trajectory) gives the same stream; anything else differs.
    @test rand(stream(UInt64(7), 3), 8) == rand(stream(UInt64(7), 3), 8)
    @test rand(stream(UInt64(7), 3), 8) != rand(stream(UInt64(7), 4), 8)
    @test rand(stream(UInt64(7), 3), 8) != rand(stream(UInt64(8), 3), 8)
    # (seed, index) pairs that share a sum must not share a stream.
    @test rand(stream(UInt64(1), 2), 8) != rand(stream(UInt64(2), 1), 8)
    @test_throws ArgumentError stream(UInt64(1), 0)

    # Uniformity of the first draw across many trajectories.
    draws = [rand(stream(UInt64(99), idx)) for idx in 1:20_000]
    @test all(0 .<= draws .< 1)
    @test abs(sum(draws) / length(draws) - 0.5) < 0.01
    @test abs(sum(abs2, draws) / length(draws) - 1 / 3) < 0.01

    @test OpenQuantumSimJL._resolve_base_seed(42) == UInt64(42)
    # 0 is an ordinary seed; only -1 asks for a fresh one.
    @test OpenQuantumSimJL._resolve_base_seed(0) == UInt64(0)
    fresh = OpenQuantumSimJL.FRESH_SEED
    @test OpenQuantumSimJL._resolve_base_seed(fresh) != OpenQuantumSimJL._resolve_base_seed(fresh)
    @test_throws ArgumentError OpenQuantumSimJL._resolve_base_seed(-2)
end

@testset "mcsolve refuses checkpoints from the old generator" begin
    gamma = 0.35
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    excited_projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 1.0; length=6))
    checkpoint_file = tempname() * ".h5"

    mcsolve(H, psi0, times, [sqrt(gamma) * sm], [excited_projector], 10, 505, 0.01, 1, checkpoint_file, 5)
    h5open(checkpoint_file, "r") do file
        @test read_attribute(file, "format_version") == "2"
        @test read_attribute(file, "rng") == OpenQuantumSimJL.MCSOLVE_RNG_NAME
    end

    h5open(checkpoint_file, "r+") do file
        delete_attribute(file, "format_version")
        write_attribute(file, "format_version", "1")
    end
    @test_throws ArgumentError mcsolve(
        H, psi0, times, [sqrt(gamma) * sm], [excited_projector], 20, 505, 0.01, 1, checkpoint_file, 5,
    )
end

@testset "waiting-time trajectories: closed-system evolution" begin
    omega = 0.8
    H = ComplexF64[0 omega / 2; omega / 2 0]
    psi0 = ComplexF64[1, 0]
    projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 20.0; length=101))

    result = single_trajectory(H, psi0, times, Matrix{ComplexF64}[], [projector], 7, 0.0, true, 1e-10, 1e-12)
    stats = result.solver_stats

    @test stats.success
    @test stats.retcode == "Success"
    @test stats.method == "waiting-time"
    @test stats.n_jumps == 0
    @test stats.nfev > 0
    @test stats.base_seed == 7
    @test real.(result.expect[1, :]) ≈ cos.(0.5 .* omega .* times) .^ 2 atol=1e-8
    @test all(norm(result.states[:, idx]) ≈ 1 for idx in axes(result.states, 2))
    @test result.final_state ≈ result.states[:, end]

    # The initial ket is normalised; its scale must not change the physics.
    scaled = single_trajectory(H, 3.0 .* psi0, times, Matrix{ComplexF64}[], [projector], 7, 0.0, false, 1e-10, 1e-12)
    @test scaled.expect ≈ result.expect
    @test size(scaled.states) == (2, 0)
end

@testset "waiting-time trajectories: jumps" begin
    gamma = 0.7
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 20.0; length=2001))

    # A decaying qubit jumps exactly once, and stays in the ground state after.
    result = single_trajectory(H, psi0, times, [sqrt(gamma) * sm], [projector], 11, 0.0, true)
    population = real.(result.expect[1, :])
    @test result.solver_stats.n_jumps == 1
    @test result.solver_stats.success
    @test all(value -> isapprox(value, 0; atol=1e-12) || isapprox(value, 1; atol=1e-12), population)
    @test issorted(population; rev=true)
    @test population[1] ≈ 1
    @test population[end] ≈ 0 atol=1e-12

    # The jump time does not depend on the output grid or the step cap.
    jump_index = findfirst(<(0.5), population)
    coarse_times = [0.0, times[jump_index - 1], times[jump_index], 20.0]
    coarse = single_trajectory(H, psi0, coarse_times, [sqrt(gamma) * sm], [projector], 11, 0.0, false)
    capped = single_trajectory(H, psi0, coarse_times, [sqrt(gamma) * sm], [projector], 11, 0.05, false)
    @test real.(coarse.expect[1, :]) ≈ [1, 1, 0, 0] atol=1e-12
    @test real.(capped.expect[1, :]) ≈ [1, 1, 0, 0] atol=1e-12

    # Same seed, same trajectory; different seed, different jump time.
    again = single_trajectory(H, psi0, times, [sqrt(gamma) * sm], [projector], 11, 0.0, false)
    other = single_trajectory(H, psi0, times, [sqrt(gamma) * sm], [projector], 12, 0.0, false)
    @test again.expect == result.expect
    @test other.expect != result.expect

    # Waiting times are exponentially distributed with rate gamma.
    jump_times = Float64[]
    for seed in 1:4000
        run = single_trajectory(H, psi0, times, [sqrt(gamma) * sm], [projector], seed, 0.0, false)
        index = findfirst(<(0.5), real.(run.expect[1, :]))
        index === nothing || push!(jump_times, times[index] - 0.005)
    end
    @test length(jump_times) == 4000
    mean_time = sum(jump_times) / length(jump_times)
    @test mean_time ≈ 1 / gamma rtol=0.05
    @test sum(abs2, jump_times .- mean_time) / length(jump_times) ≈ 1 / gamma^2 rtol=0.1
end

@testset "mcsolve is unbiased when decay is fast" begin
    # gamma * (output spacing) = 0.4: a first-order fixed-step jump rule is
    # visibly biased here (it failed this test by more than 7 standard errors).
    gamma = 20.0
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 0.5; length=26))
    n_traj = 20_000

    result = mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector], n_traj, 2026)
    exact = exp.(-gamma .* times)
    deviation = abs.(real.(result.expect[1, :]) .- exact)
    scale = max.(result.expect_stderr[1, :], 1 / n_traj)

    @test result.solver_stats.success
    @test result.solver_stats.n_failed_trajectories == 0
    @test result.solver_stats.method == "waiting-time"
    @test result.solver_stats.max_step == 0.0
    @test result.solver_stats.rng == OpenQuantumSimJL.MCSOLVE_RNG_NAME
    @test result.solver_stats.base_seed == 2026
    @test result.solver_stats.n_jumps <= n_traj
    @test result.solver_stats.n_jumps > 0.99 * n_traj
    @test maximum(deviation ./ scale) < 4.5
    @test result.expect_stderr[1, 2:end] ≈ sqrt.(exact[2:end] .* (1 .- exact[2:end]) ./ n_traj) rtol=0.2
end

@testset "mcsolve matches mesolve for a driven, decaying qubit" begin
    omega, gamma = 1.0, 0.4
    H = ComplexF64[0 omega / 2; omega / 2 0]
    sm = ComplexF64[0 0; 1 0]
    sy = ComplexF64[0 -1im; 1im 0]
    psi0 = ComplexF64[1, 0]
    projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 10.0; length=41))
    n_traj = 6000

    exact = mesolve(H, psi0 * psi0', times, [sqrt(gamma) * sm], [projector, sy], 1e-10, 1e-12, false)
    sampled = mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector, sy], n_traj, 31)

    deviation = abs.(real.(sampled.expect) .- real.(exact.expect))
    scale = max.(sampled.expect_stderr, 1 / n_traj)
    @test maximum(deviation ./ scale) < 4.5
    @test sampled.solver_stats.success
    @test sampled.solver_stats.n_jumps > n_traj
end

@testset "mcsolve result is independent of the thread request" begin
    omega, gamma = 1.0, 0.4
    H = ComplexF64[0 omega / 2; omega / 2 0]
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 5.0; length=21))

    serial = mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector], 300, 77, 0.0, 1)
    all_threads = mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector], 300, 77, 0.0, -1)
    two = mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector], 300, 77, 0.0, 2)
    batched = mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector], 300, 77, 0.0, -1, "", 64, true)

    @test serial.solver_stats.n_workers == 1
    @test !serial.solver_stats.threaded
    @test all_threads.solver_stats.n_workers == Threads.nthreads()
    @test two.solver_stats.n_workers == min(2, Threads.nthreads())
    # Bit-identical, not merely close: every trajectory starts from the same
    # integrator state and its own random stream, whatever the work split.
    for other in (all_threads, two, batched)
        @test other.expect == serial.expect
        @test other.expect_stderr == serial.expect_stderr
        @test other.solver_stats.n_jumps == serial.solver_stats.n_jumps
    end

    # Tolerances are part of the result: tighter tolerances move jump times
    # only slightly, so the averages agree far below the statistical error.
    tight = mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector], 300, 77, 0.0, 1, "", 100, false, 1e-11, 1e-13)
    @test tight.expect ≈ serial.expect atol=1e-5
    vern = mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector], 300, 77, 0.0, 1, "", 100, false, 1e-10, 1e-12, "vern7")
    @test vern.solver_stats.ode_solver == "vern7"
    @test vern.expect ≈ tight.expect atol=1e-5
end

@testset "mcsolve edge cases and failure reporting" begin
    gamma = 0.35
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    projector = ComplexF64[1 0; 0 0]

    # A single output time, and repeated output times.
    single = mcsolve(H, psi0, [0.3], [sqrt(gamma) * sm], [projector], 10, 5)
    @test single.solver_stats.success
    @test real.(single.expect[1, :]) ≈ [1.0]
    repeated = mcsolve(H, psi0, [0.0, 1.0, 1.0, 2.0], [sqrt(gamma) * sm], [projector], 200, 5)
    @test repeated.expect[1, 2] == repeated.expect[1, 3]
    @test repeated.solver_stats.success

    # A dark state never jumps.
    dark = mcsolve(H, ComplexF64[0, 1], [0.0, 5.0, 10.0], [sqrt(gamma) * sm], [projector], 50, 5)
    @test dark.solver_stats.n_jumps == 0
    @test maximum(abs, dark.expect) < 1e-12

    # A trajectory that cannot be integrated is counted, not hidden.
    broken = ComplexF64[NaN 0; 0 0]
    failed = mcsolve(broken, psi0, [0.0, 1.0], [sqrt(gamma) * sm], [projector], 4, 5)
    @test !failed.solver_stats.success
    @test failed.solver_stats.n_failed_trajectories == 4
    @test failed.solver_stats.retcode != "Success"
    @test isnan(real(failed.expect[1, end]))
    lone = single_trajectory(broken, psi0, [0.0, 1.0], [sqrt(gamma) * sm], [projector], 5)
    @test !lone.solver_stats.success
    @test lone.solver_stats.n_failed_trajectories == 1

    @test_throws ArgumentError mcsolve(H, psi0, [0.0, 1.0], [sqrt(gamma) * sm], [projector], 4, 5, -0.1)
    @test_throws ArgumentError mcsolve(H, ComplexF64[0, 0], [0.0, 1.0], [sqrt(gamma) * sm], [projector], 4, 5)
    @test_throws ArgumentError mcsolve(H, psi0, [0.0, 1.0], [sqrt(gamma) * sm], [projector], 4, 5, 0.0, 1, "", 100, false, 1e-8, 1e-10, "euler")
end

@testset "mcsolve checkpoints are tied to solver settings" begin
    gamma = 0.35
    H = zeros(ComplexF64, 2, 2)
    sm = ComplexF64[0 0; 1 0]
    psi0 = ComplexF64[1, 0]
    projector = ComplexF64[1 0; 0 0]
    times = collect(range(0.0, 2.0; length=11))
    checkpoint_file = tempname() * ".h5"

    mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector], 20, 505, 0.0, 1, checkpoint_file, 10)
    resumed = mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector], 60, 505, 0.0, -1, checkpoint_file, 10)
    clean = mcsolve(H, psi0, times, [sqrt(gamma) * sm], [projector], 60, 505)
    @test resumed.solver_stats.resumed
    @test resumed.expect == clean.expect
    @test resumed.expect_stderr == clean.expect_stderr

    # Resuming with different tolerances would mix two different solvers.
    @test_throws ArgumentError mcsolve(
        H, psi0, times, [sqrt(gamma) * sm], [projector], 80, 505, 0.0, 1, checkpoint_file, 10, false, 1e-6, 1e-8,
    )
end

# ---------------------------------------------------------------------------
# Performance paths: they must give the same answers as the plain formulas.
# ---------------------------------------------------------------------------

function _reference_liouvillian(H, c_ops)
    d = size(H, 1)
    I_d = sparse(ComplexF64(1) * I, d, d)
    L = -1im * (kron(I_d, H) - kron(transpose(H), I_d))
    for c in c_ops
        cdgc = c' * c
        L += kron(conj(c), c) - 0.5 * kron(I_d, cdgc) - 0.5 * kron(transpose(cdgc), I_d)
    end
    return sparse(L)
end

@testset "Liouvillian assembly matches the Kronecker formula" begin
    rng = Xoshiro(11)
    for d in (1, 2, 7)
        # Non-Hermitian on purpose: the assembly must not assume Hermiticity.
        H = sprandn(rng, ComplexF64, d, d, 0.5)
        c_ops = [sprandn(rng, ComplexF64, d, d, 0.5) for _ in 1:3]
        reference = _reference_liouvillian(H, c_ops)
        @test maximum(abs, OpenQuantumSimJL.liouvillian(H, c_ops) - reference; init = 0.0) < 1e-12
        @test maximum(abs, OpenQuantumSimJL.liouvillian(Matrix(H), Matrix.(c_ops)) - reference; init = 0.0) < 1e-12
        op = OpenQuantumSimJL.liouvillian_operator(H, c_ops)
        x = randn(rng, ComplexF64, d * d)
        @test op * x ≈ reference * x
        @test size(op) == (d * d, d * d)
        @test sparse(op) ≈ reference
    end
    @test OpenQuantumSimJL.liouvillian(zeros(ComplexF64, 2, 2), Matrix{ComplexF64}[]) == spzeros(ComplexF64, 4, 4)
end

@testset "row-major products, single- and multi-threaded" begin
    rng = Xoshiro(12)
    # Large enough to take the threaded path when Julia has several threads.
    n = OpenQuantumSimJL.ROW_MAJOR_THREADING_MIN_ROWS + 1_234
    A = sprandn(rng, ComplexF64, n, n, 8 / n)
    op = OpenQuantumSimJL._matvec_operator(A)
    x = randn(rng, ComplexF64, n)
    y = zeros(ComplexF64, n)
    mul!(y, op, x)
    @test y ≈ A * x
    first_result = copy(y)
    mul!(y, op, x)
    @test y == first_result  # every row is summed in a fixed order
    z = randn(rng, ComplexF64, n)
    expected = (0.5 - 2im) * (A * x) + (3 + 1im) * z
    mul!(z, op, x, 0.5 - 2im, 3 + 1im)
    @test z ≈ expected
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, n - 1), op, x)
    @test_throws DimensionMismatch mul!(y, op, zeros(ComplexF64, n + 1))
    @test_throws DimensionMismatch mul!(y, op, zeros(ComplexF64, n + 1), 1, 0)
    @test OpenQuantumSimJL._matvec_operator(Matrix(A[1:3, 1:3])) isa Matrix
end

@testset "sparse trajectories: cavity decay" begin
    # A large, mostly-zero problem takes the sparse path. A one-photon cavity
    # state decays with P(1) = exp(-kappa t) whatever the truncation.
    n = 40
    a = OpenQuantumSimJL.destroy(n)
    H = a' * a
    kappa = 0.6
    psi0 = zeros(ComplexF64, n)
    psi0[2] = 1
    projector = spzeros(ComplexF64, n, n)
    projector[2, 2] = 1
    times = collect(range(0.0, 2.0; length = 11))
    H_used, c_used, e_used = OpenQuantumSimJL._prepare_trajectory_operators(H, [sqrt(kappa) * a], [projector])
    @test H_used isa SparseMatrixCSC
    @test eltype(c_used) <: SparseMatrixCSC
    @test eltype(e_used) <: SparseMatrixCSC

    result = mcsolve(H, psi0, times, [sqrt(kappa) * a], [projector], 4000, 31, 0.0, 1)
    exact = exp.(-kappa .* times)
    @test maximum(abs.(real.(result.expect[1, :]) .- exact) ./ max.(result.expect_stderr[1, :], 1 / 4000)) < 5

    # A small or dense problem stays dense, and both forms hash alike.
    small_H, small_c, _ = OpenQuantumSimJL._prepare_trajectory_operators(
        OpenQuantumSimJL.sigmaz(), [OpenQuantumSimJL.sigmam()], Matrix{ComplexF64}[],
    )
    @test small_H isa Matrix
    @test eltype(small_c) <: Matrix
    hash_of(op) = bytes2hex(take!(OpenQuantumSimJL._hash_array!(IOBuffer(), "H", op)))
    @test hash_of(Matrix(H)) == hash_of(H)
end

@testset "time-dependent superoperator path matches constant Hamiltonians" begin
    # With a constant coefficient, H0 + f H1 is time independent, so the
    # time-dependent solver must agree with mesolve on H0 + f H1.
    n = 20
    a = kron(OpenQuantumSimJL.destroy(n), OpenQuantumSimJL.eye(2))
    sm = kron(OpenQuantumSimJL.eye(n), OpenQuantumSimJL.sigmam())
    H0 = a' * a + 0.5 * kron(OpenQuantumSimJL.eye(n), OpenQuantumSimJL.sigmaz()) + 0.1 * (a' * sm + a * sm')
    H1 = a + a'
    c_ops = [sqrt(0.05) * a, sqrt(0.02) * sm]
    rho0 = zeros(ComplexF64, 2n, 2n)
    rho0[1, 1] = 1
    times = collect(range(0.0, 4.0; length = 9))
    drive = 0.3
    td = mesolve_time_dependent(
        Matrix(H0), [Matrix(H1)], [drive], rho0, times, Matrix.(c_ops), [Matrix(a' * a)], 1e-9, 1e-11, false, false,
    )
    @test td.solver_stats.superoperator
    ti = mesolve(H0 + drive * H1, rho0, times, c_ops, [a' * a], 1e-9, 1e-11, false, "ode", 30, false)
    @test td.expect ≈ ti.expect atol = 1e-7

    # A small problem keeps the dense form and gives the same physics.
    small = mesolve_time_dependent(
        ComplexF64[1 0; 0 -1], [ComplexF64[0 1; 1 0]], [drive], ComplexF64[1 0; 0 0], times,
        Matrix{ComplexF64}[], [ComplexF64[1 0; 0 0]], 1e-9, 1e-11, false, false,
    )
    @test !small.solver_stats.superoperator
end

@testset "Bloch-Redfield eigenbasis propagation matches the generator" begin
    n = 5
    a = Matrix(kron(OpenQuantumSimJL.destroy(n), OpenQuantumSimJL.eye(2)))
    sm = Matrix(kron(OpenQuantumSimJL.eye(n), OpenQuantumSimJL.sigmam()))
    H = a' * a + 0.5 * Matrix(kron(OpenQuantumSimJL.eye(n), OpenQuantumSimJL.sigmaz())) + 0.08 * (a' * sm + a * sm')
    psi = zeros(ComplexF64, 2n)
    psi[3] = 1
    rho0 = psi * psi'
    thermal(w) = w == 0 ? 0.0 : (w < 0 ? 0.1 * (1 + 1 / (exp(-w / 0.5) - 1)) : 0.1 / (exp(w / 0.5) - 1))
    flat(w) = w < 0 ? 0.05 : 0.0
    times = collect(range(0.0, 10.0; length = 11))

    br = brmesolve(H, rho0, times, [a, sm], [thermal, flat], [a' * a], 1e-10, 1e-12, true, "ode", 30, false)
    # Propagate the computational-basis generator directly as a reference.
    L = OpenQuantumSimJL.br_liouvillian(H, [a, sm], [thermal, flat])
    reference, _, _ = OpenQuantumSimJL._solve_liouvillian(L, vec(copy(rho0)), times; rtol = 1e-10, atol = 1e-12)
    for idx in eachindex(times)
        @test br.states[:, :, idx] ≈ reshape(reference[idx], 2n, 2n) atol = 1e-8
    end
    @test real(br.expect[1, end]) ≈ real(tr(a' * a * reshape(reference[end], 2n, 2n))) atol = 1e-8

    # Grouping by sorted frequency keeps degenerate transitions together.
    groups = OpenQuantumSimJL._br_channel_groups([0.0, 1.0, 2.0])
    sizes = sort(length.(groups))
    @test sizes == [1, 1, 2, 2, 3]
    @test sum(sizes) == 9
end
