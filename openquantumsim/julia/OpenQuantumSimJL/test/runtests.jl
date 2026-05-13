using Test
using LinearAlgebra
using OpenQuantumSimJL

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
