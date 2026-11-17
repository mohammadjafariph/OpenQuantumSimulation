Theory Notes
============

Short derivations and conventions behind the solvers, so the numerics can be
checked against the physics. Throughout, :math:`H` is the system Hamiltonian,
:math:`\{C_k\}` are collapse operators, and :math:`\rho` is the density
matrix.

Lindblad Dynamics
-----------------

For a Markovian bath, the state obeys the Lindblad master equation

.. math::

   \dot\rho = -i[H, \rho] + \sum_k \left( C_k \rho C_k^\dagger
   - \tfrac{1}{2}\{C_k^\dagger C_k, \rho\} \right).

``mesolve`` integrates this equation with an adaptive Runge-Kutta scheme
(Tsit5) or, for time-independent problems, with a Krylov-subspace
exponential of the Liouvillian. The Liouvillian is assembled in the
column-stacked vectorization convention,

.. math::

   \operatorname{vec}(A X B) = (B^\mathsf{T} \otimes A)\,
   \operatorname{vec}(X),

which is the same convention used by :func:`openquantumsim.lindblad_superoperator`
and :func:`openquantumsim.apply_superoperator`.

Monte Carlo Wave Function
-------------------------

``mcsolve`` samples the standard MCWF (quantum-jump) unraveling with the
waiting-time algorithm. For each trajectory:

1. Draw a threshold :math:`r` uniformly from :math:`[0, 1)`.
2. Evolve the unnormalised state under the effective non-Hermitian Hamiltonian
   :math:`H_\text{eff} = H - \tfrac{i}{2}\sum_k C_k^\dagger C_k`. Its squared
   norm decays, and equals the probability that no jump has happened yet.
3. At the instant :math:`\lVert\psi(t)\rVert^2 = r`, apply collapse operator
   :math:`C_k` with probability proportional to
   :math:`\lVert C_k \psi\rVert^2`, normalise, draw a new threshold, and
   continue.

The no-jump evolution uses an adaptive Runge-Kutta integrator, and the jump
instant is located by root-finding on the integrator's interpolant. The
accuracy of a trajectory is therefore set by ``Options.rtol`` and
``Options.atol``; there is no time step to choose, and no bias when decay
rates are fast compared with the output grid. ``Options.max_step`` only caps
the adaptive step.

Averaging over independent trajectories reproduces the Lindblad master
equation in the limit of many trajectories; the reported standard error is the
standard error of the mean over trajectories. Each trajectory draws from its
own Xoshiro256++ stream, seeded from the SHA-256 digest of the run seed and
the trajectory index. A run is therefore reproducible from ``Options.seed``
alone, gives identical numbers on any number of threads, and continues with
the same streams when resumed from a checkpoint.

Bloch-Redfield Equation
-----------------------

For weak system-bath coupling, ``brmesolve`` derives a Redfield tensor from the
bath spectra and then applies the secular approximation: system transitions are
grouped by Bohr frequency
:math:`\omega_{mn} = E_m - E_n`, and only terms oscillating at equal frequency
are kept. With :math:`A(\omega)` collecting the matrix elements of a coupling
operator between states whose energy difference is :math:`\omega`, each channel
contributes

.. math::

   \mathcal{D}_\omega[\rho] = \gamma(\omega) \left[
   A(\omega) \rho A^\dagger(\omega)
   - \tfrac{1}{2}\{A^\dagger(\omega) A(\omega), \rho\} \right],

where :math:`\gamma(\omega)` is the user-supplied spectrum. The channels
:math:`A(\omega)` sum to :math:`A`, but a Lindblad collapse operator
:math:`\sqrt{g}\,A` also produces cross terms
:math:`A(\omega)\rho A^\dagger(\omega')` between different frequencies,
which the secular approximation drops. A constant spectrum
:math:`\gamma(\omega) = g` is therefore equivalent to :math:`\sqrt{g}\,A`
only when :math:`A` connects a single Bohr frequency, such as
:math:`\sigma_-` for a qubit; that is the case the test suite checks. For
:math:`\sigma_x` on a qubit, which connects :math:`\pm\omega_0`, the two
differ. Decay
(emission) channels sit at :math:`\omega < 0`; a zero-temperature bath
therefore assigns weight only to negative frequencies.

In the energy eigenbasis the coherent part is diagonal and each channel
:math:`A(\omega)` has one entry per transition at that frequency, so the
generator is sparse there. ``brmesolve`` builds it in that basis, propagates
the state there, and rotates only the outputs back; the rotation is unitary,
so trace, Hermiticity, eigenvalues and entropy are computed without it.

Quantum Regression Theorem
--------------------------

Two-time correlations :math:`\langle A(\tau) B(0)\rangle` are computed with
the quantum regression theorem: the operator product :math:`B\rho(0)` is
propagated over :math:`\tau` with the same Liouvillian :math:`\mathcal{L}` as
the state itself, and the correlation is the trace against :math:`A`,

.. math::

   \langle A(\tau) B(0)\rangle
   = \operatorname{Tr}\!\left[ A\, e^{\mathcal{L}\tau} \bigl(B\rho(0)\bigr)
   \right].

``correlation_2op_1t`` uses the supplied ``rho0`` as :math:`\rho(0)`;
``correlation_2op_2t`` first evolves the state to each :math:`t` with
``mesolve`` and applies the same construction to :math:`B\rho(t)`.
``spectrum_2op_1t`` transforms the sampled correlation with a
discrete Fourier transform, so the result is the discrete-time spectrum of the
sampled function rather than its continuum limit.

Secular Approximation and Degeneracies
--------------------------------------

The secular approximation is exact for nondegenerate transition frequencies and
an approximation otherwise. When several transitions share a Bohr frequency,
``brmesolve`` groups them into a single channel, which makes the generator
time-independent in the eigenbasis; the grouping is basis-independent for
nondegenerate gaps, and users with (near-)degenerate spectra should compare
against :func:`openquantumsim.mesolve` for their parameter regime.
