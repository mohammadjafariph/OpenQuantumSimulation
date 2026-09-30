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

``mcsolve`` samples the standard MCWF (quantum-jump) unraveling: between jumps
the state evolves under the effective non-Hermitian Hamiltonian
:math:`H_\text{eff} = H - \tfrac{i}{2}\sum_k C_k^\dagger C_k`, and jumps occur
with probabilities determined by the collapse operators. Averaging over
independent trajectories reproduces the Lindblad master equation in the limit
of many trajectories; the reported standard error is the standard error of the
mean over trajectories.

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

where :math:`\gamma(\omega)` is the user-supplied spectrum. Because
:math:`A(\omega)` partitions :math:`A`, a constant spectrum
:math:`\gamma(\omega) = g` is exactly equivalent to a Lindblad collapse
operator :math:`\sqrt{g}\,A` — the sanity check used in the test suite. Decay
(emission) channels sit at :math:`\omega < 0`; a zero-temperature bath
therefore assigns weight only to negative frequencies.

Quantum Regression Theorem
--------------------------

Two-time correlations :math:`\langle A(\tau) B(0)\rangle` are computed with
the quantum regression theorem: the state is evolved to :math:`t=0`, then the
operator product :math:`B\rho(0) B^\dagger` (with the regression construction
applied to each collapse operator) is propagated with the same Liouvillian as
the state itself. ``spectrum_2op_1t`` transforms the sampled correlation with a
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
