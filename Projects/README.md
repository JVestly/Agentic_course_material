# Reinforcement Learning for Quantum and Molecular Simulation

## Project set

This project set explores how reinforcement learning can be used as a computational tool in quantum mechanics and molecular dynamics.

The common theme is **adaptive scientific computation**. In many physics simulations, one does not merely compute a quantity once. One repeatedly chooses parameters, sampling moves, control variables, or simulation conditions. These choices influence the future trajectory of the computation.

Reinforcement learning provides a framework for such sequential decisions.

Students choose **one** of the following project pathways:

1. **Quantum pathway:** RL-assisted variational Monte Carlo for interacting particles in a harmonic trap.
2. **Molecular-dynamics pathway I:** RL-controlled cooling of a Lennard-Jones cluster.
3. **Molecular-dynamics pathway II:** RL-enhanced sampling of rare transitions in a molecular system.

Each pathway is a complete project. All projects require a physical simulator, baseline methods, an RL formulation, numerical experiments, and a written report.

---

# Common learning goals

After completing one pathway, you should be able to:

- formulate a computational physics problem as a sequential decision problem;
- identify states, actions, rewards, transitions, and returns;
- implement a simulator that acts as an RL environment;
- implement at least one value-based or policy-based RL method;
- compare RL against non-RL baselines using a fair computational budget;
- discuss whether the learned behavior is physically meaningful;
- identify when RL is useful and when simpler methods are preferable.

---

# Common RL background

At each time step, an agent observes a state

$$
S_t,
$$

chooses an action

$$
A_t,
$$

and receives a reward and next state

$$
(R_{t+1}, S_{t+1}).
$$

The return is

$$
G_t =
\sum_{k=0}^{\infty}
\gamma^k R_{t+k+1},
$$

where $\gamma \in [0,1]$ is the discount factor.

The goal is to learn a policy

$$
\pi(a|s)
=
P(A_t=a|S_t=s)
$$

that gives large expected return.

For value-based methods, the action-value function is

$$
q_\pi(s,a)
=
\mathbb E_\pi
[
G_t
\mid
S_t=s,A_t=a
].
$$

In Q-learning, one uses the update

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma \max_a Q(S_{t+1},a)
-
Q(S_t,A_t)
\right].
$$

In deep Q-learning, the table is replaced by a neural network

$$
Q(s,a) \approx \hat q(s,a;w).
$$

A standard DQN target is

$$
Y_t
=
R_{t+1}
+
\gamma
\max_a
\hat q(S_{t+1},a;w^-),
$$

where $w^-$ denotes target-network parameters.

The loss is

$$
L(w)
=
\left[
Y_t
-
\hat q(S_t,A_t;w)
\right]^2.
$$

For policy-gradient or actor-critic methods, the policy itself is parametrized,

$$
\pi_\theta(a|s),
$$

and the parameters are updated to increase expected return.

---

# Common deliverables

For all pathways, the final submission must contain:

- a scientific report;
- a working code base;
- a README explaining how to run the code;
- plots and tables;
- comparison against non-RL baselines;
- discussion of physical interpretation;
- discussion of RL formulation;
- reproducibility information.

The report should include:

1. introduction;
2. theory;
3. numerical method;
4. RL formulation;
5. implementation details;
6. results;
7. discussion;
8. conclusion;
9. references;
10. appendix with selected code snippets.

---

# Common reproducibility requirements

The report must state:

- programming language;
- package versions;
- random seeds;
- number of particles or degrees of freedom;
- simulation timestep;
- number of Monte Carlo or molecular-dynamics steps;
- number of RL episodes;
- reward function;
- discount factor;
- exploration schedule;
- neural-network architecture, if used;
- optimizer and learning rate;
- total computational budget.

The RL method must be compared against baselines using a fair budget.

For Monte Carlo projects, compare methods using approximately the same total number of Monte Carlo samples.

For molecular-dynamics projects, compare methods using approximately the same number of force evaluations.

---

# Pathway 1: RL-assisted variational Monte Carlo for interacting particles

## Scientific motivation

Variational Monte Carlo is a method for estimating ground-state energies of quantum systems. One chooses a trial wave function with variational parameters and evaluates the expectation value of the Hamiltonian by Monte Carlo sampling.

The variational principle states that

$$
E_0 \leq E(\theta),
$$

where $E_0$ is the true ground-state energy and

$$
E(\theta)
=
\frac{
\langle \Psi_\theta|\hat H|\Psi_\theta\rangle
}{
\langle \Psi_\theta|\Psi_\theta\rangle
}.
$$

The computational task is to find parameters $\theta$ that make $E(\theta)$ small.

This project studies whether reinforcement learning can learn how to tune variational and Monte Carlo parameters during the computation.

The RL agent controls the optimization process. It observes energy estimates, variances, acceptance rates, and current parameters. It then chooses how to modify the variational parameters or the Monte Carlo proposal step.

---

## Physical system

Consider $N$ interacting particles in a two-dimensional harmonic oscillator trap.

The Hamiltonian is

$$
\hat H
=
\sum_{i=1}^{N}
\left(
-\frac{1}{2}\nabla_i^2
+
\frac{1}{2}\omega^2 r_i^2
\right)
+
\sum_{i<j}
\frac{1}{r_{ij}}.
$$

Here

$$
r_i = |\mathbf r_i|,
$$

and

$$
r_{ij}=|\mathbf r_i-\mathbf r_j|.
$$

Use atomic units, so that

$$
\hbar = m = e = 1.
$$

The simplest version uses

$$
N=2.
$$

More advanced versions may use larger $N$.

---

## Trial wave functions

Start with a Gaussian trial wave function

$$
\Psi_\alpha(\mathbf R)
=
\exp
\left(
-\frac{\alpha \omega}{2}
\sum_{i=1}^{N}
r_i^2
\right),
$$

where

$$
\mathbf R=(\mathbf r_1,\ldots,\mathbf r_N)
$$

is the full particle configuration.

The variational parameter is

$$
\alpha.
$$

A more advanced trial wave function includes a Jastrow factor:

$$
\Psi_{\alpha,\beta}(\mathbf R)
=
\exp
\left(
-\frac{\alpha \omega}{2}
\sum_{i=1}^{N} r_i^2
\right)
\prod_{i<j}
\exp
\left(
\frac{a r_{ij}}{1+\beta r_{ij}}
\right).
$$

Here $\alpha$ and $\beta$ are variational parameters.

The project may be completed with either:

- the one-parameter wave function $\Psi_\alpha$;
- the two-parameter wave function $\Psi_{\alpha,\beta}$.

---

## Local energy

The local energy is

$$
E_L(\mathbf R)
=
\frac{1}{\Psi_\theta(\mathbf R)}
\hat H
\Psi_\theta(\mathbf R).
$$

The variational energy is

$$
E(\theta)
=
\frac{
\int d\mathbf R
|\Psi_\theta(\mathbf R)|^2
E_L(\mathbf R)
}{
\int d\mathbf R
|\Psi_\theta(\mathbf R)|^2
}.
$$

In Monte Carlo form,

$$
E(\theta)
\approx
\frac{1}{M}
\sum_{k=1}^{M}
E_L(\mathbf R_k),
$$

where

$$
\mathbf R_k \sim |\Psi_\theta(\mathbf R)|^2.
$$

The variance of the local energy is

$$
\sigma_E^2
=
\langle E_L^2\rangle
-
\langle E_L\rangle^2.
$$

A good trial wave function should have low energy and low local-energy variance.

---

## Metropolis sampling

Use the Metropolis algorithm to sample configurations.

Given the current configuration $\mathbf R$, propose

$$
\mathbf R'
=
\mathbf R
+
\delta \xi,
$$

where $\xi$ is a random displacement and $\delta$ is a proposal step length.

Accept the move with probability

$$
A(\mathbf R\rightarrow \mathbf R')
=
\min
\left(
1,
\frac{
|\Psi_\theta(\mathbf R')|^2
}{
|\Psi_\theta(\mathbf R)|^2
}
\right).
$$

The proposal length $\delta$ affects the quality of the sampling.

If $\delta$ is too small, the Markov chain moves slowly.

If $\delta$ is too large, most proposed moves are rejected.

Therefore, $\delta$ is a natural variable for an RL agent to control.

---

## RL formulation

One RL step corresponds to one block of Monte Carlo sampling.

At each RL step, the agent:

1. observes the current state;
2. chooses an action;
3. updates a variational or sampling parameter;
4. runs one Monte Carlo block;
5. receives a reward;
6. moves to the next state.

One episode is one complete variational optimization run.

---

## State space

Define the state as

$$
S_t =
(
\hat E_t,
\hat \sigma_t^2,
A_t^{\mathrm{acc}},
\alpha_t,
\beta_t,
\delta_t,
\Delta \hat E_t,
t/T
).
$$

Here:

- $\hat E_t$ is the estimated energy in block $t$;
- $\hat \sigma_t^2$ is the estimated local-energy variance;
- $A_t^{\mathrm{acc}}$ is the Metropolis acceptance rate;
- $\alpha_t$ and $\beta_t$ are variational parameters;
- $\delta_t$ is the proposal step length;
- $\Delta \hat E_t = \hat E_t-\hat E_{t-1}$;
- $t/T$ is the normalized optimization time.

For the one-parameter wave function, remove $\beta_t$.

All state variables should be normalized before being used by a neural network.

---

## Action space

Use a discrete action space.

For the one-parameter wave function, use

$$
\mathcal A
=
\{
\alpha \uparrow,
\alpha \downarrow,
\delta \uparrow,
\delta \downarrow,
\text{do nothing}
\}.
$$

For the two-parameter wave function, use

$$
\mathcal A
=
\{
\alpha \uparrow,
\alpha \downarrow,
\beta \uparrow,
\beta \downarrow,
\delta \uparrow,
\delta \downarrow,
\text{do nothing}
\}.
$$

For example,

$$
\alpha \leftarrow \alpha + \Delta \alpha,
$$

$$
\beta \leftarrow \beta + \Delta \beta,
$$

$$
\delta \leftarrow \delta + \Delta \delta.
$$

Parameters must remain inside predefined intervals, for example

$$
\alpha \in [0.2,2.5],
$$

$$
\beta \in [0.01,2.0],
$$

$$
\delta \in [0.01,5.0].
$$

---

## Reward function

The reward should encourage energy reduction, low variance, and useful sampling.

Use the dense reward

$$
R_{t+1}
=
(\hat E_t-\hat E_{t+1})
-
\lambda_\sigma \hat \sigma_{t+1}^2
-
\lambda_A
\left(
A_{t+1}^{\mathrm{acc}}-A^\star
\right)^2.
$$

Here $A^\star$ is a target acceptance rate, for example

$$
A^\star = 0.5.
$$

The first term rewards energy improvement.

The second term penalizes noisy estimates.

The third term penalizes very poor sampling.

Also test a terminal reward

$$
R_T
=
-\hat E_T
-
\lambda_\sigma \hat \sigma_T^2.
$$

The report should compare dense and terminal reward formulations.

---

## Baseline methods

Compare the RL method against at least two baselines.

### Baseline 1: grid search

Evaluate

$$
\hat E(\alpha)
$$

on a fixed grid.

For the two-parameter case, evaluate

$$
\hat E(\alpha,\beta)
$$

on a two-dimensional grid.

Report the best energy found under the same sampling budget.

### Baseline 2: fixed Metropolis proposal

Use a manually chosen fixed $\delta$.

Report:

- energy;
- variance;
- acceptance rate;
- autocorrelation if estimated.

### Baseline 3: acceptance-rate heuristic

Use the rule

$$
\delta \leftarrow
\begin{cases}
1.05\delta, & A^{\mathrm{acc}}>0.55,\\
0.95\delta, & A^{\mathrm{acc}}<0.45,\\
\delta, & \text{otherwise}.
\end{cases}
$$

This is a hand-designed controller, not an RL method.

---

## RL algorithms

Implement at least one of the following.

### Option A: tabular Q-learning

Discretize the state variables into bins.

For example, bin:

- energy change;
- variance;
- acceptance rate;
- proposal step;
- variational parameter.

Use the update

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha_{\mathrm{RL}}
\left[
R_{t+1}
+
\gamma \max_a Q(S_{t+1},a)
-
Q(S_t,A_t)
\right].
$$

### Option B: DQN

Approximate

$$
Q(s,a)
\approx
\hat q(s,a;w).
$$

Use:

- replay buffer;
- target network;
- $\epsilon$-greedy exploration;
- mini-batch gradient updates;
- normalized state variables.

The DQN target is

$$
Y_t
=
R_{t+1}
+
\gamma
\max_a
\hat q(S_{t+1},a;w^-).
$$

The loss is

$$
L(w)
=
\left[
Y_t-\hat q(S_t,A_t;w)
\right]^2.
$$

---

## Required tasks

### Task 1: Implement the variational Monte Carlo solver

Implement:

- trial wave function;
- probability density;
- local energy;
- Metropolis move;
- energy estimator;
- variance estimator;
- acceptance-rate estimator.

Validate the implementation using a noninteracting or analytically simple limiting case.

---

### Task 2: Study the energy landscape

Compute energy estimates for fixed parameter values.

Produce plots of:

- $\hat E(\alpha)$;
- $\hat E(\alpha,\beta)$ if using two parameters;
- local-energy variance;
- acceptance rate versus proposal step length.

Discuss how Monte Carlo noise affects the apparent energy landscape.

---

### Task 3: Build the RL environment

Implement an environment with:

```python
reset()
step(action)