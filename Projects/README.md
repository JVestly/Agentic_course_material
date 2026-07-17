# Project 1: Reinforcement Learning for Variational Quantum Monte Carlo

## Reinforcement Learning and Agentic AI for Computational Physics

## Overview

In this project you will study how reinforcement learning can be used as an adaptive decision-making tool in variational quantum Monte Carlo.

Variational quantum Monte Carlo is a method for estimating ground-state energies of quantum systems. The method depends on a trial wave function with variational parameters. These parameters must be chosen so that the energy expectation value becomes as small as possible.

The reinforcement-learning component in this project does not replace quantum mechanics, Monte Carlo sampling, or the variational principle. Instead, it learns how to make numerical decisions during the computation.

The agent may control:

- variational parameters;
- Monte Carlo proposal step lengths;
- optimization step sizes;
- sampling strategies;
- or combinations of these.

The project combines three themes:

1. quantum many-body physics;
2. stochastic Monte Carlo estimation;
3. reinforcement learning as adaptive control of a scientific computation.

The main scientific question is:

> Can an RL agent learn a useful strategy for navigating a noisy variational Monte Carlo optimization problem?

---

# Learning goals

After completing this project, you should be able to:

- explain the variational principle for quantum systems;
- implement a variational Monte Carlo solver;
- compute local energies and Monte Carlo estimates;
- implement Metropolis sampling;
- formulate a computational physics problem as an RL environment;
- define states, actions, rewards, and episodes;
- implement a tabular or deep RL agent;
- compare RL against non-RL baselines;
- evaluate whether the learned policy is physically and computationally meaningful.

---

# Physical background

## The variational principle

Let $\hat H$ be the Hamiltonian of a quantum system. The ground-state energy is

$$
E_0 =
\min_{\Psi}
\frac{
\langle \Psi | \hat H | \Psi \rangle
}{
\langle \Psi | \Psi \rangle
}.
$$

If we choose a trial wave function $\Psi_\theta$, depending on variational parameters $\theta$, then the variational energy is

$$
E(\theta)
=
\frac{
\langle \Psi_\theta | \hat H | \Psi_\theta \rangle
}{
\langle \Psi_\theta | \Psi_\theta \rangle
}.
$$

The variational principle gives

$$
E_0 \leq E(\theta).
$$

Therefore, minimizing $E(\theta)$ gives the best approximation to the ground-state energy within the chosen family of trial wave functions.

The computational problem is:

$$
\theta^*
\in
\arg\min_{\theta}
E(\theta).
$$

In this project, reinforcement learning is used to help search for good values of $\theta$.

---

# Physical system

Consider $N$ interacting particles in a two-dimensional harmonic oscillator trap.

Use atomic units:

$$
\hbar = 1,
\qquad
m = 1,
\qquad
e = 1.
$$

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
r_{ij}
=
|\mathbf r_i-\mathbf r_j|.
$$

The first term is the kinetic energy. The second term is the harmonic confinement. The final term is the Coulomb interaction.

A minimal implementation may use

$$
N=2.
$$

A more advanced implementation may study larger systems such as

$$
N=4,
\qquad
N=6,
\qquad
N=8.
$$

The oscillator frequency $\omega$ may be fixed at

$$
\omega = 1.
$$

As an extension, you may investigate other values of $\omega$.

---

# Trial wave functions

## One-parameter Gaussian wave function

Start with the trial wave function

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
\mathbf R =
(\mathbf r_1,\ldots,\mathbf r_N)
$$

is the full many-particle configuration.

The variational parameter is $\alpha$.

The parameter $\alpha$ controls the spatial width of the wave function.

If $\alpha$ is too large, the particles are too localized near the center of the trap.

If $\alpha$ is too small, the particles are too spread out.

The goal is to find a value of $\alpha$ that minimizes the variational energy.

---

## Optional two-parameter Jastrow wave function

A more advanced trial wave function includes correlations between particles through a Jastrow factor:

$$
\Psi_{\alpha,\beta}(\mathbf R)
=
\exp
\left(
-\frac{\alpha \omega}{2}
\sum_{i=1}^{N}
r_i^2
\right)
\prod_{i<j}
\exp
\left(
\frac{a r_{ij}}{1+\beta r_{ij}}
\right).
$$

Here $\alpha$ and $\beta$ are variational parameters.

The Gaussian part describes the trapping potential.

The Jastrow factor introduces particle-particle correlations.

The parameter $\beta$ controls the range and strength of the correlation correction.

For a basic project, use only $\Psi_\alpha$.

For a stronger project, use $\Psi_{\alpha,\beta}$.

---

# Local energy

The local energy is defined by

$$
E_L(\mathbf R)
=
\frac{1}{\Psi_\theta(\mathbf R)}
\hat H
\Psi_\theta(\mathbf R).
$$

The variational energy can be written as an expectation value over the probability distribution

$$
p_\theta(\mathbf R)
=
\frac{
|\Psi_\theta(\mathbf R)|^2
}{
\int d\mathbf R |\Psi_\theta(\mathbf R)|^2
}.
$$

Thus,

$$
E(\theta)
=
\int d\mathbf R\,
p_\theta(\mathbf R)
E_L(\mathbf R).
$$

In Monte Carlo form,

$$
E(\theta)
\approx
\frac{1}{M}
\sum_{k=1}^{M}
E_L(\mathbf R_k),
$$

where the configurations $\mathbf R_k$ are sampled from $p_\theta(\mathbf R)$.

The variance of the local energy is

$$
\sigma_E^2
=
\langle E_L^2 \rangle
-
\langle E_L \rangle^2.
$$

For an exact eigenstate, the local energy is constant and the variance is zero. Therefore, the variance is also a useful diagnostic of trial-wave-function quality.

---

# Monte Carlo sampling

## Metropolis algorithm

Use the Metropolis algorithm to sample configurations from

$$
|\Psi_\theta(\mathbf R)|^2.
$$

Given a current configuration $\mathbf R$, propose a new configuration

$$
\mathbf R'
=
\mathbf R + \delta \xi,
$$

where $\xi$ is a random displacement and $\delta$ is the proposal step length.

The proposed move is accepted with probability

$$
A(\mathbf R \rightarrow \mathbf R')
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

If the move is accepted, set

$$
\mathbf R \leftarrow \mathbf R'.
$$

If the move is rejected, keep the old configuration.

---

## Acceptance rate

The acceptance rate is

$$
A^{\mathrm{acc}}
=
\frac{
\text{number of accepted moves}
}{
\text{number of proposed moves}
}.
$$

The proposal step $\delta$ affects the acceptance rate.

If $\delta$ is too small, most moves are accepted, but the Markov chain moves slowly.

If $\delta$ is too large, most moves are rejected.

A reasonable acceptance rate is often somewhere between $0.3$ and $0.7$, but the optimal value depends on the problem.

In this project, $\delta$ may be treated as a control variable chosen by an RL agent.

---

# Why reinforcement learning?

The VMC optimization is noisy and sequential.

The energy estimate at one step depends on:

- the current variational parameters;
- the Monte Carlo samples;
- the proposal step length;
- the previous parameter choices;
- random fluctuations in the sampling.

A simple optimizer may overreact to noise or require careful hand tuning.

An RL agent can instead learn a policy that maps the current status of the computation to useful numerical decisions.

For example, the agent may learn:

- to increase $\alpha$ when the energy decreases in that direction;
- to decrease the proposal step when the acceptance rate is too low;
- to increase the proposal step when the chain moves too slowly;
- to stop changing parameters once the energy stops improving;
- to prefer conservative updates when the variance is large.

---

# RL formulation

The VMC calculation is divided into blocks.

Each RL step corresponds to one block of Monte Carlo sampling.

At RL step $t$:

1. the agent observes the current state $S_t$;
2. the agent chooses an action $A_t$;
3. the chosen action modifies a parameter such as $\alpha$, $\beta$, or $\delta$;
4. a block of Monte Carlo sampling is performed;
5. the energy, variance, and acceptance rate are estimated;
6. the agent receives a reward $R_{t+1}$;
7. the next state $S_{t+1}$ is constructed.

One episode is one full VMC optimization run.

---

# State space

For the one-parameter wave function, use the state

$$
S_t =
(
\hat E_t,
\hat \sigma_t^2,
A_t^{\mathrm{acc}},
\alpha_t,
\delta_t,
\Delta \hat E_t,
t/T
).
$$

Here:

- $\hat E_t$ is the estimated energy in block $t$;
- $\hat \sigma_t^2$ is the estimated local-energy variance;
- $A_t^{\mathrm{acc}}$ is the acceptance rate;
- $\alpha_t$ is the variational parameter;
- $\delta_t$ is the Metropolis proposal step;
- $\Delta \hat E_t = \hat E_t-\hat E_{t-1}$;
- $t/T$ is the normalized episode time.

For the two-parameter wave function, use

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

All components of the state should be normalized before being used by a neural network.

A simple normalization is

$$
\tilde x =
\frac{x-\mu_x}{\sigma_x+\epsilon},
$$

where $\mu_x$ and $\sigma_x$ are estimated from training data.

---

# Action space

Use a discrete action space.

For the one-parameter wave function, define

$$
\mathcal A =
\{
\alpha \uparrow,
\alpha \downarrow,
\delta \uparrow,
\delta \downarrow,
\text{do nothing}
\}.
$$

For example,

$$
\alpha \uparrow:
\quad
\alpha \leftarrow \alpha + \Delta \alpha,
$$

$$
\alpha \downarrow:
\quad
\alpha \leftarrow \alpha - \Delta \alpha,
$$

$$
\delta \uparrow:
\quad
\delta \leftarrow \delta + \Delta \delta,
$$

$$
\delta \downarrow:
\quad
\delta \leftarrow \delta - \Delta \delta.
$$

For the two-parameter wave function, use

$$
\mathcal A =
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

All parameters must remain inside allowed intervals.

Suggested bounds are

$$
\alpha \in [0.2,2.5],
$$

$$
\beta \in [0.01,2.0],
$$

$$
\delta \in [0.01,5.0].
$$

A simple choice is

$$
\Delta \alpha = 0.02,
$$

$$
\Delta \beta = 0.02,
$$

$$
\Delta \delta = 0.05.
$$

You may also use multiplicative updates, such as

$$
\alpha \leftarrow \alpha(1 \pm \epsilon_\alpha),
$$

and

$$
\delta \leftarrow \delta(1 \pm \epsilon_\delta).
$$

---

# Reward functions

The reward should encourage useful optimization behavior.

You must test at least two reward functions.

---

## Reward 1: dense energy-improvement reward

A simple dense reward is

$$
R_{t+1}
=
\hat E_t-\hat E_{t+1}.
$$

This reward is positive when the estimated energy decreases.

Because Monte Carlo estimates are noisy, this reward can have high variance.

---

## Reward 2: energy and variance reward

A more stable reward includes a variance penalty:

$$
R_{t+1}
=
(\hat E_t-\hat E_{t+1})
-
\lambda_\sigma \hat \sigma_{t+1}^2.
$$

The parameter $\lambda_\sigma$ controls how strongly the agent is penalized for noisy local-energy estimates.

---

## Reward 3: energy, variance, and sampling-quality reward

A more complete reward is

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

Here $A^\star$ is a target acceptance rate.

A reasonable starting value is

$$
A^\star = 0.5.
$$

This reward encourages the agent to lower the energy while maintaining useful sampling.

---

## Reward 4: terminal reward

A terminal reward is given only at the end of the episode:

$$
R_T
=
-\hat E_T
-
\lambda_\sigma \hat \sigma_T^2.
$$

All intermediate rewards are zero.

Terminal rewards are conceptually clean but harder to learn from, because the agent receives delayed feedback.

---

# Episode structure

An episode consists of $T$ RL steps.

At the beginning of each episode, initialize

$$
\alpha_0 \sim U(0.5,1.5),
$$

and

$$
\delta_0 \sim U(0.1,2.0).
$$

If using $\beta$, initialize

$$
\beta_0 \sim U(0.1,1.0).
$$

At each RL step, run a Monte Carlo block with $M_{\mathrm{block}}$ samples.

Suggested values are

$$
T = 50
$$

to

$$
T = 200,
$$

and

$$
M_{\mathrm{block}} = 500
$$

to

$$
M_{\mathrm{block}} = 5000.
$$

The total Monte Carlo budget per episode is

$$
M_{\mathrm{total}}
=
T M_{\mathrm{block}}.
$$

---

# RL algorithms

You must implement at least one RL algorithm.

You may choose tabular Q-learning, DQN, or an actor-critic method.

---

## Option A: tabular Q-learning

To use tabular Q-learning, discretize the state variables into bins.

For example, bin:

- energy change $\Delta \hat E_t$;
- variance $\hat \sigma_t^2$;
- acceptance rate $A_t^{\mathrm{acc}}$;
- variational parameter $\alpha_t$;
- proposal step $\delta_t$.

The Q-learning update is

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha_{\mathrm{RL}}
\left[
R_{t+1}
+
\gamma
\max_a Q(S_{t+1},a)
-
Q(S_t,A_t)
\right].
$$

Use $\epsilon$-greedy exploration:

$$
A_t =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon, \\
\arg\max_a Q(S_t,a), & \text{with probability } 1-\epsilon.
\end{cases}
$$

A simple exploration schedule is

$$
\epsilon_n =
\max
(
\epsilon_{\min},
\epsilon_0 \lambda^n
),
$$

where $n$ is the episode index.

---

## Option B: deep Q-learning

In DQN, the action-value function is approximated by a neural network:

$$
Q(s,a)
\approx
\hat q(s,a;w).
$$

For a discrete action space with $m$ actions, the network takes $s$ as input and outputs

$$
\hat q(s,a_1;w),
\hat q(s,a_2;w),
\ldots,
\hat q(s,a_m;w).
$$

The DQN target is

$$
Y_t
=
R_{t+1}
+
\gamma
\max_a
\hat q(S_{t+1},a;w^-),
$$

where $w^-$ are target-network parameters.

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

Use:

- replay buffer;
- target network;
- mini-batch updates;
- $\epsilon$-greedy exploration;
- state normalization.

A minimal neural network may have the form:

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)
```

---

## Option C: actor-critic extension

For an actor-critic method, parametrize the policy directly:

$$
\pi_\theta(a \mid s).
$$

The actor chooses actions. The critic estimates values:

$$
\hat v(s;w).
$$

A one-step TD error is

$$
\delta_t
=
R_{t+1}
+
\gamma
\hat v(S_{t+1};w)
-
\hat v(S_t;w).
$$

The actor update is

$$
\theta
\leftarrow
\theta
+
\alpha_\theta
\delta_t
\nabla_\theta
\log
\pi_\theta(A_t \mid S_t).
$$

The critic update is

$$
w
\leftarrow
w
+
\alpha_w
\delta_t
\nabla_w
\hat v(S_t;w).
$$

This option is recommended only after a simpler Q-learning or DQN implementation works.

---

# Baseline methods

The RL method must be compared against non-RL baselines.

Use at least two baselines.

---

## Baseline 1: grid search

Evaluate the energy on a grid of variational parameters.

For the one-parameter case, compute

$$
\hat E(\alpha)
$$

for values

$$
\alpha_1,\alpha_2,\ldots,\alpha_K.
$$

For the two-parameter case, compute

$$
\hat E(\alpha,\beta)
$$

on a two-dimensional grid.

Report the best energy found under the same total Monte Carlo budget.

---

## Baseline 2: fixed proposal step

Choose a fixed proposal step $\delta$ manually.

Run VMC for different values of $\alpha$.

Report:

- energy;
- variance;
- acceptance rate;
- uncertainty over repeated runs.

---

## Baseline 3: acceptance-rate heuristic

Use a simple adaptive rule for $\delta$:

$$
\delta
\leftarrow
\begin{cases}
1.05\delta, & A^{\mathrm{acc}}>0.55, \\
0.95\delta, & A^{\mathrm{acc}}<0.45, \\
\delta, & \text{otherwise}.
\end{cases}
$$

This is not reinforcement learning. It is a hand-designed rule.

Compare whether RL learns something more useful than this heuristic.

---

## Baseline 4: random search

Randomly sample parameter values:

$$
\alpha \sim U(\alpha_{\min},\alpha_{\max}).
$$

If using $\beta$, also sample

$$
\beta \sim U(\beta_{\min},\beta_{\max}).
$$

Evaluate the energy for each sampled parameter set.

Report the best energy found under the same computational budget.

---

# Required tasks

## Task 1: Implement the VMC solver

Implement:

- trial wave function;
- probability density;
- local energy;
- Metropolis proposal;
- Metropolis acceptance rule;
- energy estimator;
- variance estimator;
- acceptance-rate estimator.

Your code should be able to run a VMC calculation for fixed parameters.

---

## Task 2: Validate the VMC solver

Perform at least two validation checks.

Possible checks:

- verify that the Metropolis acceptance probability is correctly implemented;
- test the noninteracting limit;
- compare with known harmonic oscillator results;
- check that energy estimates become more stable as the number of Monte Carlo samples increases;
- check that the estimated uncertainty decreases approximately as $1/\sqrt{M}$.

Include validation plots in the report.

---

## Task 3: Study the variational landscape

Before training any RL agent, study the energy landscape.

For the one-parameter wave function, plot

$$
\hat E(\alpha)
$$

as a function of $\alpha$.

Also plot the variance

$$
\hat \sigma_E^2(\alpha).
$$

If using two parameters, make a contour plot of

$$
\hat E(\alpha,\beta).
$$

Discuss:

- where the energy minimum appears to be;
- how noisy the estimates are;
- whether the landscape is easy or difficult to optimize;
- how the proposal step affects the results.

---

## Task 4: Study the proposal step

For fixed variational parameters, vary $\delta$.

Plot:

- acceptance rate versus $\delta$;
- energy estimate versus $\delta$;
- variance versus $\delta$.

Discuss what happens when $\delta$ is too small or too large.

This task motivates why the proposal step can be treated as a control variable.

---

## Task 5: Define the RL environment

Implement an environment class with the methods

```python
reset()
step(action)
```

The `reset()` method should initialize a new episode.

The `step(action)` method should:

1. apply the selected action;
2. update $\alpha$, $\beta$, or $\delta$;
3. run one Monte Carlo block;
4. compute energy, variance, and acceptance rate;
5. compute reward;
6. return next state, reward, done flag, and diagnostic information.

The output format may be:

```python
next_state, reward, done, info = env.step(action)
```

The `info` dictionary should contain useful diagnostics such as:

```python
info = {
    "energy": energy,
    "variance": variance,
    "acceptance_rate": acceptance_rate,
    "alpha": alpha,
    "beta": beta,
    "delta": delta
}
```

---

## Task 6: Train the RL agent

Train the RL agent over many episodes.

Track:

- return per episode;
- final energy per episode;
- final variance per episode;
- final acceptance rate;
- parameter trajectories;
- action frequencies.

Use several random seeds.

Suggested number of episodes:

$$
N_{\mathrm{episodes}} = 500
$$

to

$$
N_{\mathrm{episodes}} = 5000.
$$

The required number depends on the complexity of the environment and the RL method.

---

## Task 7: Evaluate the trained policy

After training, freeze the policy.

Run evaluation episodes without exploration.

Compare the trained policy against baselines using the same total Monte Carlo budget.

Report:

- mean final energy;
- standard deviation of final energy;
- best final energy;
- mean variance;
- mean acceptance rate;
- number of Monte Carlo samples used.

---

## Task 8: Analyze the learned policy

Inspect what the agent learned.

Questions to answer:

- Does the agent first tune the proposal step?
- Does it keep the acceptance rate near a stable range?
- Does it consistently move $\alpha$ toward the energy minimum?
- Does it stop making large changes late in the episode?
- Does it overreact to Monte Carlo noise?
- Does it learn a strategy that generalizes across different initial parameters?
- Does it outperform simple baselines?

---

# Required figures

The report must include at least the following figures:

1. energy versus variational parameter;
2. local-energy variance versus variational parameter;
3. acceptance rate versus proposal step;
4. training return versus episode;
5. final energy versus episode;
6. final energy distribution over repeated runs;
7. learned parameter trajectory;
8. learned proposal-step trajectory;
9. action-frequency plot;
10. comparison between RL and baselines.

Additional figures are encouraged.

---

# Required tables

Include at least one table comparing methods.

Example:

| Method | Mean final energy | Std. final energy | Best energy | Acceptance rate | Samples used |
|---|---:|---:|---:|---:|---:|
| Grid search | | | | | |
| Random search | | | | | |
| Acceptance heuristic | | | | | |
| Q-learning | | | | | |
| DQN | | | | | |

---

# Suggested code structure

```text
project1_vmc_rl/
│
├── vmc.py
│   ├── wave_function()
│   ├── probability_density()
│   ├── local_energy()
│   ├── metropolis_step()
│   └── run_vmc_block()
│
├── env.py
│   └── VMCOptimizationEnv
│
├── agents.py
│   ├── QLearningAgent
│   ├── DQNAgent
│   └── ActorCriticAgent
│
├── baselines.py
│   ├── grid_search()
│   ├── random_search()
│   └── acceptance_rate_heuristic()
│
├── train.py
│
├── evaluate.py
│
├── plotting.py
│
└── report.md
```

---

# Minimal environment pseudocode

```python
class VMCOptimizationEnv:
    def __init__(self, config):
        self.config = config

    def reset(self):
        self.t = 0
        self.alpha = sample_initial_alpha()
        self.delta = sample_initial_delta()
        self.prev_energy = None

        energy, variance, acc_rate = run_vmc_block(
            alpha=self.alpha,
            delta=self.delta
        )

        self.prev_energy = energy

        return self.make_state(
            energy=energy,
            variance=variance,
            acc_rate=acc_rate
        )

    def step(self, action):
        self.apply_action(action)

        energy, variance, acc_rate = run_vmc_block(
            alpha=self.alpha,
            delta=self.delta
        )

        reward = compute_reward(
            old_energy=self.prev_energy,
            new_energy=energy,
            variance=variance,
            acc_rate=acc_rate
        )

        self.prev_energy = energy
        self.t += 1

        done = self.t >= self.config.episode_length

        state = self.make_state(
            energy=energy,
            variance=variance,
            acc_rate=acc_rate
        )

        info = {
            "energy": energy,
            "variance": variance,
            "acceptance_rate": acc_rate,
            "alpha": self.alpha,
            "delta": self.delta
        }

        return state, reward, done, info
```

---

# Suggested report structure

The final report should be written as a scientific report.

Use the following structure:

## 1. Introduction

Explain the motivation for VMC and for using RL as an adaptive optimizer.

## 2. Theory

Explain:

- the variational principle;
- the Hamiltonian;
- the trial wave function;
- the local energy;
- Metropolis sampling;
- the RL formulation.

## 3. Numerical method

Describe:

- implementation details;
- Monte Carlo block size;
- random seeds;
- parameter ranges;
- baselines;
- RL algorithm.

## 4. Results

Show:

- validation of the VMC solver;
- energy landscape;
- baseline performance;
- RL training curves;
- evaluation results.

## 5. Discussion

Discuss:

- whether RL improved the computation;
- whether the learned policy is interpretable;
- how Monte Carlo noise affected learning;
- whether the result justifies the added complexity of RL;
- limitations of the approach.

## 6. Conclusion

Summarize the main findings.

---

# Reproducibility requirements

Your submission must specify:

- programming language;
- package versions;
- random seeds;
- number of particles;
- oscillator frequency;
- trial wave function;
- number of Monte Carlo samples;
- number of RL episodes;
- RL algorithm;
- neural-network architecture, if used;
- optimizer and learning rate;
- discount factor;
- exploration schedule;
- reward weights;
- parameter bounds;
- computational budget.

---

# Minimum requirements

To pass the project, your submission must include:

- working VMC solver;
- implementation of Metropolis sampling;
- validation of the VMC code;
- at least one variational parameter;
- one RL agent;
- at least two non-RL baselines;
- repeated runs over at least five random seeds;
- at least seven figures;
- a fair comparison using equal Monte Carlo budget;
- a written discussion of the RL formulation.

---

# Extensions

Possible extensions include:

- two-parameter Jastrow wave function;
- importance sampling;
- adaptive block sizes;
- actor-critic with continuous actions;
- PPO for continuous parameter updates;
- neural-network quantum states;
- transfer learning across different oscillator frequencies;
- comparison with gradient descent;
- uncertainty-aware rewards;
- parallel Monte Carlo chains.

---

# Discussion questions

Your report should address the following questions.

## Physics questions

- What physical system is being studied?
- What is the Hamiltonian?
- What is the trial wave function?
- What is the local energy?
- How is the Monte Carlo sampling performed?
- How was the numerical implementation validated?

## RL questions

- What is the state?
- What is the action?
- What is the reward?
- What is the transition?
- What is an episode?
- Is the state approximately Markovian?
- Is the reward dense or sparse?
- Is the method on-policy or off-policy?
- Is the method model-free or model-based?

## Scientific-computing questions

- Does RL outperform simple baselines?
- Is the comparison fair?
- How sensitive are the results to hyperparameters?
- Does RL exploit Monte Carlo noise?
- Does the learned policy make physical sense?
- Is RL necessary for this problem?
- What would make the problem harder or more realistic?

---

# Grading guide

| Component | Points |
|---|---:|
| VMC theory and implementation | 25 |
| Monte Carlo sampling and validation | 15 |
| RL formulation | 20 |
| RL implementation | 15 |
| Baseline comparison | 10 |
| Results and figures | 10 |
| Discussion and interpretation | 5 |
| Total | 100 |

---

# What makes a strong project?

A strong project does not merely show that an RL algorithm runs.

A strong project shows:

- that the VMC solver is correct;
- that the RL environment is well defined;
- that the reward function is meaningful;
- that the baselines are fair;
- that the learned behavior is interpretable;
- that the comparison uses equal computational budget;
- that failures and limitations are discussed honestly.

The central question is not only whether RL can lower the energy.

The central question is whether RL learns a useful and transferable strategy for controlling a noisy quantum Monte Carlo optimization.