# Budgeted Opinion Dynamics with Varying Susceptibility to Persuasion
Joint work with [Dr. Hubert Chan](https://i.cs.hku.hk/~hubert/)

## Overview
This study focuses on an opinion formation model where each agent $`i \in \{1,\cdots,N\}`$ has an innate opinion $`s_i`$ represented as a value in the interval $`[0, 1]`$. Agents update their expressed opinions iteratively (in discrete time steps) based on a weighted average of their peers' opinions (captured by a row stochastic matrix $`P \in [0,1]^{N\times N}`$) and their own innate position, with the influence of peer opinions modulated by a resistance parameter $`\alpha_i`$. This parameter determines how susceptible an agent is to persuasion; higher values indicate lower susceptibility or higher resistance. The goal is to minimize the average equilibrium opinion (when $`t \to \infty`$) by strategically adjusting these resistance parameters $\alpha$ within given constraints. We call this the *Opinion Susceptibility Problem*.

Three key variants of this optimization problem are considered: the unbudgeted variant, where the resistance parameters can be changed without any restriction; the $`L_0`$-budgeted variant, where changes are restricted to a fixed number of agents, and the proposed $`L_1`$-budgeted variant, which allows for fractional adjustments, thereby reflecting a more realistic approach to the effort involved in modifying resistance levels.

Before our study, it is known that the unbudgeted variant is polynomial-time solvable via a local search algorithm [(Chan et al., 2019)](https://scholar.google.com/scholar_url?url=https://dl.acm.org/doi/abs/10.1145/3308558.3313509&hl=zh-TW&sa=T&oi=gsr-r&ct=res&cd=0&d=2759380754538164998&ei=nVBnaKalLZPN6rQPhMr3yAg&scisig=AAZF9b9H-stMyC3lePN8dh7B54sI). On the other hand, the $`L_0`$-budgeted variant is NP-hard [(Abebe et al., 2018)](https://scholar.google.com/scholar_url?url=https://dl.acm.org/doi/abs/10.1145/3219819.3219983&hl=zh-TW&sa=T&oi=gsr-r&ct=res&cd=0&d=13274367214912563122&ei=qlBnaJz6Fc2l6rQPh4i_0AE&scisig=AAZF9b8lUoPnuwsXZHR4iphtCRly).

## Results
- Algorithmic Result: A projected gradient algorithm for computing the optimal solution to the $`L_1`$-budgeted variant. Its implementation is given in this repo.
- Hardness Result: The $`L_1`$-budgeted variant of the opinion dynamics optimization problem is NP-hard. An implication of this result is that the optimal solution can be achieved by focusing the given $`L1`$-budget on as small a number of agents as possible. For details, please refer to our [paper](https://arxiv.org/abs/2105.04105) in COCOON 2021.
