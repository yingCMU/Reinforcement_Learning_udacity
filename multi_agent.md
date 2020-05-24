## Markov games as a framework for multi-agent reinforcement learning - [paper](https://www2.cs.duke.edu/courses/spring07/cps296.3/littman94markov.pdf)

In the Markov decision process(MDP) formalization of reinforcement learning, a single adaptive
agent interacts with an environment defined by a
probabilistic transition function. In this solipsistic view, secondary agents can only be part of the
environment and are therefore fixed in their behavior. The framework of Markov games allows
us to widen this view to include multiple adaptive agents with interacting or competing goal.

When do Markov Games reduce to an MDP? when a single agent is present in the environment.


## how to adapt single-agent RL to muti-agent case?
1. train agents independently without considering the existence of other agents. any agent consider others as part of environemnt. Since all are learning simutaneously, from the perspective of a single agent,  the environment change dynamically(non-stationarity). In most single agent algorithms, it is assumed the environment is stationary so it has certain convergence guarantees, which no longer hold.
2. meta-agent approach takes into account the existence of multiple agents, a single policy is learned for all agents, it takes as input the present state of the environment and returns the action of each agent in the form of a single join action vector. joint action space increase exponentiallly with the number of agents, if the environment is partially observable or the agent can only see locally, each agent will have a different observation of the environment state, hence it will be difficult to disambiguate state from local observations. So this approach only works when every agent knows everything about the environment

## Multi-agent version of DDPG
paper - [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)

## alphago
performance in alphago depends on expert input during the training step, and so the algorithm cannot be easily be transferred to other domains

## alphago zero
 instead of depending on expert gameplay for the training, alphago zero learned from playing against itself, only knowing the rules of the game.

[Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)

## alphazero'
 - an entirely new framework for developing AI engine,
- The best part of the alphazero algorithm is simplicity: it consists of a Monte Carlo tree search, guided by a deep neural network. This is analogous to the way humans think about board games -- where professional players employ hard calculations guides with intuitions.

[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
