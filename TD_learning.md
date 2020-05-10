
# TD Learning
MC update the estimate at every end of episode.
TD use an alternative target to update estimate at every time step.
## 1. Sarsa
![alt text](./images/td_sarsa.png)
evaluate epsilon-greedy policy

## 2. SarsaMax(Q-Learning)
S0, A0, R1, S1 ||
update the value estimate using greedy policy
S0, A0, R1, S1 || A1
Then choose the A1 still using epsilon-greedy policy with the action values just updated
![alt text](./images/q_learning.png)

evaluate greedy policy

[research paper-Technical Note Q,-Learning](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf) to read the proof that Q-Learning (or Sarsamax) converges.

sarsa evaluate whatever epsion-greedy policy that is currently being followed by the agent.

sarsamax directly attempts to approximate the optimal value function at every time step.

![alt text](./images/q_learning_code.png)
## 3. Expected Sarsa
read paper [A Theoretical and Empirical Analysis of Expected Sarsa](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.216.4144&rep=rep1&type=pdf) to learn more.
![alt text](./images/expected_sarsa.png)

## Compare 3 sarsa together
![alt text](./images/sarsa_compare.png)
# TD in practice
Greedy in the Limit with Infinite Exploration (GLIE)
The Greedy in the Limit with Infinite Exploration (GLIE) conditions were introduced in the previous lesson, when we learned about MC control. There are many ways to satisfy the GLIE conditions, all of which involve gradually decaying the value of \epsilonϵ when constructing \epsilonϵ-greedy policies.

In particular, let \epsilon_iϵ
i
​	  correspond to the ii-th time step. Then, to satisfy the GLIE conditions, we need only set \epsilon_iϵ
i
​	  such that:

\epsilon_i > 0ϵ
i
​	 >0 for all time steps ii, and
\epsilon_iϵ
i
​	  decays to zero in the limit as the time step ii approaches infinity (that is, \lim_{i\to\infty} \epsilon_i = 0lim
i→∞
​	 ϵ
i
​	 =0),
- In Theory
All of the TD control algorithms we have examined (Sarsa, Sarsamax, Expected Sarsa) are guaranteed to converge to the optimal action-value function q_*, as long as the step-size parameter \alphaα is sufficiently small, and the GLIE conditions are met.
- In practice, it is common to completely ignore the GLIE conditions and still recover an optimal policy. (You will see an example of this in the solution notebook.)
## Analyzing Performance
### Similarities
All of the TD control methods we have examined (Sarsa, Sarsamax, Expected Sarsa) converge to the optimal action-value function q_*q
∗
​	  (and so yield the optimal policy \pi_*π
∗
​	 ) if:

the value of \epsilonϵ decays in accordance with the GLIE conditions, and
the step-size parameter \alphaα is sufficiently small.
### Differences
The differences between these algorithms are summarized below:

- Sarsa and Expected Sarsa are both on-policy TD control algorithms. In this case, the same (\epsilonϵ-greedy) policy that is evaluated and improved is also used to select actions.
- Sarsamax is an off-policy method, where the (greedy) policy that is evaluated and improved is different from the (\epsilonϵ-greedy) policy that is used to select actions.
- On-policy TD control methods (like Expected Sarsa and Sarsa) have better online performance than off-policy TD control methods (like Sarsamax).
- Expected Sarsa generally achieves better performance than Sarsa.
If you would like to learn more, you are encouraged to read Chapter 6 of the textbook (especially sections 6.4-6.6).

![alt text](./images/td_txt_compare.png)

The figure shows the performance of Sarsa and Q-learning on the cliff walking environment for constant \epsilon = 0.1ϵ=0.1. As described in the textbook, in this case,

Q-learning achieves worse online performance (where the agent collects less reward on average in each episode), but learns the optimal policy, and
Sarsa achieves better online performance, but learns a sub-optimal "safe" policy.

- summary
    - On-policy TD control methods (like Expected Sarsa and Sarsa) have better online performance than off-policy TD control methods (like Q-learning).
    - Expected Sarsa generally achieves better performance than Sarsa.

## Optimism
[Convergence of Optimistic and
Incremental Q-Learning](http://papers.nips.cc/paper/1944-convergence-of-optimistic-and-incremental-q-learning.pdf)

You have learned that for any TD control method, you must begin by initializing the values in the Q-table. It has been shown that initializing the estimates to large values can improve performance. For instance, if all of the possible rewards that can be received by the agent are negative, then initializing every estimate in the Q-table to zeros is a good technique. In this case, we refer to the initialized Q-table as optimistic, since the action-value estimates are guaranteed to be larger than the true action values.
