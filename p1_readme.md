- rubric: https://review.udacity.com/#!/rubrics/1889/view
- note that you should be able to solve the project by making only minor modifications to the DQN code provided as part of the Deep Q-Networks lesson
- we were able to solve the project in fewer than 1800 episodes.
- Not sure where to start? https://fburl.com/0kjwh0jn

## bug
- forgot self.optimizer.zero_grad(), as a result, loss with more episode increase, why?
- changed network, before I was using 512-512 with drop out, it couldn't finish training with 200 reward within 2000 epochs. Now I changed it to be 64-64 with no drop-out, it is making good progress
    - but still not finished training within 2000
    - after 600 episode, progreess is very slow
        ```
        Episode 100	Average Score: -164.23
        Episode 200	Average Score: -116.71
        Episode 300	Average Score: -76.413
        Episode 400	Average Score: 30.193
        Episode 500	Average Score: 129.76
        Episode 600	Average Score: 151.75
        Episode 700	Average Score: 134.87
        Episode 800	Average Score: 134.61
        Episode 900	Average Score: 159.92
        Episode 1000	Average Score: 163.96
        Episode 1100	Average Score: 182.25
        Episode 1145	Average Score: 176.38
        ```
    - trained another round, Environment solved in 1409 episodes!	Average Score: 200.49
- tried another loss API: `loss = F.mse_loss(Q_expected, Q_targets)`

- why solution code trains so fast?
    ```
    Episode 100	Average Score: -208.91
    Episode 200	Average Score: -139.36
    Episode 300	Average Score: -78.241
    Episode 400	Average Score: -45.36
    Episode 500	Average Score: -46.01
    Episode 600	Average Score: 11.482
    Episode 700	Average Score: 125.49
    Episode 800	Average Score: 169.92
    Episode 900	Average Score: 182.33
    Episode 1000	Average Score: 187.33
    Episode 1090	Average Score: 200.02
    Environment solved in 990 episodes!	Average Score: 200.02

    ```
    - I use exactly agent code from solution, here is how it performs:
       ```
        Episode 100	Average Score: -229.15
        Episode 200	Average Score: -155.74
        Episode 300	Average Score: -122.14
        Episode 400	Average Score: -71.126
        Episode 500	Average Score: -79.03
        Episode 600	Average Score: -80.15
        Episode 700	Average Score: -8.582
        Episode 800	Average Score: 21.56
        Episode 900	Average Score: 144.20
        Episode 1000	Average Score: 48.05
        Episode 1100	Average Score: 143.33
        Episode 1200	Average Score: 138.89
        Episode 1300	Average Score: 154.17
        Episode 1400	Average Score: 160.33
        Episode 1500	Average Score: 184.27
        Episode 1600	Average Score: 194.99
        Episode 1670	Average Score: 200.00
        ```


# Navigation

Episode 100	Average Score: 0.80
Episode 200	Average Score: 3.94
Episode 300	Average Score: 7.76
Episode 400	Average Score: 9.52
Episode 469	Average Score: 12.00
Environment solved in 369 episodes!	Average Score: 12.00
