# Deep Reinforcement Learning Theory

Reinforcement learning is like training a dog. State-reward learning.

## Bellman Equation

Recursion for expected rewards. Will assign a different value
and action for each possible state and maximize that.

s = State, a = Action, R = Reward, γ - Discount
$$
V(s) = max_a(R(s, a) + γV(s'))
$$
Maximize the reward for current state and action, plus
the discount on the value of a new state.
The result of this, due to the discount, is that this
gives a value to each state.
