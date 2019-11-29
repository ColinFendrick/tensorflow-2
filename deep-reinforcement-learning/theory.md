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

## Markov Decision Process (MDP)

### Difference btwn Deterministic and Non-Deterministic

Deterministic search - if agent is in maze,
agent goes up, he goes up

Non-deterministic - 80% when agent wants to go up, he goes up
but 10% chance he goes left, 10% chance he goes right

### Markov Property

Stochastic process has _Markov property_ if the conditional
probabibility of future states depends only upon the present
state, not the events that preceeded it. This is called a
_Markov process_. A maze is an example of this. All that matters
is current location, not how the agent got there.

### Markov Decision Processes

_Markov Decision Processes_ provide a mathematical model in
situations where outcomes are partly random and partly under
the control of the decision maker. An example of this is the
maze in a non-deterministic search. If the agent moves left,
there is only a 80% chance the agent will actually move left.
This is an addon to the Bellman equation, by changing the value
of s-prime with the expected value of s-prime, since there is
randomness in s-prime's value - for example, 80% the agent moves
up, etc.
$$
V(s) = max_a\bigg(R(s, a) + γ\sum_{s'}P(s, a, s')V(s')\bigg)
$$
