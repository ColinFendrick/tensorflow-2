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

## Q-Learning

Instead of looking at values of each state the agent could end
up in, we look at the value of the action. Similar to equation
above, we have the reward for action plus discounted
expected value of the next state (due to randomness in action).
Thus the Q value is the value inside the brackets above - the
thing the agent must select the maximum of above.
$$
Q(s, a) = R(s, a) + γ\sum_{s'}(P(s,v a, s')V(s'))
$$
Substituting, since this is a recursive function of V:
$$
Q(s, a) = R(s, a) + γ\sum_{s'}(P(s,v a, s')max_{a'}Q(s', a'))
$$
In this way we can express this as a check of agent actions.

## Temporal Difference

Using the deterministic Bellman equation for simplicity sake:
$$
Q(s, a) = R(s, a) + γmax_{a'}Q(s', a')
$$
Then the temporal difference - the after value minus the
previous value.
$$
TD(s, a) = R(s, a) + γmax_{a'}Q(s', a') - Q(s, a)
$$
If the model is perfect, this would be 0. However, the Q(s, a)
value is the previous value, the new value is what is calculated
after the agent takes the action. This could be due to
randomness. So we change our Q values slightly over time
due to the temporal difference:
$$
Q_t(s, a) = Q_{t-1}(s, a) + \alpha TD_t(s, a)
$$
Then we can substitute this temporal difference out with the
corresponding Q-value to get a recursive Q formula:
$$
Q_t(s, a) = Q_{t-1}(s, a) + \alpha\big(
  R(s, a) + γmax_{a'}Q(s', a') - Q_{t-1}(s, a)
\big)
$$
Seen above, the alpha value gives a weight to how much the
previous state has, since we have
$$
Q_{t-1}(s,a)-\alpha Q{t-1}(s,a)
$$
inside of the equation. This is to account for randomness.

## Deep Q-Learning

We feed information of each state into a neural network that
gives us the Q values. Rather than comparing against previous
Q-values when determining temporal difference, neural network
makes predictions of the Q-values for agent to compare against.
The neural network can predict multiple Q-values. Then each
loss is calculated for the predicted q-values and
back-propogated. Then the action is passed through a soft-max
function to select the best action possible.

## Experience Replay

Lots of consecutive states can bias a network. E.g. a
self-driving car learning to drive on a straight road biases a
a network against turning.

Experience replay will randomly select a uniformly-distributed
sample of batched experiences and learns from that. This way the
network does not overfit to consecutive experiences. This can
be used to give more weight to rare experiences. Also helps to
learn faster, especially in smaller data.

## Action Selection Policy

We don't just select the highest Q value, but rather use a
function (softmax, epsilon-greedy, etc) to force the agent to
explore. This is to prevent bias towards local maxima. E.g.
going up is a good action to take but the agent has never tried
going down.

- epsilon-greedy means take the best action except epsilon percent of the time
- epsilon-soft means take the best action except 1-epsilon percent of the time (inverted of epsilon-greedy)
- Softmax takes however many outputs you have, squashes them to between [0, 1] that add up to 1.
