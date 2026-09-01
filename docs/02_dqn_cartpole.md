# Project 2: Deep Q-Network (DQN / Double DQN) on CartPole

Scaling value-based RL from a lookup table to a neural function approximator. A
complete from-scratch DQN (replay buffer, target network, Double DQN target,
gradient clipping and Huber loss) that solves CartPole on CPU.

## The environment

`CartPoleEnv` ([`rl/envs/cartpole.py`](../rl/envs/cartpole.py)) re-implements the
classic cart-pole control task: balance a pole hinged on a cart by pushing the
cart left or right. The physics use the same equations of motion and
semi-implicit Euler integration as the reference Gymnasium environment, with
identical termination thresholds (+/-12 degrees pole angle, +/-2.4 cart
position).

- State `(4,)`: cart position, cart velocity, pole angle, pole angular velocity.
- Actions (2): push left, push right.
- Reward: +1 per timestep the pole stays upright.
- Episode: terminates on failure, truncates at 500 steps. Max return is 500.

Because the dynamics match the canonical task, a policy trained here transfers
directly to `gymnasium.make("CartPole-v1")`.

## The agent

[`rl/deep/dqn.py`](../rl/deep/dqn.py) implements `DQNAgent`, combining the three
ingredients that make deep value-based RL stable:

1. Q-network ([`networks.py`](../rl/deep/networks.py)): an MLP (`128 -> 128 -> 2`)
   that outputs the full action-value vector `Q(s, .)` in one forward pass.
2. Experience replay ([`replay_buffer.py`](../rl/deep/replay_buffer.py)): a
   fixed-capacity ring buffer (NumPy-backed) that decorrelates the training
   minibatches.
3. Target network: a periodically-frozen copy of the Q-network used to compute
   bootstrap targets, which removes the moving-target instability.

Other engineering details: epsilon is annealed linearly over the first 10k
environment steps, the loss is Huber/SmoothL1 (robust to outliers), and
gradients are clipped by norm.

### DQN vs Double DQN

The TD target differs by which network evaluates the next action:

```
DQN         : y = r + gamma * max_a'  Q_target(s', a')
Double DQN  : y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
```

Double DQN decouples action selection (online net) from action evaluation
(target net), which reduces the systematic overestimation of Q-values that
vanilla DQN suffers from. It is the default here; switch it off with
`--no-double-dqn`.

All hyper-parameters live in the `DQNConfig` dataclass, so an experiment is fully
described by a single object.

## How to run

```bash
# Train (Double DQN by default)
python scripts/train_dqn.py --episodes 700 --seed 0

# Vanilla DQN for comparison
python scripts/train_dqn.py --episodes 700 --seed 0 --no-double-dqn

# Evaluate the saved greedy policy
python scripts/evaluate_dqn.py --episodes 20
```

Training writes the model to `models/dqn_cartpole.pt` and the learning curve to
`assets/dqn_cartpole_returns.png`.

## Results

![Double DQN on CartPole](../assets/dqn_cartpole_returns.png)

With `seed=0`, Double DQN:

- reaches the maximum episodic return of 500,
- crosses the standard "solved" criterion (a 100-episode moving average of at
  least 475) at episode 695,
- and saves a checkpoint, taken at episode 400, that returns 500 in every one of
  200 greedy episodes.

Those last two are separate facts about separate episodes. Crossing the solved
threshold at 695 says the previous hundred episodes averaged well; it does not
say the weights at episode 695 are good, and here they are not. Evaluated
greedily they score 141.6/500, against 500/500 for the checkpoint kept from
episode 400. The next two sections are about why.

### Reading the plot

The two blue lines are different series. The pale one is the raw return of each
individual episode; the solid one is its 25-episode moving average. The raw line
is the one that swings between 0 and 500 from roughly episode 380 onwards, and
that swing is worth explaining because it looks alarming.

Exploration noise is part of it, but only a small part. Measuring the saved
policy over 200 episodes puts a number on it: at epsilon 0 it returns 500 every
single time, and at the 0.02 floor used during training it averages 475.3 and
reaches 500 in 186 of those 200 episodes. At that rate the agent takes about ten
random actions in a 500-step episode, and roughly one episode in fourteen ends
early because of them. That is the baseline rate of downward spikes in the pale
line, and it is not close to what the curve shows.

Epsilon also settles at its floor around episode 225 and is constant from then
on, so it cannot produce a moving average that keeps changing. A stable policy
of the quality above would hold its 100-episode average near 475 for the rest of
the run. Instead the average dips to roughly 180 around episodes 460 to 490.

What the rest of the swing shows is the policy itself changing. The network
takes a gradient step after every environment step, so consecutive episodes are
run by slightly different policies, and for a long stretch the agent keeps
crossing back and forth between one that balances the full 500 steps and one
that drops the pole early. That dip is a real regression rather than a sampling
artefact: it is the familiar instability of bootstrapped targets combined with a
function approximator and a replay buffer whose contents shift as the policy
improves.

So the shape is the expected one for DQN on this task: a slow start while the
replay buffer fills, rapid improvement, a genuinely unstable middle, and a
stable policy at the end.

### Choosing which policy to keep

If the policy keeps changing, then the policy sitting in memory when the loop
ends is not necessarily a good one. Saving that one would make the quality of
the checkpoint depend on where training happens to stop. So `train_dqn.py`
evaluates the agent greedily every `--eval-interval` episodes and writes the
best one it has seen to `models/dqn_cartpole.pt`.

The criterion has to be a direct evaluation of the current weights. The obvious
cheaper alternative, picking the episode with the best 100-episode moving
average, does not work, and this task shows why clearly. At seed 0 the moving
average peaks at episode 695, but the weights from episode 695 score 141.6/500
under greedy evaluation, while the ones from episode 700 score 500/500. The
moving average lags by a full window and summarises 100 episodes played by 100
different policies, so a high value says very little about the policy actually
in hand at that moment.

Evaluating directly finds a policy that scores 500/500 at episode 400, three
hundred episodes before training stops. Selection runs on its own set of seeds,
separate from the ones used for the final reported figure, so the headline
number is not measured on the same episodes it was chosen on. The evaluation
printed at the end reloads the checkpoint from disk, so the number reported
belongs to the model that was actually saved.

Because greedy action selection never draws from the agent's random number
generator and no gradient steps are taken during evaluation, this leaves the
training trajectory untouched: the learning curve above regenerates byte for
byte with and without it.

## What this covers

- Building DQN end-to-end in PyTorch, including the subtle parts (target masking
  on terminal transitions, `gather` for taken actions, target-network sync).
- Why replay and target networks are necessary, not only that they are.
- The overestimation problem and the Double DQN fix.
- Practical training hygiene: reproducible seeds, checkpointing, evaluation kept
  separate from training, configurable hyper-parameters.
