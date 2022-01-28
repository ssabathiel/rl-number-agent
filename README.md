
# RL-agents communicating numbers using external tools

Code used for the experiments in 'Self-Communicating Deep Reinforcement Learning AgentsDevelop External Number Representations' (in the revision process).

**Purpose**  
Providing a computational framework that allows the investigation of how material representations might support number processing in a deep reinforcementlearning scenario.

### Environment 

The agent consists of 3 parallel input layers, a 3-layer feed forward neural network, and an output vector that is divided into a verbal output and an
action output. The action output feeds back to the environment and the input layers.

The three perceptual input layers are binary grids with the same dimensions. In this paper we simulate agents with either 1D or 2D perceptual input layers and periodic boundary conditions. There is a numerosity input layer, a tool input layer and a finger input layer.

The numerosity layer represents the numerosity that needs to be encoded and communicated in order to solve the task presented to the agent. There are two different types of visual input:  
- Spatially distributed white cells  
- Temporally distributed rectangles appearing on the screen for 1 time step.  

The tool layer is the neural network’s visual input from an external tool that the agent can manipulate and use to communicate numerical information.
One can optionally choose between two tools:  
- unconstrained drawing tool that allows the agent to flip the binary values of a grid cell by outputting the corresponding coordinates  
- an abacus-like tool that allows to move tokens to the left or right  


The framework is implemented as a gym-like environment:  
env.obs(), env.step(action), env.reset(), env.reward()  

### Learning task
In every experimental setting the goal of the agent is to produce the number word associated with the number of items presented in the numerosity layer.
In each episode, the environment is initialized with a random visual scene and a default state for the finger and tool layers.
The agent is then allowed to interact with the environment during subsequent time steps. Each episode ends after 3 time steps for the static setup, and after 3 time steps from the last event for the temporal setup.
At the final time step of the episode, all layers except for the tool layer are grayed out (uniformly set to values of 0.5).
This way, the final answer only depends on the current state of the tool, since the agent does not have an internal (e.g., recurrent) memory.
The task is said to be successfully performed when the agent outputs the correct number word at the last time step.


**RL learning algorithm**  
PPO with Clipped Surrogate Objective.  
Uses curriculum learning:  starts  from small numbers and progressively moves to larger numbers.  
Adapted version from: https://github.com/nikhilbarhate99/PPO-PyTorch

**Hyperparameters used for the experiments**  

```eval_every_n_iterations = 100 # evaluate the model every n iterations  
eval_n_episodes_per_itr = 100 # evaluate n episodes per evaluation-iteration  
collect_n_episodes_per_itr = 64 # collect n episodes per iteration  
...: 	10 # Number of actors collecting data 'parallely'  
K_epochs = 40  # training on K_epochs epochs at each policy update
eps_clip = 0.2 # clip parameter for PPO  
gamma = 0.99 # discount factor # Discount Factor  
c1 = 0.5 # value Function Coefficient c1
c2 = 0.05 # entropy Coefficient  
  
lr_actor = 0.0003 # learning rate for actor network  
lr_critic = 0.001 # learning rate for critic network  
```
