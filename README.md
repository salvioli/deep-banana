# deep-monkey
A deep reinforcement learning agent that catches yellow bananas and avoids blue ones in a large square world simulated
with Unity ML-Agents.

This project is my solution to the navigation project of udacity's 
[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project Details
The agend is a monkey moving in a 2d arena where on the floor are blue and yellow bananas, each time the agent hits a 
banana it is rewarded as follows:
- for the yellow bananas it receives a reward of +1 
- for the blue bananas it receives a reward of -1

State space has 37 continuous dimensions including:
- the agent's velocity
- ray based perception of objects around agent's forward direction

Action space has 1 discrete dimension, possible dimensions are:
- **`0`** - move forward
- **`1`** - move backward
- **`2`** - turn left
- **`3`** - turn right

The task is episodic.

The task is considered solved if the agent can achieve a score of +13 over 100 consecutive episodes.

## Getting Started
1. Download the pre-compiled unity environment
Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
1. Decompress the archive at your preferred location (e.g. in this repository working copy)
1. Open the getting-started.ipynb. This notebook installs the dependencies and explores the environment concluding
with a demonstration of an agent which chooses actions randomly.
1. Follow the instructions indicated in the getting-started.ipynb notebook. You will need to specify the path to the 
environment executable that you downloaded at the beginning.

### Code organization
The code is organized as follows (in hierarchical order from abstract to detailed):
- Report.ipynb: notebook illustrating the result of this project.
- deep_monkey.py: includes high level functions used in the notebook, these are used for training, plotting results and 
saving model checkpoints.
- agent.py: this file include the classes that model the agent and its dependencies. Implements a high level Agent which
generalizes each variant of the original DQN algorithm.
- model.py: this file include the neural network class, implemented using pytorch, which is used by the agent for
approximating the q function.

## Instructions

### Prerequisites
A working python 3 environment is required. You can easily setup one installing [anaconda] (https://www.anaconda.com/download/)

### Installation
If you are using anaconda is suggested to create a new environment as follows:
```bash
conda create --name deepmonkey python=3.6
```
activate the environment
```bash
source activate deepmonkey
```
start the jupyter server
```bash
python jupyter-notebook --no-browser --ip 127.0.0.1 --port 8888 --port-retries=0
```

### Future development
Agent's learning performances (number of episodes to solve the task) should be improved implementing the following:
- Prioritized experience replay
- Dueling DQN

In addition, the same task should be solved starting from raw pixel observations.
