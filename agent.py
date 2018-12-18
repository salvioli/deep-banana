import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, seed,
                 fc1_size=64, fc2_size=64,
                 checkpoint_filename=''):
        """
        Initializes an agent object
        TODO make the structure of the qfunction approximator more flexible
        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param seed: random seed
        :param fc1_size: number of units of the first fully connected layer of the q function approximator
        :param fc2_size: number of units of the second fully connected layer of the q function approximator
        :param checkpoint_filename: name of the checkpoint file which contains the load_state_dict pickled
                                    weights of the q function approximator.
        :return agent: initialized agent
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.BUFFER_SIZE = int(1e5)  # replay buffer size
        self.BATCH_SIZE = 64  # minibatch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 1e-3  # for soft update of target parameters
        self.LR = 5e-4  # learning rate
        self.UPDATE_EVERY = 4  # how often to update the network

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_size, fc2_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_size, fc2_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)
        self.criterion = torch.nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        if checkpoint_filename != '':
            self.qnetwork_local.load_state_dict(torch.load(checkpoint_filename))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: (Tuple[torch.Variable]) tuple of (s, a, r, s', done) tuples
        :param gamma: (float) discount factor
        """

        y = self._q_target(experiences, gamma)
        y_pred = self._q_estimated(experiences, gamma)

        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        :param tau: interpolation parameter
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def _q_target(self, experiences, gamma):
        """Method that calculates the target q value used for training"""
        raise NotImplementedError

    def _q_estimated(self, experiences, gamma):
        """Method that calculates the estimated q value used for training"""
        states, actions, rewards, next_states, dones = experiences
        # feedforward the local network
        return self.qnetwork_local(states).gather(1, actions)

class DQN(Agent):
    """Agent which implement the vanilla DQN algorithm"""
    def _q_target(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        # Propagate the target network and detach as we don't need gradient
        q_next_target = self.qnetwork_target(next_states).detach()
        # take maximum q values and add one single dimension to go from a [n] to a [n, 1] tensor
        q_next_target = q_next_target.max(1)[0].unsqueeze(1)
        # set the target
        return rewards + gamma * q_next_target * (1 - dones)

class DoubleDQN(Agent):
    """Agent which implement the double DQN algorithm"""
    def _q_target(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        # get actions that maximize the local network
        argmax_local_action = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
        # evaluate the actions on the target network
        q_next_target = self.qnetwork_target(next_states).detach().gather(1, argmax_local_action)
        # set the target
        return rewards + gamma * q_next_target * (1 - dones)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.

        :param action_size: dimension of each action
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param seed: random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
