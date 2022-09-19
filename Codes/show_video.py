import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io

import numpy as np
from collections import deque, namedtuple

# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import glob
# enviroment
import gym
env = gym.make('LunarLander-v2')
state_size = 8  #TODO: find observation size
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ExperienceReplay():
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def store_trans(self, s, a, sp, r, done):
        # TODO: store new transition in memory
        exp = Transition(s, a, r, sp, done)
        self.memory.append(exp)
        pass

    def sample(self, batch_size):
        # TODO: take RANDOM sample from memory
        sample_exp = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack([exp.state for exp in sample_exp if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in sample_exp if exp is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in sample_exp if exp is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in sample_exp if exp is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([exp.done for exp in sample_exp if exp is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# DQN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size) -> None:
        super(DeepQNetwork, self).__init__()
        # TODO: define the architecture
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        # NOTE: input=observation/state, output=action
        pass

    def forward(self, x):
        # TODO: forward propagation
        """Build a network that maps state -> action values."""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
        # NOTE: use ReLu for activation function in all layers
        # NOTE: last layer has no activation function (predict action)
        pass


"""## DQN"""

# DQN agent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNAgent():
    # NOTE: DON'T change initial values
    def __init__(self, state_size, action_size, batch_size,
                 gamma=0.99, buffer_size=25000, alpha=5e-4):
        # network parameter
        self.state_size = state_size
        self.action_size = action_size

        # hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma

        # experience replay
        self.experience_replay = ExperienceReplay(buffer_size)

        # network
        self.value_net = DeepQNetwork(state_size, action_size).to(device)

        # optimizer
        # TODO: create adam for optimizing network's parameter (learning rate=alpha)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=alpha)

    def take_action(self, state, eps=0.0):
        # TODO: take action using e-greedy policy
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.value_net.eval()
        with torch.no_grad():
            action_values = self.value_net(state)
        self.value_net.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        # NOTE: takes action using the greedy policy with a probability of 1−𝜖 and a random action with a probability of 𝜖
        # NOTE:

    def update_params(self):
        if len(self.experience_replay) < self.batch_size:
            return
        # transition batch
        # batch = Transition(*zip(*self.experience_replay.sample(self.batch_size))) #gives error for my code
        # batch = random.sample(self.experience_replay.memory, k=self.batch_size)
        bbatch = agent.experience_replay.sample(self.batch_size)
        batch = Transition(bbatch[0], bbatch[1], bbatch[3], bbatch[2], bbatch[4])
        """
        state_batch = torch.from_numpy(np.vstack([e.state for e in batch if e is not None])).float().to(device)
        action_batch = torch.tensor(np.vstack([e.action for e in batch if e is not None])).long().to(device)
        next_state_batch = torch.from_numpy(np.vstack([e.next_state for e in batch if e is not None])).float().to(device)
        reward_batch = torch.tensor(np.vstack([e.reward for e in batch if e is not None])).float().to(device)
        done_batch = torch.tensor(np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float().to(device)
        """
        state_batch = (batch.state.cpu()).float().to(device)
        action_batch = (batch.action.cpu()).long().to(device)
        next_state_batch = (batch.next_state.cpu()).float().to(device)
        reward_batch = (batch.reward.cpu()).float().to(device)
        done_batch = (batch.done.cpu()).to(device)

        # calculate loss w.r.t DQN algorithm

        # STEP1
        # TODO: compute the expected Q values [y]
        q_expected = self.value_net(state_batch).gather(1, action_batch)

        # STEP2
        # TODO: compute Q values [Q(s_t, a)]
        Q_max = self.value_net(next_state_batch).detach().max(1)[0].unsqueeze(1)
        q_table = reward_batch + self.gamma * Q_max * (1 - done_batch)
        # STEP3
        # TODO: compute mse loss
        loss = F.mse_loss(q_expected, q_table)

        # TODO: optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # NOTE: DON'T forget to set the gradients to zeros

    def save(self, fname):
        # TODO: save checkpoint
        torch.save(agent.value_net.state_dict(), fname + ".pth")

    def load(self, fname, device):
        # TODO: load checkpoint
        agent.value_net.load_state_dict(torch.load(fname + ".pth"))


# NOTE: DON'T change values
n_episodes = 250
eps = 1.0
eps_decay_rate = 0.99

#scores = dqn()



def show_video_of_model(agent, env_name):
    frames=[]
    env = gym.make(env_name)
    vid = video_recorder.VideoRecorder(env, path="video/{}.mp4".format(env_name))
    agent.value_net.load_state_dict(torch.load('ckpt_DDQN_64/ckpt5.pth',map_location=torch.device('cpu')))
    state = env.reset()
    done = False
    while not done:
        env.render()
        frames.append(env.render(mode="rgb_array"))
        vid.capture_frame()
        action = agent.take_action(state)
        state, reward, done, _ = env.step(action)
    env.close()
    return frames
agent = DQNAgent(state_size=8, action_size=4, batch_size = 64)
frames = show_video_of_model(agent, 'LunarLander-v2')

