import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, state):
        return self.fc(state)

import numpy as np

class Environment:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = np.random.randint(self.size)
        self.destination_pos = np.random.randint(self.size)
        return self.get_state()

    def get_state(self):
        return np.array([self.agent_pos, self.destination_pos])

    def step(self, action):
        if action == 0:  # Move left
            self.agent_pos = max(0, self.agent_pos - 1)
        elif action == 1:  # Move right
            self.agent_pos = min(self.size - 1, self.agent_pos + 1)
        done = self.agent_pos == self.destination_pos
        reward = 1 if done else -1
        return self.get_state(), reward, done

import torch.optim as optim

# Create agent and optimizer
agent = Agent()
optimizer = optim.Adam(agent.parameters())

# Create environment
env = Environment()

# Training loop
for episode in range(1000):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    while not done:
        # Select action
        state_tensor = torch.FloatTensor(state)
        action_probs = nn.Softmax(dim=-1)(agent(state_tensor))
        action = torch.distributions.Categorical(action_probs).sample()
        log_prob = torch.log(action_probs[action])

        # Take a step
        state, reward, done = env.step(action.item())

        # Record log_prob and reward
        log_probs.append(log_prob)
        rewards.append(reward)

    # Compute policy gradient loss
    returns = [sum(rewards[i:]) for i in range(len(rewards))]
    loss = -sum(r * lp for r, lp in zip(returns, log_probs))

    # Perform optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if episode % 10 == 0:
        print(f"Episode {episode}: loss = {loss.item()}")

def print_env(env, agent_pos, destination_pos):
    for i in range(env.size):
        if i == agent_pos:
            print('A', end='')
        elif i == destination_pos:
            print('D', end='')
        else:
            print('-', end='')
    print()

# Testing loop
with torch.no_grad():
    state = env.reset()
    print_env(env, env.agent_pos, env.destination_pos)
    done = False
    while not done:
        # Select action
        state_tensor = torch.FloatTensor(state)
        action_probs = nn.Softmax(dim=-1)(agent(state_tensor))
        action = torch.distributions.Categorical(action_probs).sample()

        # Take a step
        state, reward, done = env.step(action.item())

        # Print current state
        print_env(env, env.agent_pos, env.destination_pos)
