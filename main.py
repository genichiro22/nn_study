import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, state):
        return self.fc(state)

import numpy as np

class Environment:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = np.random.randint(self.size, size=2)  # 2D position
        self.destination_pos = np.random.randint(self.size, size=2)  # 2D position
        return self.get_state()

    def get_state(self):
        return np.concatenate([self.agent_pos, self.destination_pos])

    def step(self, action):
        if action == 0:  # Move left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Move right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Move up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Move down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        done = np.array_equal(self.agent_pos, self.destination_pos)
        reward = 1 if done else -0.1
        return self.get_state(), reward, done

import torch.optim as optim

# Create agent and optimizer
agent = Agent()
optimizer = optim.Adam(agent.parameters())

# Create environment
env = Environment()

# Training loop
for episode in range(10000):
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
    if episode % 100 == 0:
        print(f"Episode {episode}: loss = {loss.item()}")

torch.save(agent.state_dict(), 'agent_model.pth')

def print_env(env):
    for i in range(env.size):
        for j in range(env.size):
            if np.array_equal([i, j], env.agent_pos):
                print('A', end=' ')
            elif np.array_equal([i, j], env.destination_pos):
                print('D', end=' ')
            else:
                print('-', end=' ')
        print()  # newline at the end of each row
    print()  # newline to separate each step


# Testing loop
with torch.no_grad():
    state = env.reset()
    print_env(env)
    done = False
    while not done:
        # Select action
        state_tensor = torch.FloatTensor(state)
        action_probs = nn.Softmax(dim=-1)(agent(state_tensor))
        action = torch.distributions.Categorical(action_probs).sample()

        # Take a step
        state, reward, done = env.step(action.item())

        # Print current state
        print_env(env)
