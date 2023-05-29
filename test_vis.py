import torch
import torch.nn as nn
import numpy as np

from main import Environment, Agent, print_env
# Create a new instance of the model
agent = Agent()

# Load the saved parameters into the new model
agent.load_state_dict(torch.load('agent_model.pth'))

env = Environment()

with torch.no_grad():
    state = env.reset()
    dist = np.abs(env.agent_pos[0] - env.destination_pos[0]) + np.abs(env.agent_pos[1] - env.destination_pos[1])
    dist = dist.copy()
    print_env(env)
    done = False
    s = 0
    while not done:
        # Select action
        state_tensor = torch.FloatTensor(state)
        action_probs = nn.Softmax(dim=-1)(agent(state_tensor))
        action = torch.distributions.Categorical(action_probs).sample()

        # Take a step
        state, reward, done = env.step(action.item())

        # Print current state
        print_env(env)
        s += 1
    print(s,dist)