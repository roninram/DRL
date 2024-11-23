import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Neural network architecture for Deep Q-Learning
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Define a simple feedforward neural network:
        # input_size -> 64 -> 64 -> output_size (number of actions)
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),      # First hidden layer
            nn.ReLU(),                      # Activation function
            nn.Linear(64, 64),              # Second hidden layer
            nn.ReLU(),                      # Activation function
            nn.Linear(64, output_size)      # Output layer (Q-values for each action)
        )

    def forward(self, x):
        return self.network(x)

# Agent that learns to play Pong using Deep Q-Learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Experience replay buffer to store and sample past experiences
        self.memory = deque(maxlen=2000)    # Store last 2000 experiences
        
        # Hyperparameters for Q-learning
        self.gamma = 0.95      # Discount factor for future rewards
        self.epsilon = 1.0     # Initial exploration rate (100% random actions)
        self.epsilon_min = 0.01  # Minimum exploration rate (1% random actions)
        self.epsilon_decay = 0.995  # Rate at which exploration decreases
        self.learning_rate = 0.001  # Learning rate for the optimizer
        
        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the Q-network and optimizer
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        # Explore: choose random action
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploit: choose best action based on Q-values
        with torch.no_grad():  # No need to compute gradients for forward pass
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return np.argmax(act_values.cpu().numpy())

    def replay(self, batch_size):
        """Train the network using random samples from replay memory"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare batch data
        states = np.array([i[0] for i in minibatch])
        states = torch.FloatTensor(states).squeeze(1).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states = np.array([i[3] for i in minibatch])
        next_states = torch.FloatTensor(next_states).squeeze(1).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)

        # Compute Q values for current states and chosen actions
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Compute Q values for next states (target network concept)
        next_q = self.model(next_states).max(1)[0].detach()
        # Compute target Q values using Bellman equation
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Update network weights
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    """Main training loop"""
    # Initialize Pong environment with visual display
    env = gym.make('Pong-v0', render_mode="human")
    
    # Calculate state size (flattened RGB image)
    state_size = 210 * 160 * 3  # Height * Width * RGB channels
    action_size = env.action_space.n  # Number of possible actions
    
    # Initialize DQN agent
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000

    # Training loop
    for e in range(episodes):
        # Reset environment for new episode
        state = env.reset()[0]
        state = state.flatten()  # Flatten 3D image to 1D array
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        # Episode loop
        for time in range(500):  # Max 500 steps per episode
            env.render()  # Display game
            # Choose and perform action
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Process new state
            next_state = next_state.flatten()
            next_state = np.reshape(next_state, [1, state_size])
            
            # Store experience and update state
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Episode ended
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
            
            # Train on past experiences
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    env.close()

if __name__ == "__main__":
    main()
