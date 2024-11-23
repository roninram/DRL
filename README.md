
```markdown
# Deep Q-Learning Pong Agent

This project implements a Deep Q-Learning (DQL) agent that learns to play the Atari game Pong using PyTorch and OpenAI Gym.

## Overview

The agent uses a deep neural network to learn optimal actions directly from pixel inputs. It employs several key reinforcement learning concepts:
- Deep Q-Network (DQN)
- Experience Replay
- Epsilon-Greedy Exploration

## Requirements

```python
pip install -r requirements.txt
```

Required packages:
- PyTorch
- OpenAI Gym
- NumPy
- Python 3.7+

## Project Structure

```
pong_DRL/
│
├── pong_DRL.py          # Main implementation file
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## How It Works

1. **Neural Network Architecture**
   - Input: Flattened game screen (210x160x3 RGB pixels)
   - Hidden layers: 2 fully connected layers (64 neurons each)
   - Output: Q-values for each possible action

2. **DQN Agent Features**
   - Experience replay buffer (stores 2000 recent experiences)
   - Epsilon-greedy exploration (starts at 100%, decays to 1%)
   - Learning rate: 0.001
   - Discount factor (gamma): 0.95

3. **Training Process**
   - Episodes: 1000
   - Max steps per episode: 500
   - Batch size: 32
   - Visual rendering enabled

## Usage

To run the training:

```bash
python pong_DRL.py
```

The program will:
1. Open a window showing the Pong game
2. Train the agent while displaying gameplay
3. Print episode scores and exploration rate

## Training Output

The program prints progress information in the format:
```
Episode: X/1000, Score: Y, Epsilon: Z
```
Where:
- X: Current episode number
- Y: Total reward for the episode
- Z: Current exploration rate (epsilon)

## Performance Notes

- Training with visual rendering is slower but allows monitoring of progress
- GPU acceleration is automatically used if available
- Initial performance will be poor as the agent explores randomly
- Performance should improve as epsilon decreases and the agent learns

## Customization

Key parameters that can be modified:
- `episodes`: Number of training episodes
- `batch_size`: Size of training batches
- `epsilon_decay`: Rate of exploration decay
- `memory`: Size of replay buffer
- Neural network architecture in the `DQN` class

## Known Issues

- The environment may show deprecation warnings for 'Pong-v0'
- Training can be slow on CPU-only systems
- High memory usage due to storing raw pixel data

## Future Improvements

Potential enhancements:
- Implement a target network for more stable training
- Add frame stacking for temporal information
- Implement prioritized experience replay
- Add model saving/loading functionality
- Optimize state preprocessing for better performance

## License

This project is open source and available under the MIT License.

## Acknowledgments

This implementation is based on the DQN algorithm introduced by DeepMind and uses OpenAI's Gym framework for the Pong environment.
```
