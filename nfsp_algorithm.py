import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pyspiel
import random
import chess
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt

class ChessNet(nn.Module):
    def __init__(self, hidden_size=512):
        super(ChessNet, self).__init__()
        # 8x8x13 input: 8x8 board, 13 piece types (6 white, 6 black, empty)
        self.input_channels = 13
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, 4672)  # Max possible chess moves
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class NFSPChessAgent:
    def __init__(self, game, learning_rate=0.001, batch_size=32, replay_capacity=100000):
        self.game = game
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ChessNet().to(self.device)
        self.target_net = ChessNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.batch_size = batch_size
        
        self.piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'average_reward': []
        }

    def board_to_tensor(self, state):
        """Convert OpenSpiel chess state to tensor representation"""
        board = chess.Board(str(state))
        state_tensor = np.zeros((13, 8, 8), dtype=np.float32)
        
        # Fill the state tensor
        for i in range(64):
            rank, file = i // 8, i % 8
            piece = board.piece_at(i)
            if piece is not None:
                piece_idx = self.piece_to_index[piece.symbol()]
                state_tensor[piece_idx][rank][file] = 1
            else:
                state_tensor[12][rank][file] = 1  # Empty square channel
        
        return torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)

    def select_action(self, state, epsilon=0.1):
        """Select an action from the given state"""
        legal_actions = state.legal_actions()
        if not legal_actions:
            return None
            
        if random.random() < epsilon:
            return random.choice(legal_actions)
        
        with torch.no_grad():
            state_tensor = self.board_to_tensor(state)
            policy, _ = self.policy_net(state_tensor)
            
            # Get probabilities only for legal actions
            legal_probs = torch.zeros(len(legal_actions))
            for i, action in enumerate(legal_actions):
                legal_probs[i] = policy[0][action]
            
            # Select the legal action with highest probability
            action_idx = legal_probs.argmax().item()
            return legal_actions[action_idx]

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert states to tensors
        state_batch = torch.cat([self.board_to_tensor(s) for s in states])
        next_state_batch = torch.cat([self.board_to_tensor(s) for s in next_states])
        
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Compute policy loss
        current_policy, current_value = self.policy_net(state_batch)
        policy_loss = F.cross_entropy(current_policy, action_batch)
        
        # Compute value loss
        next_policy, next_value = self.target_net(next_state_batch)
        expected_value = reward_batch + (1 - done_batch) * 0.99 * next_value.squeeze()
        value_loss = F.mse_loss(current_value.squeeze(), expected_value.detach())
        
        # Combined loss
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['value_loss'].append(value_loss.item())
        
        return policy_loss.item(), value_loss.item()

    def train(self, num_episodes=1000, steps_per_update=100):
        print("Starting training...")
        episode_rewards = []
        
        for episode in tqdm(range(num_episodes)):
            state = self.game.new_initial_state()
            episode_reward = 0
            
            while not state.is_terminal():
                action = self.select_action(state)
                if action is None:
                    break
                    
                # Store current state
                current_state = state.clone()
                
                # Apply action and get next state
                state.apply_action(action)
                reward = state.returns()[current_state.current_player()]
                done = state.is_terminal()
                
                # Store experience
                self.replay_buffer.push(
                    current_state,
                    action,
                    reward,
                    state,
                    done
                )
                
                if len(self.replay_buffer) > self.batch_size:
                    self.train_step()
                
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            self.training_stats['average_reward'].append(episode_reward)
            
            # Update target network periodically
            if episode % steps_per_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.plot_training_progress()
        return episode_rewards

    def plot_training_progress(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot policy loss
        ax1.plot(self.training_stats['policy_loss'])
        ax1.set_title('Policy Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        
        # Plot value loss
        ax2.plot(self.training_stats['value_loss'])
        ax2.set_title('Value Loss')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Loss')
        
        # Plot average reward with smoothing
        window_size = 100
        rewards = np.array(self.training_stats['average_reward'])
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(smoothed_rewards)
        ax3.set_title('Average Reward (Smoothed)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        
        plt.tight_layout()
        plt.savefig('nfsp_training_progress.png')
        plt.close()

    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class NFSPBot:
    """Bot interface for OpenSpiel integration"""
    def __init__(self, game, player_id, agent_path):
        self.game = game
        self.player_id = player_id
        self.agent = NFSPChessAgent(game)
        self.agent.load_model(agent_path)
        
    def step(self, state):
        """Returns the action to take in the given state."""
        return self.agent.select_action(state, epsilon=0)
        
    def restart(self):
        pass
        
    def inform_action(self, state, player_id, action):
        pass

def train_nfsp_agent():
    game = pyspiel.load_game("chess")
    agent = NFSPChessAgent(game)
    rewards = agent.train(num_episodes=1000)
    agent.save_model("nfsp_chess_model.pth")
    print("Training completed and model saved!")
    return agent

if __name__ == "__main__":
    trained_agent = train_nfsp_agent()