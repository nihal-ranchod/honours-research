import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyspiel

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ChessNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChessNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class NFSPChessAgent:
    def __init__(self, game, player_id, hidden_size=512, learning_rate=0.001,
                 buffer_size=100000, batch_size=128, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        
        self.game = game
        self.player_id = player_id
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize state and action space sizes
        self.input_size = game.observation_tensor_size()
        self.output_size = game.num_distinct_actions()
        
        # Initialize networks
        self.q_network = ChessNet(self.input_size, hidden_size, self.output_size)
        self.target_network = ChessNet(self.input_size, hidden_size, self.output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
        self.win_rates = []
        self.epsilon_history = []
        
    def state_to_tensor(self, state):
        """Convert OpenSpiel state to PyTorch tensor"""
        obs_tensor = state.observation_tensor(self.player_id)
        return torch.FloatTensor(obs_tensor).unsqueeze(0)
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(state.legal_actions())
        
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            
            # Mask illegal actions with large negative values
            legal_actions = state.legal_actions()
            mask = torch.ones(self.output_size) * float('-inf')
            mask[legal_actions] = 0
            q_values += mask
            
            return q_values.argmax().item()
    
    def update_networks(self):
        """Update neural networks using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.cat([self.state_to_tensor(s) for s in batch.state])
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_state_batch = torch.cat([self.state_to_tensor(s) for s in batch.next_state])
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1)
        
        # Compute current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss and update Q-network
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def update_target_network(self):
        """Update target network parameters"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, path):
        """Save model parameters"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
    
    def load_model(self, path):
        """Load model parameters"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
    
    def plot_metrics(self):
        """Plot training metrics"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot loss
        ax1.plot(self.losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Update Step')
        ax1.set_ylabel('Loss')
        
        # Plot win rate
        ax2.plot(self.win_rates)
        ax2.set_title('Win Rate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate')
        
        # Plot epsilon
        ax3.plot(self.epsilon_history)
        ax3.set_title('Epsilon Value')
        ax3.set_xlabel('Update Step')
        ax3.set_ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig('nfsp_training_metrics.png')
        plt.close()
    
    def train(self, num_episodes, opponent=None):
        """Train the agent through self-play or against an opponent"""
        if opponent is None:
            opponent = NFSPChessAgent(self.game, 1 - self.player_id)
        
        wins = 0
        for episode in tqdm(range(num_episodes)):
            state = self.game.new_initial_state()
            done = False
            
            while not state.is_terminal():
                current_player = state.current_player()
                
                if current_player == self.player_id:
                    action = self.select_action(state, training=True)
                else:
                    action = opponent.select_action(state, training=False)
                
                # Store experience
                old_state = state.clone()
                state.apply_action(action)
                reward = state.returns()[self.player_id] if state.is_terminal() else 0
                
                self.memory.push(Experience(
                    old_state,
                    action,
                    reward,
                    state.clone(),
                    state.is_terminal()
                ))
                
                # Update networks
                self.update_networks()
                
                if episode % 10 == 0:
                    self.update_target_network()
            
            # Track wins
            if state.returns()[self.player_id] > 0:
                wins += 1
            
            # Calculate and store win rate every 100 episodes
            if (episode + 1) % 100 == 0:
                win_rate = wins / 100
                self.win_rates.append(win_rate)
                wins = 0
        
        return self.win_rates[-1] if self.win_rates else 0

# Integration with OpenSpiel environment
def create_nfsp_bot(game, player_id):
    """Factory function to create NFSP bot compatible with OpenSpiel"""
    agent = NFSPChessAgent(game, player_id)
    
    class NFSPBot:
        def __init__(self):
            self.agent = agent
        
        def step(self, state):
            return self.agent.select_action(state, training=False)
        
        def inform_action(self, state, player_id, action):
            pass
        
        def restart(self):
            pass
    
    return NFSPBot()


if __name__ == '__main__':
    game = pyspiel.load_game("chess")
    nfsp_agent = NFSPChessAgent(game, player_id=0)
    nfsp_agent.train(num_episodes=500)
    nfsp_agent.save_model("nfsp_chess_model")
    nfsp_agent.plot_metrics()