import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import pyspiel
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class ChessNet(nn.Module):
    def __init__(self, hidden_size=256):
        super(ChessNet, self).__init__()
        # Input: 8x8x13 (6 piece types x 2 colors + 1 for empty squares)
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Policy head - output space is 64 * 64 (all possible from-to square combinations)
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 64, 64 * 64)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * 64, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Policy head
        policy = self.relu(self.policy_conv(x))
        policy = policy.view(-1, 32 * 64)
        policy = self.policy_fc(policy)
        
        # Value head
        value = self.relu(self.value_conv(x))
        value = value.view(-1, 32 * 64)
        value = self.relu(self.value_fc1(value))
        value = self.tanh(self.value_fc2(value))
        
        return policy, value

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class NFSPChess:
    def __init__(self, learning_rate=0.001, buffer_size=100000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ChessNet().to(self.device)
        self.target_net = ChessNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
        self.training_losses = []
        self.average_rewards = []

    def move_to_index(self, move):
        """Convert a chess.Move to an index between 0 and 4095 (64*64-1)"""
        return move.from_square * 64 + move.to_square

    def index_to_move(self, index):
        """Convert an index back to a chess.Move"""
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)
    
    def board_to_tensor(self, board):
        # Convert chess board to 8x8x13 tensor
        tensor = torch.zeros(13, 8, 8, dtype=torch.float32)
        piece_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
        
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                rank, file = i // 8, i % 8
                tensor[piece_idx[piece.symbol()]][rank][file] = 1
            else:
                rank, file = i // 8, i % 8
                tensor[12][rank][file] = 1
        
        return tensor

    def train_on_pgn(self, pgn_file, num_epochs=10):
        from chess import pgn
        import io
        
        epoch_losses = []
        
        with open(pgn_file) as f:
            for epoch in range(num_epochs):
                epoch_loss = 0
                num_games = 0
                
                f.seek(0)
                pbar = tqdm(desc=f"Epoch {epoch+1}/{num_epochs}")
                
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    board = game.board()
                    for move in game.mainline_moves():
                        # Create tensors with proper dtype
                        state = self.board_to_tensor(board)
                        action_idx = self.move_to_index(move)
                        
                        board.push(move)
                        next_state = self.board_to_tensor(board)
                        
                        # Determine reward based on position evaluation
                        reward = 0
                        if board.is_checkmate():
                            reward = 1 if board.turn else -1
                        elif board.is_stalemate():
                            reward = 0
                            
                        self.replay_buffer.push(state, action_idx, reward, next_state, board.is_game_over())
                        
                        if len(self.replay_buffer) > self.batch_size:
                            loss = self.train_step()
                            epoch_loss += loss
                            
                    num_games += 1
                    pbar.update(1)
                    if num_games % 100 == 0:
                        pbar.set_postfix({'loss': epoch_loss/max(1, num_games)})
                        
                        # Update target network periodically
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                
                avg_epoch_loss = epoch_loss / max(1, num_games)
                epoch_losses.append(avg_epoch_loss)
                self.training_losses.append(avg_epoch_loss)
                
                # Save intermediate model every epoch
                self.save_model(f"nfsp_checkpoint_epoch_{epoch+1}.pt")
                
                pbar.close()
                
        return epoch_losses

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0
            
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Properly clone and detach tensors
        state_batch = torch.stack([s.clone().detach() for s in batch[0]]).to(self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(self.device)
        next_state_batch = torch.stack([s.clone().detach() for s in batch[3]]).to(self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).to(self.device)
        
        # Compute policy loss
        policy, value = self.policy_net(state_batch)
        policy_loss = nn.CrossEntropyLoss()(policy, action_batch)
        
        # Compute value loss
        # Reshape value and next_value to match dimensions
        next_value = self.target_net(next_state_batch)[1].squeeze(-1)  # Remove last dimension
        value = value.squeeze(-1)  # Remove last dimension
        
        # Ensure reward_batch and done_batch have correct shape
        reward_batch = reward_batch.view(-1)
        done_batch = done_batch.view(-1)
        
        expected_value = reward_batch + (1 - done_batch) * 0.99 * next_value
        value_loss = nn.MSELoss()(value, expected_value)
        
        # Combined loss
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'average_rewards': self.average_rewards
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint['training_losses']
        self.average_rewards = checkpoint['average_rewards']

    def plot_training_progress(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_losses)
        plt.title('Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.average_rewards)
        plt.title('Average Reward')
        plt.xlabel('Games')
        plt.ylabel('Reward')
        
        plt.tight_layout()
        plt.savefig('NFSP_training_progress.png')
        plt.close()

class NFSPBot:
    """OpenSpiel bot implementation wrapper for NFSP"""
    def __init__(self, player_id, model_path):
        self.player_id = player_id
        self.nfsp = NFSPChess()
        self.nfsp.load_model(model_path)
        self.nfsp.policy_net.eval()

    def restart(self):
        pass

    def inform_action(self, state, player_id, action):
        pass

    def step(self, state):
        board = chess.Board(fen=str(state))
        state_tensor = self.nfsp.board_to_tensor(board).unsqueeze(0).to(self.nfsp.device)
        
        with torch.no_grad():
            policy, _ = self.nfsp.policy_net(state_tensor)
            
        # Get legal moves and their indices
        legal_moves = list(board.legal_moves)
        legal_move_indices = [self.nfsp.move_to_index(move) for move in legal_moves]
        
        # Get probabilities for legal moves only
        legal_move_probs = torch.softmax(policy[0][legal_move_indices], dim=0)
        
        # Sample move based on policy
        move_idx = torch.multinomial(legal_move_probs, 1).item()
        selected_move = legal_moves[move_idx]
        
        # Convert chess.Move to OpenSpiel action
        return state.string_to_action(selected_move.uci())