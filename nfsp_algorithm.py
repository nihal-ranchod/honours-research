import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import pyspiel
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import chess.pgn
import io

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class ChessNetwork(nn.Module):
    def __init__(self, input_size=773, hidden_size=512, output_size=4096):  # Increased output size
        super(ChessNetwork, self).__init__()
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

class NFSPChessBot:
    def __init__(self, game, player_id, learning_rate=0.001, buffer_size=100000, 
                 batch_size=32, eta=0.1, model_path=None):
        self.game = game
        self.player_id = player_id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.eta = eta
        
        self.move_to_index = {}
        self.index_to_move = {}
        self.create_move_mapping()
        
        output_size = len(self.move_to_index)
        self.q_network = ChessNetwork(output_size=1)  # Q-network outputs single value
        self.target_network = ChessNetwork(output_size=1)
        self.average_policy_network = ChessNetwork(output_size=output_size)
        
        if model_path:
            self.load_model(model_path)
        
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.average_policy_network.parameters(), lr=learning_rate)
        
        self.reservoir_buffer = ReplayBuffer(buffer_size)
        self.sl_buffer = ReplayBuffer(buffer_size)
        
        self.training_losses = []
        self.win_rates = []
    
    def create_move_mapping(self):
        """Create mapping between moves and indices"""
        idx = 0
        board = chess.Board()
        
        # Generate all possible moves from starting position
        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:
                # Regular moves
                move = chess.Move(from_square, to_square)
                if move not in self.move_to_index:
                    self.move_to_index[str(move)] = idx
                    self.index_to_move[idx] = move
                    idx += 1
                
                # Promotion moves
                if ((from_square // 8 == 1 and to_square // 8 == 0) or 
                    (from_square // 8 == 6 and to_square // 8 == 7)):
                    for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        move = chess.Move(from_square, to_square, promotion=promotion)
                        if str(move) not in self.move_to_index:
                            self.move_to_index[str(move)] = idx
                            self.index_to_move[idx] = move
                            idx += 1
        
        print(f"Total number of possible moves: {len(self.move_to_index)}")
    
    def board_to_input(self, board):
        """Convert python-chess board to neural network input"""
        input_tensor = torch.zeros(773)
        
        # Piece placement
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = (piece.piece_type - 1) + (6 if piece.color else 0)
                input_tensor[square * 12 + piece_idx] = 1
        
        # Extra features
        base_idx = 768
        input_tensor[base_idx] = int(board.has_kingside_castling_rights(True))
        input_tensor[base_idx + 1] = int(board.has_queenside_castling_rights(True))
        input_tensor[base_idx + 2] = int(board.has_kingside_castling_rights(False))
        input_tensor[base_idx + 3] = int(board.has_queenside_castling_rights(False))
        input_tensor[base_idx + 4] = int(board.turn)
        
        return input_tensor
    
    def move_to_action(self, move):
        """Convert python-chess move to action index"""
        try:
            return self.move_to_index[str(move)]
        except KeyError:
            return None
    
    def action_to_move(self, action):
        """Convert action index to python-chess move"""
        try:
            return self.index_to_move[action]
        except KeyError:
            return None

    def train_on_pgn(self, pgn_file, num_games=1000):
        """Train the model using PGN data"""
        with open(pgn_file) as f:
            for _ in tqdm(range(num_games), desc="Training on PGN games"):
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                board = game.board()
                for move in game.mainline_moves():
                    action = self.move_to_action(move)
                    
                    if action is not None:
                        state_tensor = self.board_to_input(board)
                        board.push(move)
                        next_state_tensor = self.board_to_input(board)
                        
                        self.reservoir_buffer.push(Experience(
                            state_tensor,
                            action,
                            0,  # Intermediate reward
                            next_state_tensor,
                            False
                        ))
                        
                        if len(self.reservoir_buffer) >= self.batch_size:
                            self.train_step()
    
    def train_step(self):
        """Perform a single training step"""
        if len(self.reservoir_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.reservoir_buffer.sample(self.batch_size)
        state_batch = torch.stack([exp.state for exp in batch])
        action_batch = torch.tensor([exp.action for exp in batch], dtype=torch.long)
        next_state_batch = torch.stack([exp.next_state for exp in batch])
        
        # Train Q-network
        self.q_optimizer.zero_grad()
        q_values = self.q_network(state_batch)
        next_q_values = self.target_network(next_state_batch).detach()
        target_q_values = 0.99 * next_q_values  # Using gamma = 0.99
        q_loss = nn.MSELoss()(q_values, target_q_values)
        q_loss.backward()
        self.q_optimizer.step()
        
        # Train average policy network
        self.policy_optimizer.zero_grad()
        logits = self.average_policy_network(state_batch)
        policy_loss = nn.CrossEntropyLoss()(logits, action_batch)
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.training_losses.append(q_loss.item())
    
    def step(self, board):
        """Choose an action using the current policy"""
        if random.random() < self.eta:
            # Use best response policy (Q-network)
            with torch.no_grad():
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    return None
                
                action_values = []
                for move in legal_moves:
                    action = self.move_to_action(move)
                    if action is not None:
                        next_board = board.copy()
                        next_board.push(move)
                        next_state_tensor = self.board_to_input(next_board)
                        value = self.q_network(next_state_tensor.unsqueeze(0))
                        action_values.append((action, value.item()))
                
                if action_values:
                    best_action = max(action_values, key=lambda x: x[1])[0]
                    return self.action_to_move(best_action)
                return random.choice(legal_moves)
        else:
            # Use average policy network
            with torch.no_grad():
                state_tensor = self.board_to_input(board)
                logits = self.average_policy_network(state_tensor.unsqueeze(0))[0]
                
                # Create mask for legal moves
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    return None
                
                legal_moves_mask = torch.zeros_like(logits)
                legal_actions = []
                for move in legal_moves:
                    action = self.move_to_action(move)
                    if action is not None:
                        legal_moves_mask[action] = 1
                        legal_actions.append(action)
                
                if not legal_actions:
                    return random.choice(legal_moves)
                
                masked_logits = logits * legal_moves_mask
                masked_logits[masked_logits == 0] = float('-inf')
                probs = torch.softmax(masked_logits, dim=0)
                
                try:
                    action = torch.multinomial(probs, 1).item()
                    return self.action_to_move(action)
                except:
                    return random.choice(legal_moves)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'average_policy_network': self.average_policy_network.state_dict(),
            'move_to_index': self.move_to_index,
            'index_to_move': self.index_to_move
        }, path)
    
    def load_model(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.average_policy_network.load_state_dict(checkpoint['average_policy_network'])
        self.move_to_index = checkpoint['move_to_index']
        self.index_to_move = checkpoint['index_to_move']
    
    def plot_training_progress(self):
        """Visualize training metrics"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_losses)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.win_rates)
        plt.title('Win Rate')
        plt.xlabel('Game')
        plt.ylabel('Win Rate')
        
        plt.tight_layout()
        plt.savefig('nfsp_training_progress.png')

if __name__ == '__main__':
    game = pyspiel.load_game("chess")
    nfsp_standard = NFSPChessBot(game, player_id=0)
    nfsp_standard.train_on_pgn("PGN_Data/lichess_db_standard_rated_2013-01.pgn")
    nfsp_standard.save_model("nfsp_standard.pth")