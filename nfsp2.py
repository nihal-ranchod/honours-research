import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pyspiel
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import os
from tqdm import tqdm
import chess
import chess.pgn

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ChessStateEncoder:
    """Encodes chess state into a neural network-compatible format"""
    def __init__(self):
        self.input_size = 8 * 8 * 6 * 2 + 8  # 8x8 board, 6 piece types, 2 colors, 8 extra features
        
    def encode_state(self, state):
        """Convert OpenSpiel state to tensor"""
        # Get FEN string from state
        fen = state.observation_string(0)  # Get observation for player 0
        board = chess.Board(fen)
        
        # Initialize the encoded state
        encoded = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Encode each piece
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = square // 8
                file = square % 8
                # Get piece index (0-5 for white pieces, 6-11 for black pieces)
                piece_idx = piece.piece_type - 1
                if not piece.color:  # If black
                    piece_idx += 6
                encoded[rank][file][piece_idx] = 1
        
        # Add extra features
        extra_features = np.zeros(8, dtype=np.float32)
        extra_features[0] = 1 if board.turn else 0  # Current player
        extra_features[1] = int(board.is_check())  # Check state
        extra_features[2] = board.fullmove_number / 100.0  # Normalized move number
        extra_features[3] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        extra_features[4] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        extra_features[5] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        extra_features[6] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
        extra_features[7] = board.halfmove_clock / 100.0  # Normalized halfmove clock
        
        # Flatten and concatenate
        return np.concatenate([encoded.flatten(), extra_features])

class ChessNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChessNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class AggressiveNFSP:
    def __init__(self, game, learning_rate=0.001, hidden_size=512, memory_size=100000, 
                 batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, 
                 epsilon_decay=0.995):
        self.game = game
        self.state_encoder = ChessStateEncoder()
        self.input_size = self.state_encoder.input_size
        self.output_size = 4672  # Maximum possible moves in chess
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = ChessNet(self.input_size, hidden_size, self.output_size).to(self.device)
        self.target_net = ChessNet(self.input_size, hidden_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training metrics
        self.losses = []
        self.avg_rewards = []
        
    def san_to_move(self, board, san_move):
        """Convert SAN move to chess.Move object"""
        try:
            return board.parse_san(san_move)
        except ValueError:
            return None

    def state_to_tensor(self, state):
        """Convert game state to tensor"""
        encoded_state = self.state_encoder.encode_state(state)
        return torch.FloatTensor(encoded_state).unsqueeze(0).to(self.device)
    
    def evaluate_position(self, board):
        """Evaluate chess position with emphasis on aggressive features"""
        if board.is_checkmate():
            return 1.0 if board.turn else -1.0
        
        score = 0.0
        # Material count with aggressive weights
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        for piece_type in piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
        
        # Bonus for attacking moves
        if board.is_check():
            score += 0.5
            
        # Bonus for controlling center
        center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    score += 0.2
                else:
                    score -= 0.2
                    
        return score
    
    def calculate_reward(self, state, action, next_state):
        """Calculate reward with emphasis on aggressive play"""
        try:
            # Get board states
            current_board = chess.Board(state.observation_string(0))
            next_board = chess.Board(next_state.observation_string(0))
            
            # Convert SAN move to chess.Move object
            san_move = state.action_to_string(state.current_player(), action)
            move = self.san_to_move(current_board, san_move)
            
            if move is None:
                return 0.0  # Return neutral reward if move parsing fails
            
            # Calculate position evaluation change
            current_eval = self.evaluate_position(current_board)
            next_eval = self.evaluate_position(next_board)
            reward = next_eval - current_eval
            
            # Add bonus rewards for aggressive play
            # Bonus for captures
            if current_board.is_capture(move):
                captured_piece = current_board.piece_at(move.to_square)
                if captured_piece:
                    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
                    reward += piece_values.get(captured_piece.symbol().upper(), 0) * 0.1
            
            # Bonus for checks
            if next_board.is_check():
                reward += 0.2
            
            # Bonus for attacking center squares
            center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
            if move.to_square in center_squares:
                reward += 0.1
            
            return reward
            
        except (ValueError, AttributeError) as e:
            print(f"Error calculating reward: {e}")
            return 0.0  # Return neutral reward in case of error
    
    def select_action(self, state, legal_actions, training=True):
        """Select action using epsilon-greedy policy with preference for aggressive moves"""
        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            
            # Mask illegal actions
            legal_actions_mask = torch.zeros(self.output_size, device=self.device)
            legal_actions_mask[legal_actions] = 1
            q_values = q_values * legal_actions_mask - 1e9 * (1 - legal_actions_mask)
            
            return q_values.argmax(1).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append(Experience(
            self.state_to_tensor(state).squeeze(0),
            action,
            torch.FloatTensor([reward]).to(self.device),
            self.state_to_tensor(next_state).squeeze(0) if next_state else None,
            done
        ))
    
    def train_batch(self):
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                    device=self.device, dtype=torch.bool)
        if any(non_final_mask):
            non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
            next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute expected Q values
        expected_q_values = reward_batch + self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes=1000):
        """Train the agent through self-play"""
        print(f"Training on {self.device}...")
        progress_bar = tqdm(range(num_episodes), desc="Training Episodes")
        
        for episode in progress_bar:
            state = self.game.new_initial_state()
            episode_reward = 0
            
            while not state.is_terminal():
                legal_actions = state.legal_actions()
                if not legal_actions:
                    break
                    
                current_player = state.current_player()
                
                # Select and apply action
                action = self.select_action(state, legal_actions, training=True)
                next_state = state.clone()
                next_state.apply_action(action)
                
                # Calculate reward
                reward = self.calculate_reward(state, action, next_state)
                episode_reward += reward if current_player == 0 else -reward
                
                # Store experience and train
                if current_player == 0:  # Only store experiences for the main agent
                    self.store_experience(state, action, reward, next_state, next_state.is_terminal())
                    loss = self.train_batch()
                    if loss:
                        self.losses.append(loss)
                
                state = next_state
            
            # Update metrics
            self.avg_rewards.append(episode_reward)
            
            # Update target network periodically
            if episode % 100 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Epsilon': f'{self.epsilon:.3f}',
                'Avg Reward': f'{np.mean(self.avg_rewards[-100:]):.3f}'
            })
            
            # Plot training progress periodically
            if (episode + 1) % 100 == 0:
                self.plot_training_progress()
                # self.save_model(f"aggressive_nfsp_model_episode_{episode+1}.pth")
        
        # Save final model
        self.save_model("aggressive_nfsp_model_final.pth")
        print("Training completed!")
    
    def plot_training_progress(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        
        # Plot average rewards
        plt.subplot(1, 2, 2)
        plt.plot(self.avg_rewards)
        plt.title('Average Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_metrics': {
                'losses': self.losses,
                'avg_rewards': self.avg_rewards,
            }
        }, path)
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['training_metrics']['losses']
        self.avg_rewards = checkpoint['training_metrics']['avg_rewards']

class NFSPBot(pyspiel.Bot):
    def __init__(self, game, player_id, model_path):
        pyspiel.Bot.__init__(self)
        self.agent = AggressiveNFSP(game)
        self.agent.load_model(model_path)
        self.player_id = player_id

    def step(self, state):
        legal_actions = state.legal_actions()
        if not legal_actions:
            return None
        return self.agent.select_action(state, legal_actions, training=False)
    
    def inform_action(self, state, player_id, action):
        pass

    def restart(self):
        pass

def train_agent():
    game = pyspiel.load_game("chess")
    agent = AggressiveNFSP(game)
    agent.train(num_episodes=5000)
    return agent

if __name__ == "__main__":
    trained_agent = train_agent()