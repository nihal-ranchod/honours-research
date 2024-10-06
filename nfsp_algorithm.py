import numpy as np
import chess
import chess.engine
import random
import tensorflow as tf
import matplotlib.pyplot as plt

class NFSPChessAgent:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        self.model = self.build_model(input_dim, hidden_dim, output_dim, learning_rate)
        self.memory = []
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate of epsilon
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def build_model(self, input_dim, hidden_dim, output_dim, learning_rate):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden_dim, input_dim=input_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_moves):
        """ 
        Select an action: exploration (random legal move) or exploitation (best move predicted by the model).
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)  # Explore by selecting a random legal move
        state = np.array([state])
        q_values = self.model.predict(state)
        
        # Select the best legal move based on predicted Q-values
        legal_q_values = [(move, q_values[0][self.move_to_action(move)]) for move in legal_moves]
        return max(legal_q_values, key=lambda x: x[1])[0]  # Pick the move with the highest Q-value

    def replay(self):
        """ Train the model using randomly sampled mini-batch from memory """
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.array([next_state])
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            
            # Predict the current Q-values and update for the selected action
            target_f = self.model.predict(np.array([state]))
            target_f[0][self.move_to_action(action)] = target

            # Perform a gradient descent step
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

        # Update epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def move_to_action(self, move):
        """Convert chess move to a unique action index."""
        return chess.SQUARES.index(move.from_square) * 64 + chess.SQUARES.index(move.to_square)

    def action_to_move(self, action):
        """Convert action index back to a chess move."""
        from_square = action // 64
        to_square = action % 64
        return chess.Move(from_square, to_square)

    def get_board_state(self, board):
        """
        Convert the current chess board into a feature vector.
        Uses an 8x8 grid to represent the board, with each piece assigned a number.
        """
        piece_to_value = {
            chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
            chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
        }

        board_state = np.zeros((8, 8), dtype=int)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                value = piece_to_value[piece.piece_type]
                board_state[row][col] = value if piece.color == chess.WHITE else -value

        return board_state.flatten()  # Flatten into a 64-length vector

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

# Training and evaluation logic
def train_nfsp_agent(agent, num_episodes, stockfish_path):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    episode_rewards = []
    for episode in range(num_episodes):
        board = chess.Board()
        total_reward = 0
        done = False

        while not board.is_game_over():
            state = agent.get_board_state(board)
            legal_moves = list(board.legal_moves)

            # Agent's move
            action = agent.act(state, legal_moves)
            move = chess.Move.from_uci(str(action))
            
            if move in board.legal_moves:
                board.push(move)
                reward = 1 if board.is_checkmate() else 0
            else:
                reward = -1  # Penalize illegal moves
                done = True

            next_state = agent.get_board_state(board)
            agent.remember(state, move, reward, next_state, done)

            # Stockfish's move
            if not board.is_game_over():
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)

            total_reward += reward
            agent.replay()

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {total_reward}")

    engine.quit()
    return episode_rewards

# Visualization of learning progress
def plot_learning_curve(episode_rewards):
    plt.plot(episode_rewards)
    plt.title('Learning Progress of NFSP Agent')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

# Usage example
if __name__ == "__main__":
    input_dim = 64  # The flattened 8x8 chess board
    hidden_dim = 128
    output_dim = 4096  # 64 squares x 64 possible target squares
    learning_rate = 0.001
    num_episodes = 500
    stockfish_path = "stockfish/stockfish" 

    agent = NFSPChessAgent(input_dim, hidden_dim, output_dim, learning_rate)
    
    # Train the agent
    rewards = train_nfsp_agent(agent, num_episodes, stockfish_path)

    # Save model weights
    agent.save_weights("nfsp_chess_model.h5")

    # Plot the learning curve
    plot_learning_curve(rewards)
