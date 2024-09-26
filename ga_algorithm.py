import chess
import numpy as np
from typing import List, Tuple
import random
import pickle
import matplotlib.pyplot as plt

class GeneticAlgorithmBot:
    def __init__(self, population_size=100, generations=50, mutation_rate=0.05, tournament_size=5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        self.position_weights = self._initialize_position_weights()
        self.training_metrics = {'best_fitness': [], 'avg_fitness': []}

    def _initialize_position_weights(self) -> np.ndarray:
        return np.random.uniform(-1, 1, (6, 8, 8))

    def evaluate_board(self, board: chess.Board) -> float:
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                color_multiplier = 1 if piece.color == chess.WHITE else -1
                row, col = divmod(square, 8)
                position_value = self.position_weights[piece.piece_type - 1][row][col]
                score += color_multiplier * (value + position_value)
        return score

    def mutate(self, weights: np.ndarray) -> np.ndarray:
        mask = np.random.random(weights.shape) < self.mutation_rate
        mutations = np.random.normal(0, 0.1, weights.shape)
        return weights + mask * mutations

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        child = parent1.copy()
        mask = np.random.random(parent1.shape) < 0.5
        child[mask] = parent2[mask]
        return child

    def tournament_selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        selected = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
        return max(selected, key=lambda x: x[1])[0]

    def evolve_population(self, population: List[np.ndarray], fitness_scores: List[float]) -> List[np.ndarray]:
        new_population = []
        
        # Elitism: keep the best individual
        best_idx = fitness_scores.index(max(fitness_scores))
        new_population.append(population[best_idx])
        
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        return new_population

    def train(self, num_games=100):
        population = [self._initialize_position_weights() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            fitness_scores = []
            
            for individual in population:
                self.position_weights = individual
                score = self.play_games(num_games)
                fitness_scores.append(score)
            
            population = self.evolve_population(population, fitness_scores)
            
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.training_metrics['best_fitness'].append(best_fitness)
            self.training_metrics['avg_fitness'].append(avg_fitness)
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
        
        self.position_weights = population[fitness_scores.index(max(fitness_scores))]  # Use the best individual
        self.plot_training_metrics()

    def play_games(self, num_games: int) -> float:
        total_score = 0
        for _ in range(num_games):
            board = chess.Board()
            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    move = self.get_best_move(board)
                else:
                    move = random.choice(list(board.legal_moves))
                board.push(move)
            
            result = board.result()
            if result == "1-0":
                total_score += 1
            elif result == "1/2-1/2":
                total_score += 0.5
        
        return total_score / num_games

    def get_best_move(self, board: chess.Board) -> chess.Move:
        best_move = None
        best_score = float('-inf')
        
        for move in board.legal_moves:
            board.push(move)
            score = self.evaluate_board(board)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None:
            return random.choice(list(board.legal_moves))
        
        return best_move

    def step(self, state) -> int:
        board = chess.Board(state.observation_string())
        chess_move = self.get_best_move(board)
        
        print(f"Current board state: {board}")
        print(f"Chosen chess move: {chess_move}")
        
        # Convert chess move to OpenSpiel action
        openspiel_move = self.chess_move_to_openspiel(chess_move, board)
        print(f"Converted to OpenSpiel move: {openspiel_move}")
        
        legal_actions = state.legal_actions()
        for action in legal_actions:
            action_str = state.action_to_string(state.current_player(), action)
            if openspiel_move == action_str:
                return action
        
        print(f"Warning: Couldn't find matching action for {chess_move} ({openspiel_move}). Legal actions: {[state.action_to_string(state.current_player(), action) for action in legal_actions]}")
        return random.choice(legal_actions)

    def chess_move_to_openspiel(self, move: chess.Move, board: chess.Board) -> str:
        """Convert a chess.Move to OpenSpiel action string."""
        from_square = chess.SQUARE_NAMES[move.from_square]
        to_square = chess.SQUARE_NAMES[move.to_square]
        piece = board.piece_at(move.from_square)
        
        if piece is None:
            return f"{from_square}{to_square}"

        piece_symbol = piece.symbol().upper()
        
        # Handle castling
        if piece_symbol == 'K':
            if from_square == 'e1' and to_square == 'g1':
                return "e1g1"  # King-side castling for White
            elif from_square == 'e1' and to_square == 'c1':
                return "e1c1"  # Queen-side castling for White
            elif from_square == 'e8' and to_square == 'g8':
                return "e8g8"  # King-side castling for Black
            elif from_square == 'e8' and to_square == 'c8':
                return "e8c8"  # Queen-side castling for Black
        
        # Handle promotions
        if move.promotion:
            promotion_piece = chess.PIECE_SYMBOLS[move.promotion].upper()
            return f"{from_square}{to_square}{promotion_piece}"
        
        # Handle regular moves
        if piece_symbol == 'P':
            return f"{to_square}"
        else:
            return f"{piece_symbol}{to_square}"

    def plot_training_metrics(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_metrics['best_fitness'], label='Best Fitness')
        plt.plot(self.training_metrics['avg_fitness'], label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Training Progress')
        plt.legend()
        plt.savefig('training_progress.png')
        plt.close()

    def save_weights(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.position_weights, f)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename: str):
        with open(filename, 'rb') as f:
            self.position_weights = pickle.load(f)
        print(f"Weights loaded from {filename}")

    def inform_action(self, state, player_id, action):
        pass

    def restart(self):
        pass