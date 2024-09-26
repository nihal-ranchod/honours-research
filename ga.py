import random
import numpy as np
import chess
import pickle
from typing import List, Tuple
import pyspiel

class GeneticAlgorithmBot:
    def __init__(self, population_size=50, generations=10, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        self.position_weights = self._initialize_position_weights()

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
        mutations = np.random.uniform(-0.2, 0.2, weights.shape)
        return weights + mask * mutations

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        child = parent1.copy()
        mask = np.random.random(parent1.shape) < 0.5
        child[mask] = parent2[mask]
        return child

    def evolve_population(self, population: List[np.ndarray], fitness_scores: List[float]) -> List[np.ndarray]:
        # Convert to list of tuples for sorting
        population_with_fitness = list(zip(fitness_scores, population))
        
        # Sort based on fitness scores
        sorted_population = sorted(population_with_fitness, key=lambda x: x[0], reverse=True)
        
        new_population = []
        
        # Elitism: keep the best individual
        new_population.append(sorted_population[0][1])
        
        while len(new_population) < self.population_size:
            # Select parents from the top half of the sorted population
            parent1, parent2 = random.choices([p[1] for p in sorted_population[:len(sorted_population)//2]], k=2)
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
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
        
        self.position_weights = population[0]  # Use the best individual

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
        
        return best_move

    def step(self, state) -> int:
        board = chess.Board(state.observation_string())
        move = self.get_best_move(board)
        
        # Convert chess move to OpenSpiel action
        legal_actions = state.legal_actions()
        for action in legal_actions:
            if state.action_to_string(action) == move.uci():
                return action
        
        # If no matching action found, return a random legal action
        print(f"Warning: Couldn't find matching action for {move.uci()}. Choosing random action.")
        return random.choice(legal_actions)

    def inform_action(self, state, player_id, action):
        pass

    def restart(self):
        pass
    
    def save_weights(self, filename: str):
        """Save the trained weights to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.position_weights, f)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename: str):
        """Load trained weights from a file."""
        with open(filename, 'rb') as f:
            self.position_weights = pickle.load(f)
        print(f"Weights loaded from {filename}")
