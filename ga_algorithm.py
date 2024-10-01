import chess
import numpy as np
from typing import List, Tuple
import random
import pickle
import matplotlib.pyplot as plt

class GeneticAlgorithmBot:
    def __init__(self, population_size=100, generations=50, mutation_rate=0.1, tournament_size=10):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        self.piece_square_tables = self._initialize_piece_square_tables()
        self.endgame_weights = self._initialize_endgame_weights()
        self.mobility_weight = np.random.uniform(0, 0.1)
        self.king_safety_weight = np.random.uniform(0, 0.1)
        self.training_metrics = {'best_fitness': [], 'avg_fitness': []}

    def _initialize_piece_square_tables(self):
        # Initialize piece-square tables for each piece type
        return {piece_type: np.random.uniform(-1, 1, (8, 8)) for piece_type in chess.PIECE_TYPES}

    def _initialize_endgame_weights(self):
        # Initialize endgame weights
        return np.random.uniform(0, 1, 6)

    def evaluate_board(self, board: chess.Board) -> float:
        score = 0
        piece_count = sum(len(board.pieces(piece_type, color)) 
                          for piece_type in chess.PIECE_TYPES 
                          for color in [chess.WHITE, chess.BLACK])
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                color_multiplier = 1 if piece.color == chess.WHITE else -1
                row, col = divmod(square, 8)
                if piece.color == chess.BLACK:
                    row, col = 7 - row, col
                # Check if the piece type exists in the piece_square_tables
                if piece.piece_type in self.piece_square_tables:
                    position_value = self.piece_square_tables[piece.piece_type][row][col]
                    score += color_multiplier * (value + position_value)
                else:
                    score += color_multiplier * value

        # Evaluate mobility
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        board.turn = chess.WHITE
        score += self.mobility_weight * (white_mobility - black_mobility)

        # Evaluate king safety
        score += self.king_safety_weight * (self._evaluate_king_safety(board, chess.WHITE) - 
                                            self._evaluate_king_safety(board, chess.BLACK))

        # Apply endgame weights
        if piece_count <= 10:
            endgame_score = self._evaluate_endgame(board)
            score = 0.7 * score + 0.3 * endgame_score

        return score

    def _evaluate_king_safety(self, board: chess.Board, color: chess.Color) -> float:
        king_square = board.king(color)
        if king_square is None:
            return 0
        
        safety_score = 0
        for square in chess.SQUARES:
            if chess.square_distance(king_square, square) <= 2:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    safety_score += 1
        return safety_score

    def _evaluate_endgame(self, board: chess.Board) -> float:
        score = 0
        for piece_type in chess.PIECE_TYPES:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            score += self.endgame_weights[piece_type - 1] * (white_count - black_count)
        return score

    def mutate(self, weights: dict) -> dict:
        mutated_weights = {}
        for key, value in weights.items():
            if isinstance(value, np.ndarray):
                mask = np.random.random(value.shape) < self.mutation_rate
                mutations = np.random.normal(0, 0.2, value.shape)
                mutated_weights[key] = np.clip(value + mask * mutations, -1, 1)
            elif isinstance(value, (float, int)):
                if random.random() < self.mutation_rate:
                    mutated_weights[key] = np.clip(value + random.gauss(0, 0.2), 0, 1)
                else:
                    mutated_weights[key] = value
        return mutated_weights

    def crossover(self, parent1: dict, parent2: dict) -> dict:
        child = {}
        for key in parent1.keys():
            if isinstance(parent1[key], np.ndarray):
                alpha = np.random.random(parent1[key].shape)
                child[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]
            elif isinstance(parent1[key], (float, int)):
                child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def tournament_selection(self, population: List[dict], fitness_scores: List[float]) -> dict:
        selected = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
        return max(selected, key=lambda x: x[1])[0]

    def evolve_population(self, population: List[dict], fitness_scores: List[float]) -> List[dict]:
        new_population = []
        
        # Elitism: keep the top 10% of individuals
        elite_count = self.population_size // 10
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        new_population.extend([population[i] for i in elite_indices])
        
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        return new_population

    def train(self, num_games=100):
        population = [
            {
                'piece_square_tables': self._initialize_piece_square_tables(),
                'endgame_weights': self._initialize_endgame_weights(),
                'mobility_weight': np.random.uniform(0, 0.1),
                'king_safety_weight': np.random.uniform(0, 0.1)
            }
            for _ in range(self.population_size)
        ]
        
        for generation in range(self.generations):
            fitness_scores = []
            
            for individual in population:
                self.piece_square_tables = individual['piece_square_tables']
                self.endgame_weights = individual['endgame_weights']
                self.mobility_weight = individual['mobility_weight']
                self.king_safety_weight = individual['king_safety_weight']
                score = self.play_games(num_games)
                fitness_scores.append(score)
            
            population = self.evolve_population(population, fitness_scores)
            
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.training_metrics['best_fitness'].append(best_fitness)
            self.training_metrics['avg_fitness'].append(avg_fitness)
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
            
            # Early stopping if no improvement for 10 generations
            if generation > 10 and best_fitness <= max(self.training_metrics['best_fitness'][-10:]):
                print(f"Early stopping at generation {generation + 1}")
                break
        
        best_individual = population[fitness_scores.index(max(fitness_scores))]
        self.piece_square_tables = best_individual['piece_square_tables']
        self.endgame_weights = best_individual['endgame_weights']
        self.mobility_weight = best_individual['mobility_weight']
        self.king_safety_weight = best_individual['king_safety_weight']
        self.plot_training_metrics()

    def play_games(self, num_games: int) -> float:
        total_score = 0
        for _ in range(num_games):
            board = chess.Board()
            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    move = self.get_best_move(board)
                else:
                    move = self.get_best_move(board)  # Both sides use the AI
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
        plt.savefig('GA_Plots/training_progress.png')
        plt.close()

    def save_weights(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.piece_square_tables, f)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename: str):
        with open(filename, 'rb') as f:
            self.piece_square_tables = pickle.load(f)
        print(f"Weights loaded from {filename}")

    def inform_action(self, state, player_id, action):
        pass

    def restart(self):
        pass