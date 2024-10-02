import chess
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python.observation import make_observation

class GeneticAlgorithmChessBot(pyspiel.Bot):
    def __init__(self, player_id, population_size=200, generations=100, mutation_rate=0.1):
        super().__init__()
        self.player_id = player_id
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
        }
        self.best_individual = None
        self.fitness_history = []

    def create_individual(self):
        return {
            'piece_weights': np.random.uniform(0.8, 1.2, len(self.piece_values)),
            'position_weights': np.random.uniform(-10, 10, (6, 64)),  # One 8x8 grid for each piece type
            'mobility_weight': np.random.uniform(0, 0.1),
            'pawn_structure_weight': np.random.uniform(0, 0.1),
            'king_safety_weight': np.random.uniform(0, 0.1)
        }

    def evaluate_board(self, board, individual):
        score = 0
        for piece_type, base_value in self.piece_values.items():
            for square in board.pieces(piece_type, chess.WHITE):
                score += base_value * individual['piece_weights'][piece_type - 1]
                score += individual['position_weights'][piece_type - 1][square]
            for square in board.pieces(piece_type, chess.BLACK):
                score -= base_value * individual['piece_weights'][piece_type - 1]
                score -= individual['position_weights'][piece_type - 1][63 - square]  # Flip for black

        # Mobility
        score += len(list(board.legal_moves)) * individual['mobility_weight']

        # Pawn structure
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        score += (len(white_pawns) - len(black_pawns)) * individual['pawn_structure_weight']
        score += (self.count_doubled_pawns(white_pawns) - self.count_doubled_pawns(black_pawns)) * individual['pawn_structure_weight'] * -10

        # King safety
        score += self.evaluate_king_safety(board, chess.WHITE, individual) - self.evaluate_king_safety(board, chess.BLACK, individual)

        return score if board.turn == chess.WHITE else -score

    def count_doubled_pawns(self, pawns):
        files = [0] * 8
        for pawn in pawns:
            files[chess.square_file(pawn)] += 1
        return sum(f - 1 for f in files if f > 1)

    def evaluate_king_safety(self, board, color, individual):
        king_square = board.king(color)
        if king_square is None:
            return 0
        
        safety = 0
        for square in chess.SQUARES:
            if chess.square_distance(king_square, square) <= 2:
                if board.is_attacked_by(not color, square):
                    safety -= 10
                elif board.is_attacked_by(color, square):
                    safety += 5
        
        return safety * individual['king_safety_weight']

    def play_game(self, individual1, individual2):
        board = chess.Board()
        for _ in range(200):  # Max 200 moves
            if board.is_game_over():
                break
            if board.turn == chess.WHITE:
                move = self.get_best_move(board, individual1, list(board.legal_moves))
            else:
                move = self.get_best_move(board, individual2, list(board.legal_moves))
            board.push(move)
        return board.result()

    def get_best_move(self, board, individual, legal_moves):
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        for move in legal_moves:
            board.push(move)
            score = self.evaluate_board(board, individual)
            board.pop()
            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        return best_move or random.choice(legal_moves)

    def fitness(self, individual):
        score = 0
        for _ in range(10):  # Play 10 games against random opponent
            opponent = self.create_individual()
            result = self.play_game(individual, opponent)
            if result == '1-0':
                score += 1
            elif result == '1/2-1/2':
                score += 0.5
        return score

    def select_parent(self, population, fitnesses):
        tournament_size = 5
        selected = random.sample(list(zip(population, fitnesses)), tournament_size)
        return max(selected, key=lambda x: x[1])[0]

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            if isinstance(parent1[key], np.ndarray):
                mask = np.random.rand(*parent1[key].shape) < 0.5
                child[key] = np.where(mask, parent1[key], parent2[key])
            else:
                child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def mutate(self, individual):
        for key, value in individual.items():
            if isinstance(value, np.ndarray):
                mask = np.random.random(value.shape) < self.mutation_rate
                mutation = np.random.normal(0, 0.1, value.shape)
                individual[key] = np.where(mask, value + mutation, value)
            else:
                if random.random() < self.mutation_rate:
                    individual[key] += np.random.normal(0, 0.05)
        return individual

    def train(self):
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]
            best_fitness = max(fitnesses)
            self.fitness_history.append(best_fitness)
            print(f"Generation {generation + 1}/{self.generations}, Best Fitness: {best_fitness}")
            
            new_population = []
            elite_size = self.population_size // 10
            sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
            new_population.extend(sorted_population[:elite_size])
            
            while len(new_population) < self.population_size:
                parent1 = self.select_parent(population, fitnesses)
                parent2 = self.select_parent(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        best_index = fitnesses.index(max(fitnesses))
        self.best_individual = population[best_index]

    def step(self, state):
        if self.best_individual is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Get legal actions from the OpenSpiel state
        legal_actions = state.legal_actions()
        
        # Convert OpenSpiel state to a chess.Board object
        board = chess.Board(state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID))
        
        # Get all legal moves from the chess board
        legal_chess_moves = list(board.legal_moves)
        
        while legal_chess_moves:
            move = self.get_best_move(board, self.best_individual, legal_chess_moves)
            # Convert chess.Move to OpenSpiel action string
            action_string = move.uci()
            # Check if the action is in the legal actions
            if action_string in [state.action_to_string(action) for action in legal_actions]:
                return state.string_to_action(action_string)
            
            # If the chosen move is not legal, remove it from consideration and try again
            legal_chess_moves.remove(move)
        
        # If no legal moves are left, return a random legal action
        return random.choice(legal_actions)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.best_individual, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.best_individual = pickle.load(f)

    def plot_learning_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history)
        plt.title('Learning Curve')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.savefig('learning_curve.png')
        plt.close()

    def restart(self):
        pass

    def inform_action(self, state, player_id, action):
        pass