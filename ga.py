import chess
import chess.pgn
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import multiprocessing
from functools import partial
import pyspiel

class GeneticAlgorithmChessBot:
    def __init__(self, player_id, population_size=200, generations=1000, mutation_rate=0.005, crossover_rate=0.75, max_games=2000, search_depth=4):
        self.player_id = player_id
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_games = max_games
        self.search_depth = search_depth
        self.early_stopping_patience = 50
        self.piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
        }
        self.best_individual = None
        self.training_progress = []

    def _initialize_individual(self):
        return {
            'piece_square_tables': {piece: np.random.uniform(-10, 10, (8, 8)) for piece in chess.PIECE_TYPES},
            'mobility_weight': np.random.uniform(0, 1),
            'king_safety_weight': np.random.uniform(0, 1),
            'pawn_structure_weight': np.random.uniform(0, 1)
        }

    def _evaluate_board(self, board: chess.Board, individual) -> float:
        if board.is_checkmate():
            return 10000 if board.turn != self.player_id else -10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                color_multiplier = 1 if piece.color == chess.WHITE else -1
                row, col = divmod(square, 8)
                if piece.color == chess.BLACK:
                    row, col = 7 - row, col
                position_value = individual['piece_square_tables'][piece.piece_type][row][col]
                score += color_multiplier * (value + position_value)

        # Mobility
        mobility_score = len(list(board.legal_moves))
        score += individual['mobility_weight'] * mobility_score

        # King safety (simplified)
        king_safety_score = self._evaluate_king_safety(board)
        score += individual['king_safety_weight'] * king_safety_score

        # Pawn structure (simplified)
        pawn_structure_score = self._evaluate_pawn_structure(board)
        score += individual['pawn_structure_weight'] * pawn_structure_score

        return score if board.turn == chess.WHITE else -score

    def _evaluate_king_safety(self, board: chess.Board) -> float:
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        if white_king_square is None or black_king_square is None:
            return 0
        
        white_safety = sum(1 for sq in board.attacks(white_king_square) if board.color_at(sq) == chess.WHITE)
        black_safety = sum(1 for sq in board.attacks(black_king_square) if board.color_at(sq) == chess.BLACK)
        
        return white_safety - black_safety

    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        return len(white_pawns) - len(black_pawns)

    def _mutate(self, individual):
        for piece in individual['piece_square_tables']:
            mask = np.random.random(individual['piece_square_tables'][piece].shape) < self.mutation_rate
            individual['piece_square_tables'][piece] += mask * np.random.normal(0, 5, individual['piece_square_tables'][piece].shape)
        
        for weight in ['mobility_weight', 'king_safety_weight', 'pawn_structure_weight']:
            if random.random() < self.mutation_rate:
                individual[weight] += random.gauss(0, 0.1)
                individual[weight] = max(0, min(1, individual[weight]))
        
        return individual

    def _crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            if isinstance(parent1[key], dict):
                child[key] = {}
                for subkey in parent1[key]:
                    alpha = np.random.random(parent1[key][subkey].shape)
                    child[key][subkey] = alpha * parent1[key][subkey] + (1 - alpha) * parent2[key][subkey]
            else:
                child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def _tournament_selection(self, population, fitness_scores):
        tournament_size = 5
        selected = random.sample(list(zip(population, fitness_scores)), tournament_size)
        return max(selected, key=lambda x: x[1])[0]

    def _evaluate_individual(self, individual, games):
        total_score = 0
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                score = self._evaluate_board(board, individual)
                total_score += score if board.turn == chess.WHITE else -score
        return total_score / len(games)

    def _minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self._evaluate_board(board, self.best_individual)

        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def train(self, pgn_file):
        population = [self._initialize_individual() for _ in range(self.population_size)]
        
        games = self._load_pgn(pgn_file)
        games = games[:min(len(games), self.max_games)]  # Limit the number of games
        
        pool = multiprocessing.Pool()
        evaluate_partial = partial(self._evaluate_individual, games=games)

        best_fitness = float('-inf')
        generations_without_improvement = 0

        for generation in range(self.generations):
            fitness_scores = pool.map(evaluate_partial, population)
            
            current_best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.training_progress.append((current_best_fitness, avg_fitness))
            
            print(f"Generation {generation + 1}: Best Fitness = {current_best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}, Games: {len(games)}")
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                generations_without_improvement = 0
                self.best_individual = population[fitness_scores.index(current_best_fitness)]
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping: No improvement for {self.early_stopping_patience} generations.")
                break

            new_population = []
            elite_count = self.population_size // 10
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            new_population.extend([population[i] for i in elite_indices])
            
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._tournament_selection(population, fitness_scores)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        pool.close()
        pool.join()
        
        print(f"Training completed. Best fitness: {best_fitness:.2f}")
        self.plot_learning_curve()

    def _load_pgn(self, pgn_file):
        games = []
        with open(pgn_file) as f:
            for _ in range(self.max_games):
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
        return games

    def _chess_move_to_openspiel_action(self, board, chess_move):
        """Convert a chess.Move to an OpenSpiel action (integer)."""
        from_square = chess_move.from_square
        to_square = chess_move.to_square

        # Check if it's an en passant capture
        if board.is_en_passant(chess_move):
            # For en passant, we need to encode it differently
            # The format is: (from_square * 64 + to_square) + 64 * 64
            return (from_square * 64 + to_square) + 64 * 64
        
        # Check if it's a promotion
        if chess_move.promotion:
            # For promotions, we need to encode the promotion piece
            # The format is: (from_square * 64 + to_square) + piece_type * 64 * 64
            piece_type_offset = {
                chess.KNIGHT: 1,
                chess.BISHOP: 2,
                chess.ROOK: 3,
                chess.QUEEN: 4
            }
            return (from_square * 64 + to_square) + piece_type_offset[chess_move.promotion] * 64 * 64

        # Regular move
        return from_square * 64 + to_square

    def get_best_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            board.push(move)
            score = self._minimax(board, self.search_depth - 1, float('-inf'), float('inf'), False)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None:
            return None
        
        # Convert the chess.Move to an OpenSpiel action
        return self._chess_move_to_openspiel_action(board, best_move)

    def step(self, state):
        """
        This method is called by the OpenSpiel framework.
        It should return an action (integer) compatible with OpenSpiel.
        """
        board = chess.Board(fen=str(state))
        return self.get_best_move(board)

    def inform_action(self, state, player_id, action):
        """
        This method is called to inform the bot about actions taken by other players.
        For the GA bot, we don't need to do anything here, but we need to implement it
        to conform to the bot interface expected by OpenSpiel.
        """
        pass

    def restart(self):
        """
        This method is called when a game ends and a new one is about to start.
        For the GA bot, we don't need to do anything here, but we need to implement it
        to conform to the bot interface expected by OpenSpiel.
        """
        pass

    def plot_learning_curve(self):
        generations = range(1, len(self.training_progress) + 1)
        best_fitness, avg_fitness = zip(*self.training_progress)
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, label='Best Fitness')
        plt.plot(generations, avg_fitness, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Genetic Algorithm Learning Curve')
        plt.legend()
        plt.savefig('ga_learning_curve.png')
        plt.close()

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.best_individual, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.best_individual = pickle.load(f)
    def inform_action(self, state, player_id, action):
        pass

    def restart(self):
        pass