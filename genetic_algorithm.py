import numpy as np
import chess
import chess.pgn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle
import io
import random
from collections import defaultdict
import csv

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input: 8x8x12 (6 piece types x 2 colors)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)  # Output single evaluation score
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Output between -1 and 1
        return x

class LoadedChessModel:
    """Wrapper class for loaded models to interface with OpenSpiel"""
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.model.eval()  # Set to evaluation mode
        
    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        pieces = ['p', 'n', 'b', 'r', 'q', 'k']
        tensor = torch.zeros(12, 8, 8)
        
        for i in range(64):
            piece = board.piece_at(i)
            if piece is None:
                continue
            
            rank = i // 8
            file = i % 8
            piece_type = pieces.index(piece.symbol().lower())
            color_idx = 0 if piece.color else 6
            tensor[piece_type + color_idx][rank][file] = 1
            
        return tensor

    def evaluate_position(self, board: chess.Board) -> float:
        tensor = self.board_to_tensor(board)
        with torch.no_grad():
            return self.model(tensor.unsqueeze(0)).item()

    def convert_move_to_openspiel(self, move: chess.Move) -> str:
        """Convert python-chess move to OpenSpiel format"""
        # OpenSpiel uses lowercase letters for files and numbers for ranks
        # without any additional characters
        uci = move.uci()
        from_square = uci[:2]
        to_square = uci[2:4]
        promotion = uci[4:] if len(uci) > 4 else ''
        
        # For promotions, OpenSpiel uses format like 'e7e8q'
        if promotion:
            return f"{from_square}{to_square}{promotion}"
        else:
            return f"{from_square}{to_square}"

    def get_legal_moves_from_state(self, state) -> List[str]:
        """Get legal moves in OpenSpiel format"""
        legal_moves = []
        for action in state.legal_actions():
            move_str = state.action_to_string(state.current_player(), action)
            legal_moves.append(move_str)
        return legal_moves

    def step(self, state) -> int:
        """Interface with OpenSpiel - returns action ID"""
        board = chess.Board(fen=str(state))
        legal_moves = list(board.legal_moves)
        openspiel_legal_moves = self.get_legal_moves_from_state(state)
        
        if not legal_moves:
            return None
        
        # Evaluate all legal moves
        move_scores = []
        best_score = float('-inf')
        best_move = None
        best_action = None
        
        for move in legal_moves:
            board.push(move)
            score = self.evaluate_position(board)
            
            # Convert move to OpenSpiel format
            openspiel_move = self.convert_move_to_openspiel(move)
            
            # Only consider the move if it's in OpenSpiel's legal moves
            if openspiel_move in openspiel_legal_moves:
                if score > best_score:
                    best_score = score
                    best_move = openspiel_move
                    best_action = state.string_to_action(state.current_player(), openspiel_move)
            
            board.pop()
        
        if best_action is None:
            # If we couldn't find a valid move, choose randomly from OpenSpiel's legal moves
            import random
            best_action = random.choice(state.legal_actions())
        
        return best_action

    def restart(self):
        """Required by OpenSpiel interface"""
        pass

    def inform_action(self, state, player_id, action):
        """Required by OpenSpiel interface"""
        pass

class GeneticChessBot:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.population = [ChessNet() for _ in range(population_size)]
        self.fitness_history = []
        self.best_fitness = float('-inf')
        self.best_genome = None

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        pieces = ['p', 'n', 'b', 'r', 'q', 'k']
        tensor = torch.zeros(12, 8, 8)
        
        for i in range(64):
            piece = board.piece_at(i)
            if piece is None:
                continue
            
            rank = i // 8
            file = i % 8
            piece_type = pieces.index(piece.symbol().lower())
            color_idx = 0 if piece.color else 6
            tensor[piece_type + color_idx][rank][file] = 1
            
        return tensor

    def evaluate_position(self, model: ChessNet, board: chess.Board) -> float:
        tensor = self.board_to_tensor(board)
        with torch.no_grad():
            return model(tensor.unsqueeze(0)).item()

    def crossover(self, parent1: ChessNet, parent2: ChessNet) -> ChessNet:
        child = ChessNet()
        for param_name, param in child.named_parameters():
            if random.random() < 0.5:
                param.data.copy_(parent1.state_dict()[param_name])
            else:
                param.data.copy_(parent2.state_dict()[param_name])
        return child

    def mutate(self, genome: ChessNet, mutation_rate=0.1, mutation_strength=0.1):
        for param in genome.parameters():
            mask = torch.rand_like(param) < mutation_rate
            noise = torch.randn_like(param) * mutation_strength
            param.data += mask.float() * noise

    def train_on_pgn(self, pgn_file: str, num_generations: int, games_per_genome: int):
        """Train the population on standard chess games"""
        for generation in range(num_generations):
            fitness_scores = []
            
            for genome in self.population:
                score = 0
                with open(pgn_file) as f:
                    for _ in range(games_per_genome):
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                            
                        board = game.board()
                        result = game.headers["Result"]
                        target = 0
                        if result == "1-0":
                            target = 1
                        elif result == "0-1":
                            target = -1
                            
                        eval = self.evaluate_position(genome, board)
                        score -= (eval - target) ** 2
                
                fitness_scores.append(score / games_per_genome)
            
            max_fitness = max(fitness_scores)
            self.fitness_history.append(max_fitness)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_genome = self.population[fitness_scores.index(max_fitness)]
            
            sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), 
                                                    key=lambda pair: pair[0], reverse=True)]
            
            new_population = sorted_population[:2]
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(sorted_population[:10], 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            print(f"Generation {generation + 1}/{num_generations}, Best Fitness: {max_fitness:.4f}")

    def train_on_puzzles(self, puzzle_pgn_file: str, num_generations: int, puzzles_per_genome: int):
        """Train the population on chess puzzles"""
        for generation in range(num_generations):
            fitness_scores = []
            
            for genome in self.population:
                score = 0
                with open(puzzle_pgn_file) as f:
                    for _ in range(puzzles_per_genome):
                        puzzle = chess.pgn.read_game(f)
                        if puzzle is None:
                            break
                            
                        board = puzzle.board()
                        moves = list(puzzle.mainline_moves())
                        if len(moves) < 2:
                            continue
                            
                        board.push(moves[0])
                        eval_before = self.evaluate_position(genome, board)
                        board.push(moves[1])
                        eval_after = self.evaluate_position(genome, board)
                        
                        if board.turn:
                            score += eval_after - eval_before
                        else:
                            score += eval_before - eval_after
                
                fitness_scores.append(score / puzzles_per_genome)
            
            max_fitness = max(fitness_scores)
            self.fitness_history.append(max_fitness)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_genome = self.population[fitness_scores.index(max_fitness)]
            
            sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), 
                                                    key=lambda pair: pair[0], reverse=True)]
            new_population = sorted_population[:2]
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(sorted_population[:10], 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            print(f"Generation {generation + 1}/{num_generations}, Best Fitness: {max_fitness:.4f}")

    def save_best_genome(self, filepath: str):
        """Save the best genome to a file"""
        if self.best_genome is not None:
            self.best_genome.eval()  # Set to evaluation mode before saving
            with open(filepath, 'wb') as f:
                pickle.dump(self.best_genome, f)

    def plot_learning_progress(self, title):
        """Plot the fitness history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('Genetic Algorithm Learning Progress for ' + title)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.savefig(f'learning_progress_for_{title}.png')
        plt.close()

    def save_fitness_history_to_csv(self, filepath: str):
        """Save the fitness history to a CSV file"""
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Generation', 'Best Fitness'])
            for generation, fitness in enumerate(self.fitness_history):
                writer.writerow([generation + 1, fitness])