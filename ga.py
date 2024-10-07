import random
import chess
import chess.engine
import numpy as np
import matplotlib.pyplot as plt
import pickle

class GeneticAlgorithmBot:
    def __init__(self, population_size=50, generations=500, mutation_rate=0.1, crossover_rate=0.7):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.fitness_scores = []
        
        # Initialize Stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish")
        self.learning_progress = []

    def evaluate_board(self, board):
        # Use Stockfish to evaluate the board state
        result = self.engine.analyse(board, chess.engine.Limit(time=0.1))
        return result["score"].relative.score() or 0  # Return 0 for None values

    def initialize_population(self):
        for _ in range(self.population_size):
            board = chess.Board()
            moves = []
            for _ in range(5):  # Generate a sequence of 5 legal moves
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = random.choice(legal_moves)
                moves.append(move)
                board.push(move)
            self.population.append(moves)

    def fitness(self, individual):
        board = chess.Board()
        for move in individual:
            if board.is_legal(move):
                board.push(move)
            else:
                return 0  # Return 0 instead of a large negative value
        return self.evaluate_board(board)

    def selection(self):
        # Weighted random selection based on fitness
        weights = [self.fitness(individual) for individual in self.population]
        selected = random.choices(self.population, weights=weights, k=2)
        return selected

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index = random.randint(0, len(individual) - 1)
            legal_moves = list(chess.Board().legal_moves)
            individual[index] = random.choice(legal_moves)
        return individual

    def train(self):
        self.initialize_population()
        
        for generation in range(self.generations):
            new_population = []
            self.fitness_scores = [self.fitness(individual) for individual in self.population]
            self.learning_progress.append(np.mean(self.fitness_scores))
            
            while len(new_population) < self.population_size:
                parent1, parent2 = self.selection()
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            
            self.population = new_population
            print(f"Generation {generation} - Avg Fitness: {self.learning_progress[-1]}")
        
        self.save_model("genetic_algorithm_model.pkl")
        self.plot_learning_progress()

    def plot_learning_progress(self):
        plt.plot(self.learning_progress)
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness Score')
        plt.title('Genetic Algorithm Learning Progress')
        plt.savefig('learning_progress.png')
        plt.show()

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.population, f)
    
    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.population = pickle.load(f)

    def make_move(self, board):
        # Find the best move according to the bot
        best_move = None
        best_fitness = -float("inf")
        for move in self.population[0]:  # Evaluate moves in the best individual
            board.push(move)
            fitness = self.evaluate_board(board)
            if fitness > best_fitness:
                best_fitness = fitness
                best_move = move
            board.pop()
        return best_move

# Usage example
bot = GeneticAlgorithmBot()
bot.train()
# To load a saved model and play a move
# bot.load_model("genetic_algorithm_model.pkl")
# move = bot.make_move(board)
