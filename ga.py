import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import chess
import chess.engine

class GeneticAlgorithmBot:
    def __init__(self, population_size=100, mutation_rate=0.1, crossover_rate=0.7, num_generations=20, engine_path="stockfish/stockfish"):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.population = self._initialize_population()
        self.fitness_history = []

    def _initialize_population(self):
        # Initialize population with random weights
        return [np.random.rand(64) for _ in range(self.population_size)]

    def _evaluate_individual(self, individual):
        # Evaluation logic using Stockfish
        board = chess.Board()
        score = 0
        for _ in range(10):  # Play up to 10 random moves to evaluate the position
            legal_moves = list(board.legal_moves)
            
            # Check if there are legal moves available
            if not legal_moves:
                break  # Exit if the game is in a terminal state
            
            move = random.choice(legal_moves)
            board.push(move)
            result = self.engine.analyse(board, chess.engine.Limit(time=0.1))
            eval_score = result["score"].relative.score()

            # Handle None scores
            if eval_score is None:
                eval_score = 0

            score += eval_score if board.turn else -eval_score
        return score

    def _selection(self):
        # Tournament selection
        tournament = random.sample(self.population, k=4)
        tournament.sort(key=self._evaluate_individual, reverse=True)
        return tournament[0], tournament[1]

    def _crossover(self, parent1, parent2):
        # Single point crossover
        crossover_point = np.random.randint(64)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def _mutate(self, individual):
        # Mutate individual with some probability
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = np.random.rand()
        return individual

    def _next_generation(self):
        new_population = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = self._selection()
            child1, child2 = self._crossover(parent1, parent2), self._crossover(parent2, parent1)
            new_population.append(self._mutate(child1))
            new_population.append(self._mutate(child2))
        self.population = new_population

    def train(self, num_games=100):
        for generation in tqdm(range(self.num_generations), desc="Training Progress"):
            fitness_scores = [self._evaluate_individual(individual) for individual in self.population]
            self.fitness_history.append(np.mean(fitness_scores))
            print(f"Generation {generation+1} - Average Fitness: {np.mean(fitness_scores)}")
            self._next_generation()

    def save_weights(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.population, f)
        print("Model weights saved successfully.")

    def load_weights(self, file_path):
        with open(file_path, 'rb') as f:
            self.population = pickle.load(f)
        print("Model weights loaded successfully.")

    def plot_learning_progress(self):
        plt.plot(self.fitness_history)
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness")
        plt.title("GA Bot Learning Progress")
        plt.show()
        
    def step(self, state):
        # Select best move based on evaluation from population individuals
        moves = list(state.legal_moves)
        move_scores = [self._evaluate_individual(np.random.choice(self.population)) for move in moves]
        best_move = moves[np.argmax(move_scores)]
        return best_move

    def __del__(self):
        if hasattr(self, 'engine'):
            self.engine.quit()
