import chess.pgn
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import pyspiel

class GeneticAlgorithmBot(pyspiel.Bot):
    def __init__(self, population_size=100, generations=50, mutation_rate=0.01, elitism_rate=0.1, tournament_size=5):
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.population = self._initialize_population()
        self.fitness_history = []

    def _initialize_population(self):
        # Initialize a population of random strategies
        return [self._random_strategy() for _ in range(self.population_size)]

    def _random_strategy(self):
        # Create a random strategy (e.g., random weights for evaluation function)
        return np.random.rand(64)

    def _evaluate_fitness(self, strategy, pgn_data):
        # Evaluate the fitness of a strategy based on PGN data
        fitness = 0
        for game in pgn_data:
            board = chess.Board()
            for move in game.mainline_moves():
                board.push(move)
                if board.is_game_over():
                    result = board.result()
                    if result == "1-0":
                        fitness += 1
                    elif result == "0-1":
                        fitness -= 1
        return fitness

    def _select_parents(self):
        # Select parents using tournament selection
        parents = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, self.tournament_size)
            best_parent = max(tournament, key=lambda s: self._evaluate_fitness(s, self.pgn_data))
            parents.append(best_parent)
        return parents

    def _crossover(self, parent1, parent2):
        # Apply crossover to create a new strategy
        crossover_point = random.randint(0, len(parent1))
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def _mutate(self, strategy):
        # Apply mutation to a strategy
        for i in range(len(strategy)):
            if random.random() < self.mutation_rate:
                strategy[i] = np.random.rand()
        return strategy

    def train(self, pgn_file):
        # Load PGN data
        self.pgn_data = []
        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                self.pgn_data.append(game)

        # Train the population
        for generation in range(self.generations):
            new_population = []
            parents = self._select_parents()

            # Elitism: carry over the best strategies to the next generation
            num_elites = int(self.elitism_rate * self.population_size)
            elites = sorted(self.population, key=lambda s: self._evaluate_fitness(s, self.pgn_data), reverse=True)[:num_elites]
            new_population.extend(elites)

            # Create new candidates through crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self._mutate(self._crossover(parent1, parent2))
                new_population.append(child)

            self.population = new_population

            # Track fitness
            best_fitness = max([self._evaluate_fitness(strategy, self.pgn_data) for strategy in self.population])
            self.fitness_history.append(best_fitness)
            print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # Plot learning progress
        plt.plot(self.fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Genetic Algorithm Learning Progress')
        plt.savefig('ga_learning_progress.png')

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.population, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.population = pickle.load(f)

    def step(self, state):
        # Select the best strategy and use it to choose a move
        best_strategy = max(self.population, key=lambda s: self._evaluate_fitness(s, self.pgn_data))
        legal_moves = state.legal_actions()
        best_move = max(legal_moves, key=lambda move: self._evaluate_move(state, move, best_strategy))
        return best_move

    def _evaluate_move(self, state, move, strategy):
        # Evaluate a move based on the strategy
        board = chess.Board(state.fen())
        board.push(move)
        return sum(strategy[board.piece_at(i).square] for i in range(64) if board.piece_at(i))

    def inform_action(self, state, player_id, action):
        pass

if __name__ == "__main__":
    pgn_file = "PGN_Data/lichess_db_standard_rated_2013-01.pgn"
    ga_bot = GeneticAlgorithmBot()
    ga_bot.train(pgn_file)
    ga_bot.save_model("ga_chess_bot.pkl")