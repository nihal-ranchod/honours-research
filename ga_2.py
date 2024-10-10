import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import chess.pgn

class GeneticAlgorithmBot:
    def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.rewards = []
        self.generation_rewards = []

    def initialize_population(self):
        """Initialize the population with random parameters."""
        self.population = [self._random_individual() for _ in range(self.population_size)]

    def _random_individual(self):
        """Create a random individual with parameters suited for chess moves."""
        return {"weights": np.random.rand(64)}

    def evaluate(self, individual, training_data):
        """Evaluate the individual's performance on training data."""
        total_reward = 0
        for puzzle in training_data:
            board = puzzle.board()
            node = puzzle

            # Initialize reward components
            move_count = 0
            won_game = False

            while node.variations:
                move_count += 1
                next_node = node.variation(0)
                board.push(next_node.move)
                
                # Evaluate based on individual's weights
                reward = np.sum([individual['weights'][square] for square in board.piece_map().keys()])

                # Reward for winning/losing state
                if board.is_checkmate():
                    won_game = True
                    reward += 100  # Example reward for checkmate
                elif board.is_stalemate():
                    reward -= 50  # Penalty for stalemate
                elif board.is_insufficient_material() or board.is_seventyfive_moves():
                    reward -= 50  # Penalty for other types of draws

                total_reward += reward
                node = next_node

            # Encourage winning in fewer moves
            if won_game:
                total_reward += 1000 / (move_count + 1)

        return total_reward

    def select_parents(self):
        """Select individuals to reproduce based on their fitness."""
        sorted_population = sorted(self.population, key=lambda x: x['reward'], reverse=True)
        return sorted_population[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        if random.random() < self.crossover_rate:
            cross_point = random.randint(1, len(parent1['weights']) - 1)
            child1_weights = np.concatenate((parent1['weights'][:cross_point], parent2['weights'][cross_point:]))
            child2_weights = np.concatenate((parent2['weights'][:cross_point], parent1['weights'][cross_point:]))
            return [{'weights': child1_weights}, {'weights': child2_weights}]
        return [parent1, parent2]

    def mutate(self, individual):
        """Mutate the individual by changing some of its weights."""
        if random.random() < self.mutation_rate:
            index = random.randint(0, len(individual['weights']) - 1)
            individual['weights'][index] = np.random.rand()
        return individual

    def train(self, training_data):
        """Train the bot using genetic algorithm on the provided training data."""
        self.initialize_population()
        
        for generation in range(100):  # Define the number of generations
            print(f"Generation {generation + 1}")
            
            # Evaluate individuals and assign rewards
            for individual in self.population:
                # Initialize reward to ensure the key exists
                individual['reward'] = 0
                try:
                    individual['reward'] = self.evaluate(individual, training_data)
                except Exception as e:
                    print(f"Error evaluating individual: {e}")
                    individual['reward'] = 0  # Fallback in case of evaluation error
            
            parents = self.select_parents()
            self.population = []
            
            # Generate new population
            while len(self.population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                children = self.crossover(parent1, parent2)
                self.population.extend(children)
            
            # Mutate the new population
            self.population = [self.mutate(ind) for ind in self.population]
            
            # Compute the average reward for the generation
            average_reward = np.mean([ind['reward'] for ind in self.population if 'reward' in ind])
            self.generation_rewards.append(average_reward)
            print(f"Average reward: {average_reward}")

    def plot_learning_progress(self):
        plt.plot(self.generation_rewards)
        plt.xlabel('Generation')
        plt.ylabel('Average Reward')
        plt.title('Learning Progress of Genetic Algorithm Bot on Standard PGN Data')
        plt.show()

    def save_model(self, filename):
        """Save the trained model to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.population, f)

    def load_model(self, filename):
        """Load a model from a file."""
        with open(filename, 'rb') as f:
            self.population = pickle.load(f)

    def step(self, state):
        """Decide the next move based on the current population leader."""
        best_individual = max(self.population, key=lambda x: x['reward'])
        legal_moves = state.legal_actions()
        # Select a move based on weights - this is a placeholder and should be customized
        return random.choice(legal_moves)
