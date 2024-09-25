import random
import chess
import chess.engine
import os

class GeneticAlgorithmBot:
    def __init__(self, population_size=20, generations=5, mutation_rate=0.1, engine_path='stockfish/stockfish'):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Stockfish engine not found at {engine_path}")
        if not os.access(engine_path, os.X_OK):
            raise PermissionError(f"Stockfish engine at {engine_path} is not executable")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.engine_path = engine_path
        self.population = self.initialize_population()

    def initialize_population(self):
        num_actions = chess.Board().legal_moves.count()
        return [self.random_strategy(num_actions) for _ in range(self.population_size)]

    def random_strategy(self, num_actions):
        return [random.random() for _ in range(num_actions)]

    def fitness(self, strategy):
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        board = chess.Board()
        score = 0
        for move in strategy:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)
            # Evaluate the board position
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            score += info["score"].relative.score(mate_score=10000)  # Use a high value for mate score
        engine.quit()
        return score

    def select_parents(self):
        self.population.sort(key=self.fitness, reverse=True)
        return self.population[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, len(parent1))
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(self, strategy):
        for i in range(len(strategy)):
            if random.random() < self.mutation_rate:
                strategy[i] = random.random()
        return strategy

    def evolve(self):
        for generation in range(self.generations):
            parents = self.select_parents()
            offspring = []
            for i in range(self.population_size - len(parents)):
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                offspring.append(child)
            self.population = parents + offspring
            print(f"Generation {generation}: Best fitness = {self.fitness(self.population[0])}")

    def step(self, state):
        # Evolve the population before choosing an action
        self.evolve()
        best_strategy = max(self.population, key=self.fitness)
        legal_actions = list(state.legal_actions())
        action_probabilities = [best_strategy[action] for action in range(len(legal_actions))]
        total_prob = sum(action_probabilities)
        if total_prob <= 0:
            return random.choice(legal_actions)
        action_probabilities = [prob / total_prob for prob in action_probabilities]
        action = random.choices(legal_actions, weights=action_probabilities, k=1)[0]
        return action

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions."""
        action_string = state.action_to_string(player_id, action)
        print(f"Player {player_id} took action: {action_string}")

# if __name__ == "__main__":
#     ga_bot = GeneticAlgorithmBot()
#     ga_bot.evolve()