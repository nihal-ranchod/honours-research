import random
import numpy as np
import chess.pgn
import pyspiel
from concurrent.futures import ThreadPoolExecutor

class GeneticAlgorithmBot(pyspiel.Bot):
    """Bot that uses a Genetic Algorithm to play chess."""

    def __init__(self, game, population_size, mutation_rate, generations, evaluator, random_state=None, verbose=False):
        pyspiel.Bot.__init__(self)
        self._game = game
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.evaluator = evaluator
        self.verbose = verbose
        self._random_state = random_state or np.random.RandomState()
        self.evaluation_cache = {}

    def restart_at(self, state):
        pass

    def step_with_policy(self, state):
        """Returns bot's policy and action at given state."""
        best_move = self.genetic_algorithm_search(state)
        policy = [(action, (1.0 if action == best_move else 0.0))
                  for action in state.legal_actions(state.current_player())]
        return policy, best_move

    def step(self, state):
        return self.step_with_policy(state)[1]

    def genetic_algorithm_search(self, state):
        """Performs a Genetic Algorithm search to find the best move."""
        population = self._initialize_population(state)
        best_individual = None
        best_score = float('-inf')

        for generation in range(self.generations):
            scores = self._evaluate_population(state, population)
            best_gen_score = max(scores)
            best_gen_individual = population[np.argmax(scores)]

            if best_gen_score > best_score:
                best_score = best_gen_score
                best_individual = best_gen_individual

            selected = self._selection(population, scores)
            offspring = self._crossover(selected)
            population = self._mutate(offspring, state)

            if self.verbose:
                print(f"Generation {generation}: Best score {best_score}")

        return best_individual

    def _initialize_population(self, state):
        """Initializes a population of random moves."""
        legal_actions = state.legal_actions(state.current_player())
        return [self._random_state.choice(legal_actions) for _ in range(self.population_size)]

    def _evaluate_population(self, state, population):
        """Evaluates the population of moves using parallel processing."""
        with ThreadPoolExecutor() as executor:
            scores = list(executor.map(lambda move: self._evaluate_move(state, move), population))
        return scores

    def _evaluate_move(self, state, move):
        """Evaluates a single move with memoization."""
        if move in self.evaluation_cache:
            return self.evaluation_cache[move]

        cloned_state = state.clone()
        cloned_state.apply_action(move)
        evaluation = self.evaluator.evaluate(cloned_state)

        if isinstance(evaluation, np.ndarray):
            evaluation = np.mean(evaluation)

        self.evaluation_cache[move] = evaluation
        return evaluation

    def _selection(self, population, scores):
        """Selects the best moves using tournament selection."""
        selected = []
        for _ in range(self.population_size):
            indices = self._random_state.choice(len(population), 3)
            tournament = [(population[i], scores[i]) for i in indices]
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def _crossover(self, selected):
        """Performs crossover to generate new moves."""
        offspring = []
        for _ in range(self.population_size):
            parent1, parent2 = self._random_state.choice(selected, 2)
            offspring.append(self._crossover_moves(parent1, parent2))
        return offspring

    def _crossover_moves(self, move1, move2):
        """Combines two moves to create a new move."""
        return self._random_state.choice([move1, move2])

    def _mutate(self, offspring, state):
        """Mutates the offspring to maintain diversity with adaptive mutation rate."""
        legal_actions = state.legal_actions(state.current_player())
        diversity = len(set(offspring)) / len(offspring)
        adaptive_mutation_rate = self.mutation_rate * (1 - diversity)

        for i in range(len(offspring)):
            if self._random_state.rand() < adaptive_mutation_rate:
                offspring[i] = self._random_state.choice(legal_actions)
        return offspring