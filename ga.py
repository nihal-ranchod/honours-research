import chess
import random
import pickle
import os
import matplotlib.pyplot as plt

class GeneticAlgorithmBot:
    def __init__(self):
        self.population_size = 100  # Increased from 10
        self.generations = 200  # Increased from 100
        self.mutation_rate = 0.05  # Decreased from 0.1
        self.tournament_size = 5  # New parameter for tournament selection
        self.elite_size = 2  # New parameter for elitism
        self.crossover_rate = 0.8  # New parameter for crossover probability
        self.population = [self.random_strategy() for _ in range(self.population_size)]
        self.learning_progress = []  # Track learning progress
        self.fitness_history = []  # Track average fitness over generations

    def random_strategy(self):
        # Create a strategy that randomly selects legal moves from a predefined set of heuristics
        return random.choice([self.random_move, self.aggressive_move, self.defensive_move])

    def random_move(self, board):
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

    def aggressive_move(self, board):
        legal_moves = list(board.legal_moves)
        attacking_moves = [move for move in legal_moves if self.is_attacking_move(move, board)]
        return random.choice(attacking_moves) if attacking_moves else self.random_move(board)

    def defensive_move(self, board):
        legal_moves = list(board.legal_moves)
        defensive_moves = [move for move in legal_moves if self.is_defensive_move(move, board)]
        return random.choice(defensive_moves) if defensive_moves else self.random_move(board)

    def is_attacking_move(self, move, board):
        board.push(move)
        attacked_piece = board.piece_at(move.to_square)
        board.pop()
        return attacked_piece is not None and attacked_piece.color != board.turn

    def is_defensive_move(self, move, board):
        # Check if the move helps to defend against opponent's threats
        board.push(move)
        is_safe = not board.is_check()  # Check if the king is safe after the move
        board.pop()
        return is_safe

    def evaluate_fitness(self, strategy):
        num_games = 10  # Number of games to simulate
        wins = 0
        draws = 0
        losses = 0

        for _ in range(num_games):
            result = self.simulate_game(strategy)
            if result == "win":
                wins += 1
            elif result == "draw":
                draws += 1
            else:
                losses += 1

        return wins + (draws * 0.5)  # Wins are worth 1 point, draws are worth 0.5

    def simulate_game(self, strategy):
        board = chess.Board()
        move_list = []  # Keep track of moves for potential analysis
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = strategy(board)  # Use the bot's strategy to get a move
            else:
                move = self.random_move(board)  # Use a random strategy for the opponent

            if move is not None:
                board.push(move)
                move_list.append(move)  # Track moves
            else:
                break  # If no legal moves are available, the game ends

        if board.is_checkmate():
            return "win" if board.turn == chess.BLACK else "loss"
        elif board.is_stalemate():
            return "draw"
        elif board.is_insufficient_material():
            return "draw"
        elif board.is_seventyfive_moves():
            return "draw"
        else:
            return "loss"  # Default to loss if none of the above conditions are met

    def evolve_population(self):
        fitness_scores = [self.evaluate_fitness(strategy) for strategy in self.population]
        self.learning_progress.append(fitness_scores)  # Track fitness scores for this generation
        
        # Track average fitness for plotting
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        self.fitness_history.append(avg_fitness)  # Track average fitness over generations
        
        # Perform selection, crossover, and mutation to create a new population
        new_population = self.elitism_selection(fitness_scores)
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(fitness_scores)
            parent2 = self.tournament_selection(fitness_scores)
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1  # No crossover, just copy parent1
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population

    def elitism_selection(self, fitness_scores):
        elite = sorted(zip(fitness_scores, self.population), key=lambda x: x[0], reverse=True)[:self.elite_size]
        return [strategy for _, strategy in elite]

    def tournament_selection(self, fitness_scores):
        tournament = random.sample(list(zip(fitness_scores, self.population)), self.tournament_size)
        return max(tournament, key=lambda x: x[0])[1]

    def crossover(self, parent1, parent2):
        # Implement a more sophisticated crossover method
        # This is a simple example and can be improved
        return random.choice([parent1, parent2])

    def mutate(self, strategy):
        if random.random() < self.mutation_rate:
            return self.random_strategy()
        return strategy

    def save_model(self, filename='ga_chess_bot.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.population, f)
        print(f"Model saved to {filename}.")

    def load_model(self, filename='ga_chess_bot.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.population = pickle.load(f)
            print(f"Model loaded from {filename}.")
        else:
            print(f"No model found at {filename}.")

    def plot_learning_progress(self):
        generations = range(1, len(self.learning_progress) + 1)
        plt.figure(figsize=(12, 6))
        
        # Plot fitness scores for each generation
        for score in zip(*self.learning_progress):  # Unpack fitness scores for each generation
            plt.plot(generations, score, marker='o', linestyle='-', alpha=0.5)
        
        # Plot average fitness score
        plt.plot(generations, self.fitness_history, color='red', label='Average Fitness', linewidth=2, marker='x')
        
        plt.title('Learning Progress Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.ylim(0, max(max(self.learning_progress)) + 1)
        plt.grid()
        plt.legend()
        plt.savefig('learning_progress.png')  # Save the plot
        plt.show()

    def train(self, save_interval=50):
        for generation in range(self.generations):
            self.evolve_population()
            print(f"Generation {generation + 1} complete. Avg Fitness: {self.fitness_history[-1]:.2f}")
            
            if (generation + 1) % save_interval == 0:
                self.save_model(f'ga_chess_bot_gen_{generation+1}.pkl')
                self.plot_learning_progress()
                print(f"Progress saved at generation {generation + 1}")

    def inform_action(self, state, player, action):
        # This method is called to inform the bot about actions taken in the game
        # For now, we'll just pass since we're not using this information
        pass

    def step(self, state):
        print(f"Type of state: {type(state)}")
        print(f"State: {state}")
        
        # Check if state is already a chess.Board object
        if isinstance(state, chess.Board):
            board = state
        elif isinstance(state, str):
            # If state is a string, assume it's a FEN representation
            board = chess.Board(state)
        else:
            # If state is a pyspiel.ChessState object
            board = chess.Board(state.observation_string())

        # Use the best strategy to get the next move
        best_strategy = max(self.population, key=self.evaluate_fitness)
        move = best_strategy(board)

        # Ensure the move is valid
        if move not in board.legal_moves:
            # If the move is not legal, fall back to a random legal move
            move = self.random_move(board)

        return move

if __name__ == "__main__":
    bot = GeneticAlgorithmBot()
    bot.train()
    bot.plot_learning_progress()  # Plot learning progress
    bot.save_model()  # Save the model after training
