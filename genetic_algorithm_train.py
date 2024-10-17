import os
from genetic_algorithm import GeneticChessBot

def train_standard_bot():
    bot = GeneticChessBot(population_size=50)
    bot.train_on_pgn(
        pgn_file="PGN_Data/lichess_db_standard_rated_2013-01.pgn",
        num_generations=1000,
        games_per_genome=100
    )
    bot.save_best_genome("best_genome.pkl")
    bot.plot_learning_progress("Standard PGN Data")
    bot.save_fitness_history_to_csv("ga_fitness_history_standard_pgn.csv")

def train_puzzle_bot():
    bot = GeneticChessBot(population_size=50)
    bot.train_on_puzzles(
        puzzle_pgn_file="PGN_Data/lichess_db_puzzle_converted.pgn",
        num_generations=1000,
        puzzles_per_genome=100
    )
    bot.save_best_genome("best_genome_puzzle.pkl")
    bot.plot_learning_progress("Puzzle PGN Data")
    bot.save_fitness_history_to_csv("ga_fitness_history_puzzle_pgn.csv")

if __name__ == "__main__":
    print("Training standard bot...")
    train_standard_bot()
    
    print("\nTraining puzzle bot...")
    train_puzzle_bot()