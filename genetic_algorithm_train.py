import os
from genetic_algorithm import GeneticChessBot

def train_standard_bot():
    bot = GeneticChessBot(population_size=50)
    bot.train_on_pgn(
        pgn_file="PGN_Data/lichess_db_standard_rated_2013-01.pgn",
        num_generations=100,
        games_per_genome=50
    )
    bot.save_best_genome("best_genome.pkl")
    bot.plot_learning_progress()

def train_puzzle_bot():
    bot = GeneticChessBot(population_size=50)
    bot.train_on_puzzles(
        puzzle_pgn_file="PGN_Data/lichess_db_puzzle_converted.pgn",
        num_generations=100,
        puzzles_per_genome=100
    )
    bot.save_best_genome("best_genome_puzzle.pkl")
    bot.plot_learning_progress()

if __name__ == "__main__":
    print("Training standard bot...")
    train_standard_bot()
    
    print("\nTraining puzzle bot...")
    train_puzzle_bot()