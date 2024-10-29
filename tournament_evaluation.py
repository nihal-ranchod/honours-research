import time
import numpy as np
import pyspiel
from absl import app
from absl import flags
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

from open_spiel.python.bots import uniform_random
import mcts_algorithm as mcts
from nfsp_algorithm import NFSPBot
from baseline import StockfishBot
from genetic_algorithm import LoadedChessModel

_KNOWN_PLAYERS = [
    "mcts",
    "random",
    "mcts_trained_pgn",
    "mcts_trained_puzzle",
    "nfsp",
    "stockfish",
    "ga",
    "ga_puzzle"
]

flags.DEFINE_string("game", "chess", "Name of the game.")
flags.DEFINE_integer("num_games", 50, "Number of games to play between each pair of bots (must be even).")
flags.DEFINE_integer("rollout_count", 3, "Number of rollouts for the random rollout evaluator.")
flags.DEFINE_float("uct_c", 2.0, "UCT exploration constant.")
flags.DEFINE_integer("max_simulations", 1000, "Maximum number of MCTS simulations.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("verbose", False, "Show detailed stats for each game.")
flags.DEFINE_float("elo_k_factor", 32, "K-factor for Elo rating system.")
flags.DEFINE_string("stockfish_path", "stockfish/stockfish", "Path to Stockfish engine executable.")
flags.DEFINE_string("output_dir", "tournament_results", "Directory to store results.")

FLAGS = flags.FLAGS

def _init_bot(bot_type, game, player_id):
    rng = np.random.RandomState(FLAGS.seed)
    if bot_type == "mcts":
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTSBot(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
    if bot_type == "mcts_trained_pgn":
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTS_with_PGN_Data(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            training_data = "PGN_Data/lichess_db_standard_rated_2013-01.pgn",
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
    if bot_type == "mcts_trained_puzzle":
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTS_with_PGN_Data(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            training_data = "PGN_Data/lichess_db_puzzle_converted.pgn",
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
    if bot_type == "nfsp":
        return NFSPBot(game, player_id, "aggressive_nfsp_model_final.pth")
    if bot_type == "ga":
        return LoadedChessModel("best_genome.pkl")
    if bot_type == "ga_puzzle":
        return LoadedChessModel("best_genome_puzzle.pkl")
    if bot_type == "stockfish":
        return StockfishBot(player_id, FLAGS.stockfish_path)
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    raise ValueError(f"Unknown bot type: {bot_type}")


class TournamentManager:
    def __init__(self):
        self.elo_ratings = defaultdict(lambda: 1500)
        self.results = defaultdict(lambda: {
            "total_games": 0,
            "wins_as_white": 0,
            "wins_as_black": 0,
            "draws": 0,
            "elo_rating": 1500,
            "average_game_length": 0,
            "total_moves": 0
        })
        
    def _update_elo(self, white_bot, black_bot, result):
        """Update Elo ratings after a game."""
        if result == 1:  # White wins
            score_white, score_black = 1, 0
        elif result == -1:  # Black wins
            score_white, score_black = 0, 1
        else:  # Draw
            score_white, score_black = 0.5, 0.5
            
        # Calculate expected scores
        rating_diff = self.elo_ratings[white_bot] - self.elo_ratings[black_bot]
        expected_white = 1 / (1 + 10 ** (-rating_diff / 400))
        expected_black = 1 - expected_white
        
        # Update ratings
        self.elo_ratings[white_bot] += FLAGS.elo_k_factor * (score_white - expected_white)
        self.elo_ratings[black_bot] += FLAGS.elo_k_factor * (score_black - expected_black)

    def play_game(self, game, white_bot, black_bot):
        """Play a single game between two bots."""
        state = game.new_initial_state()
        moves = 0
        start_time = time.time()
        
        try:
            while not state.is_terminal():
                if state.current_player() == pyspiel.PlayerId.CHANCE:
                    outcomes = state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes)
                    action = np.random.choice(action_list, p=prob_list)
                else:
                    current_bot = white_bot if state.current_player() == 0 else black_bot
                    action = current_bot.step(state)
                state.apply_action(action)
                moves += 1
                
            game_time = time.time() - start_time
            returns = state.returns()
            
            # Determine result
            if returns[0] > returns[1]:
                result = 1  # White wins
            elif returns[0] < returns[1]:
                result = -1  # Black wins
            else:
                result = 0  # Draw
                
            return result, moves, game_time
            
        except Exception as e:
            return None, moves, time.time() - start_time

    def run_tournament(self):
        game = pyspiel.load_game(FLAGS.game)
        
        if FLAGS.num_games % 2 != 0:
            raise ValueError("Number of games must be even to ensure equal colors for both players")
        
        games_per_pair = FLAGS.num_games // 2
        
        for bot1_type in _KNOWN_PLAYERS:
            for bot2_type in _KNOWN_PLAYERS:
                if bot1_type >= bot2_type:
                    continue
                    
                try:
                    bot1 = _init_bot(bot1_type, game, 0)
                    bot2 = _init_bot(bot2_type, game, 1)
                except Exception:
                    continue
                
                for _ in range(games_per_pair):
                    result, moves, _ = self.play_game(game, bot1, bot2)
                    if result is not None:
                        self._process_game_result(bot1_type, bot2_type, result, moves)
                    
                    result, moves, _ = self.play_game(game, bot2, bot1)
                    if result is not None:
                        self._process_game_result(bot2_type, bot1_type, result, moves)

    def _process_game_result(self, white_bot, black_bot, result, moves):
        for bot in (white_bot, black_bot):
            self.results[bot]["total_games"] += 1
            self.results[bot]["total_moves"] += moves
            self.results[bot]["average_game_length"] = (
                self.results[bot]["total_moves"] / self.results[bot]["total_games"]
            )
        
        if result == 1:
            self.results[white_bot]["wins_as_white"] += 1
        elif result == -1:
            self.results[black_bot]["wins_as_black"] += 1
        else:
            self.results[white_bot]["draws"] += 1
            self.results[black_bot]["draws"] += 1
        
        self._update_elo(white_bot, black_bot, result)
        self.results[white_bot]["elo_rating"] = self.elo_ratings[white_bot]
        self.results[black_bot]["elo_rating"] = self.elo_ratings[black_bot]

    def save_results(self):
        output_dir = Path(FLAGS.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "final_results.json", "w") as f:
            json.dump({
                "bot_results": self.results,
                "tournament_config": {
                    "num_games": FLAGS.num_games,
                    "timestamp": datetime.now().isoformat()
                }
            }, f, indent=2)

def main(argv):
    if FLAGS.num_games <= 0 or FLAGS.num_games % 2 != 0:
        raise ValueError("num_games must be a positive even number")
    
    np.random.seed(FLAGS.seed)
    tournament = TournamentManager()
    tournament.run_tournament()
    tournament.save_results()

if __name__ == "__main__":
    app.run(main)