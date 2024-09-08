import collections
import random
import sys
import time
import chess
import numpy as np
import pyspiel
from absl import app
from absl import flags

from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random

import mcts_algorithm as mcts

_KNOWN_PLAYERS = [
    "mcts",
    "random",
    "human",
    "mcts_trained",
]

flags.DEFINE_string("game", "chess", "Name of the game.")
flags.DEFINE_enum("player1", "mcts", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "random", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_integer("num_games", 30, "Number of games to play between each pair of bots.")
flags.DEFINE_integer("rollout_count", 10, "Number of rollouts for the random rollout evaluator.")
flags.DEFINE_float("uct_c", 2.0, "UCT exploration constant.")
flags.DEFINE_integer("max_simulations", 100, "Maximum number of MCTS simulations.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")
flags.DEFINE_float("elo_k_factor", 32, "K-factor for Elo rating system.")

FLAGS = flags.FLAGS

# Initialize Elo ratings
elo_ratings = {"bot1": 1500, "bot2": 1500}

def _opt_print(*args, **kwargs):
    if not FLAGS.quiet:
        print(*args, **kwargs)

def _elo_update(player_rating, opponent_rating, result, k=None):
    if k is None:
        k = FLAGS.elo_k_factor  # Access the flag after it has been parsed
    expected_score = 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))
    new_rating = player_rating + k * (result - expected_score)
    return new_rating

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
    if bot_type == "mcts_trained":
        pgn_file = "training_data.pgn"
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTSWithTraining(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            pgn_file,
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    raise ValueError(f"Unknown bot type: {bot_type}")

def evaluate_bots(game, bot1, bot2, num_games):
    results = {
        "bot1_wins": 0, 
        "bot2_wins": 0, 
        "draws": 0, 
        "total_moves": 0, 
        "total_time": 0, 
        "move_times": []
    }
    
    for i in range(num_games):
        state = game.new_initial_state()
        start_time = time.time()
        move_times = []
        
        while not state.is_terminal():
            move_start = time.time()  # Start move time tracking
            if state.current_player() == pyspiel.PlayerId.CHANCE:
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
            elif state.current_player() == 0:
                action = bot1.step(state)
                # Track time taken for the move
                move_time = time.time() - move_start
                move_times.append(move_time)
            else:
                action = bot2.step(state)
                # Track time taken for the move
                move_time = time.time() - move_start
                move_times.append(move_time)
            state.apply_action(action)
        
        end_time = time.time()
        game_time = end_time - start_time
        results["total_time"] += game_time
        results["total_moves"] += state.move_number()
        results["move_times"].append(move_times)
        
        returns = state.returns()
        if returns[0] > returns[1]:
            results["bot1_wins"] += 1
            # Update Elo ratings for bot1 win
            elo_ratings["bot1"] = _elo_update(elo_ratings["bot1"], elo_ratings["bot2"], 1)
            elo_ratings["bot2"] = _elo_update(elo_ratings["bot2"], elo_ratings["bot1"], 0)
        elif returns[0] < returns[1]:
            results["bot2_wins"] += 1
            # Update Elo ratings for bot2 win
            elo_ratings["bot1"] = _elo_update(elo_ratings["bot1"], elo_ratings["bot2"], 0)
            elo_ratings["bot2"] = _elo_update(elo_ratings["bot2"], elo_ratings["bot1"], 1)
        else:
            results["draws"] += 1
            # Update Elo ratings for a draw
            elo_ratings["bot1"] = _elo_update(elo_ratings["bot1"], elo_ratings["bot2"], 0.5)
            elo_ratings["bot2"] = _elo_update(elo_ratings["bot2"], elo_ratings["bot1"], 0.5)

    return results

def main(argv):
    # Ensure flags are parsed before calling the _elo_update function
    game = pyspiel.load_game(FLAGS.game)
    
    bot_types = ["random", "mcts", "mcts_trained"]
    results = {}
    
    for bot1_type in bot_types:
        for bot2_type in bot_types:
            if bot1_type != bot2_type:
                bot1 = _init_bot(bot1_type, game, 0)
                bot2 = _init_bot(bot2_type, game, 1)
                match_results = evaluate_bots(game, bot1, bot2, FLAGS.num_games)
                match_results["final_elo_bot1"] = elo_ratings["bot1"]
                match_results["final_elo_bot2"] = elo_ratings["bot2"]
                results[f"{bot1_type}_vs_{bot2_type}"] = match_results
                _opt_print(f"Results for {bot1_type} vs {bot2_type}: {match_results}")
    
    # Save results and Elo ratings
    with open("evaluation_results2.txt", "w") as f:
        for match, result in results.items():
            f.write(f"{match}:\n")
            for key, value in result.items():
                f.write(f"  {key}: {value}\n")
    
    # Print summary including Elo ratings
    _opt_print("\nFinal Elo Ratings:")
    _opt_print(f"Bot1 Elo Rating: {elo_ratings['bot1']}")
    _opt_print(f"Bot2 Elo Rating: {elo_ratings['bot2']}")

if __name__ == "__main__":
    app.run(main)
