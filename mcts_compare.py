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
flags.DEFINE_integer("num_games", 10, "Number of games to play between each pair of bots.")
flags.DEFINE_integer("rollout_count", 10, "Number of rollouts for the random rollout evaluator.")
flags.DEFINE_float("uct_c", 2.0, "UCT exploration constant.")
flags.DEFINE_integer("max_simulations", 100, "Maximum number of MCTS simulations.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS

def _opt_print(*args, **kwargs):
    if not FLAGS.quiet:
        print(*args, **kwargs)

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
        pgn_file = "master_games.pgn"
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

def evaluate_bots(game, bot1, bot2, num_games=10):
    results = {"bot1_wins": 0, "bot2_wins": 0, "draws": 0, "total_moves": 0, "total_time": 0}
    
    for i in range(num_games):
        state = game.new_initial_state()
        start_time = time.time()
        while not state.is_terminal():
            if state.current_player() == pyspiel.PlayerId.CHANCE:
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
            elif state.current_player() == 0:
                action = bot1.step(state)
            else:
                action = bot2.step(state)
            state.apply_action(action)
        
        end_time = time.time()
        game_time = end_time - start_time
        results["total_time"] += game_time
        results["total_moves"] += state.move_number()
        
        returns = state.returns()
        if returns[0] > returns[1]:
            results["bot1_wins"] += 1
        elif returns[0] < returns[1]:
            results["bot2_wins"] += 1
        else:
            results["draws"] += 1
    
    return results

def main(argv):
    game = pyspiel.load_game(FLAGS.game)
    
    bot_types = ["random", "mcts", "mcts_trained"]
    results = {}
    
    for bot1_type in bot_types:
        for bot2_type in bot_types:
            if bot1_type != bot2_type:
                bot1 = _init_bot(bot1_type, game, 0)
                bot2 = _init_bot(bot2_type, game, 1)
                result = evaluate_bots(game, bot1, bot2, FLAGS.num_games)
                results[f"{bot1_type}_vs_{bot2_type}"] = result
                _opt_print(f"Results for {bot1_type} vs {bot2_type}: {result}")
    
    # Save results to a file for further analysis
    with open("evaluation_results.txt", "w") as f:
        for match, result in results.items():
            f.write(f"{match}: {result}\n")
    
    # Print summary
    for match, result in results.items():
        _opt_print(f"{match}: {result}")

if __name__ == "__main__":
    app.run(main)