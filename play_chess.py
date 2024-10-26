""" Play File

    Python file to simulate a single Chess game between any two bots.
    
    Change Bot Player in the defined Flags:
      player 1 - play as White
      player 2 - play as Black
"""

import collections
import random
import sys
import chess
from absl import app
from absl import flags
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

# Add created Agents
import mcts_algorithm as mcts
from nfsp_algorithm import NFSPBot
from baseline import StockfishBot

_KNOWN_PLAYERS = [
    # A vanilla Monte Carlo Tree Search agent.
    "mcts",

    # A generic random agent.
    "random",

    # You'll be asked to provide the moves.
    "human",

    # A MCTS agent trained on past PGN data
    "mcts_trained_pgn",

    # A MCTS agent trained on past chess puzzle data
    "mcts_trained_puzzle",

    # A Neural Fictitious Self-Play agent
    "nfsp",

    # A Genetic Algorithm agent
    "ga",

    # Baseline Stockfish agent
    "stockfish"
]

flags.DEFINE_string("game", "chess", "Name of the game.")
flags.DEFINE_enum("player1", "ga", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "random", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_string("gtp_path", None, "Where to find a binary for gtp.")
flags.DEFINE_multi_string("gtp_cmd", [], "GTP commands to run at init.")
flags.DEFINE_string("az_path", None, "Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 2, "How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 1000, "How many simulations to run.")
flags.DEFINE_integer("num_games", 1, "How many games to play.")
flags.DEFINE_integer("seed", None, "Seed for the random number generator.")
flags.DEFINE_bool("random_first", False, "Play the first move randomly.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")
flags.DEFINE_string("stockfish_path", "stockfish_5/stockfish", "Path to Stockfish engine executable.")

FLAGS = flags.FLAGS

# print messages if the quiet flag is not set
def _opt_print(*args, **kwargs):
  if not FLAGS.quiet:
    print(*args, **kwargs)

def _init_bot(bot_type, game, player_id):
  """Initializes a bot by type."""
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
  if bot_type == "stockfish":
    return StockfishBot(player_id, FLAGS.stockfish_path)
  if bot_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  if bot_type == "human":
    return human.HumanBot()
  raise ValueError("Invalid bot type: %s" % bot_type)

# Converts action string to action object using the game's legal actions
def _get_action(state, action_str):
  for action in state.legal_actions():
    if action_str == state.action_to_string(state.current_player(), action):
      return action
  return None

def _play_game(game, bots, initial_actions):
    """Plays one game."""
    state = game.new_initial_state()
    _opt_print("Initial state:\n{}".format(state))

    history = []
    # If the 'random_first' flag is set, a random initial action is chosen
    if FLAGS.random_first:
        assert not initial_actions
        initial_actions = [state.action_to_string(
            state.current_player(), random.choice(state.legal_actions()))]

    for action_str in initial_actions:
        action = _get_action(state, action_str)
        if action is None:
            sys.exit("Invalid action: {}".format(action_str))

        history.append(action_str)
        for bot in bots:
            bot.inform_action(state, state.current_player(), action)
        state.apply_action(action)
        _opt_print("Forced action", action_str)
        _opt_print("Next state:\n{}".format(state))
        # _opt_print(chess.Board(fen=str(state)))

    while not state.is_terminal():
        current_player = state.current_player()
        bot = bots[current_player]
        
        action = bot.step(state)
        if action is None or action not in state.legal_actions():
            print(f"Bot {current_player} couldn't find a valid move. Choosing randomly.")
            action = random.choice(state.legal_actions())

        action_str = state.action_to_string(current_player, action)
        _opt_print("Player {} sampled action: {}".format(current_player, action_str))

        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
        history.append(action_str)
        state.apply_action(action)

        _opt_print("Next state:\n{}".format(state))
        # _opt_print(chess.Board(fen=str(state)))

    # Game is now done. Print return for each player
    returns = state.returns()
    print("Returns:", " ".join(map(str, returns)), ", Game actions:",
          " ".join(history))

    for bot in bots:
        bot.restart()

    return returns, history


def main(argv):
  game = pyspiel.load_game(FLAGS.game)
  if game.num_players() > 2:
    sys.exit("This game requires more players than the example can handle.")
  bots = [
      _init_bot(FLAGS.player1, game, 0),
      _init_bot(FLAGS.player2, game, 1),
  ]
  histories = collections.defaultdict(int)
  overall_returns = [0, 0]
  overall_wins = [0, 0]
  game_num = 0
  try:
    for game_num in range(FLAGS.num_games):
      returns, history = _play_game(game, bots, argv[1:])
      histories[" ".join(history)] += 1
      for i, v in enumerate(returns):
        overall_returns[i] += v
        if v > 0:
          overall_wins[i] += 1
  except (KeyboardInterrupt, EOFError):
    game_num -= 1
    print("Caught a KeyboardInterrupt, stopping early.")

  print("Number of games played:", game_num + 1)
  print("Number of distinct games played:", len(histories))
  print("Players:", FLAGS.player1, FLAGS.player2)
  print("Overall wins", overall_wins)
  print("Overall returns", overall_returns)


if __name__ == "__main__":
  app.run(main)