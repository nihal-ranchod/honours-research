import time
import numpy as np
import pyspiel
from absl import app
from absl import flags
import chess
import chess.pgn
import io
import json
import datetime

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

# Define command-line flags
flags.DEFINE_string("game", "chess", "Name of the game.")
flags.DEFINE_string("puzzle_pgn", "PGN_Data/lichess_db_puzzle_converted.pgn", "Path to puzzle PGN file.")
flags.DEFINE_integer("num_puzzles", 50, "Number of puzzles to evaluate per bot.")
flags.DEFINE_integer("rollout_count", 3, "Number of rollouts for the random rollout evaluator.")
flags.DEFINE_float("uct_c", 2.0, "UCT exploration constant.")
flags.DEFINE_integer("max_simulations", 1000, "Maximum number of MCTS simulations.")
flags.DEFINE_integer("max_moves_per_puzzle", 20, "Maximum moves allowed to solve a puzzle.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")
flags.DEFINE_string("stockfish_path", "stockfish_5/stockfish", "Path to Stockfish engine executable.")

FLAGS = flags.FLAGS

def _opt_print(*args, **kwargs):
    if not FLAGS.quiet:
        print(*args, **kwargs)

def _init_bot(bot_type, game, player_id):
    """Initialize a bot based on the specified type."""
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
            training_data="PGN_Data/lichess_db_standard_rated_2013-01.pgn",
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
            training_data="PGN_Data/lichess_db_puzzle_converted.pgn",
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

def convert_chess_move_to_pyspiel(chess_move, board):
    """
    Convert a chess.Move to a pyspiel action number.
    """
    # Get the move in UCI format (e.g., 'e2e4')
    uci_move = chess_move.uci()
    
    # Calculate source and destination squares
    from_square = chess_move.from_square
    to_square = chess_move.to_square
    
    # Handle promotion if present
    promotion = chess_move.promotion
    promotion_offset = 0
    if promotion:
        # Map promotion pieces to numbers (1=Knight, 2=Bishop, 3=Rook, 4=Queen)
        promotion_map = {
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4
        }
        promotion_offset = promotion_map.get(promotion, 0) * 64 * 64
    
    # Calculate the final action number
    # Base action is from_square * 64 + to_square
    # Add promotion offset if there is a promotion
    action = from_square * 64 + to_square + promotion_offset
    
    return action

def convert_fen_to_pyspiel_actions(game, fen):
    """
    Convert a FEN position to a sequence of pyspiel actions.
    Returns the game state at that position.
    """
    state = game.new_initial_state()
    board = chess.Board(fen)
    
    # Get move history that led to this position
    # For puzzles, we'll just use the position as-is
    # and handle the moves from there
    
    return state

def load_puzzles(pgn_file, num_puzzles, game):
    """
    Load chess puzzles from PGN file.
    """
    puzzles = []
    with open(pgn_file) as f:
        while len(puzzles) < num_puzzles:
            game_pgn = chess.pgn.read_game(f)
            if game_pgn is None:
                break
            
            # Create a board from the FEN position
            fen = game_pgn.headers.get('FEN')
            board = chess.Board(fen)
            
            # Convert moves to pyspiel format
            moves = []
            node = game_pgn
            while node.variations:
                node = node.variation(0)
                # Convert chess.Move to pyspiel action format
                pyspiel_move = convert_chess_move_to_pyspiel(node.move, board)
                moves.append(pyspiel_move)
                board.push(node.move)
            
            puzzle = {
                'fen': fen,
                'moves': moves,
                'id': game_pgn.headers.get('PuzzleId', ''),
                'starting_color': 'black' if board.turn == chess.BLACK else 'white'
            }
            puzzles.append(puzzle)
            
            if not FLAGS.quiet:
                _opt_print(f"Loaded puzzle {puzzle['id']}")
    
    return puzzles

def evaluate_puzzle_solving(game, bot, puzzle):
    """
    Evaluate how well a bot solves a specific puzzle.
    """
    start_time = time.time()
    
    # Get the initial state with the puzzle position
    state = convert_fen_to_pyspiel_actions(game, puzzle['fen'])
    
    correct_moves = puzzle['moves']
    moves_taken = 0
    solved = False
    move_accuracy = []
    
    while moves_taken < FLAGS.max_moves_per_puzzle:
        if state.is_terminal():
            break
            
        # Get the bot's move
        action = bot.step(state)
        moves_taken += 1
        
        # Check if the move matches the correct solution
        if moves_taken <= len(correct_moves):
            move_matches = action == correct_moves[moves_taken - 1]
            move_accuracy.append(move_matches)
            
            if not move_matches:
                break
            
            if moves_taken == len(correct_moves):
                solved = True
                break
        
        state.apply_action(action)
        
        # If there are more moves in the puzzle sequence, apply the opponent's move
        if moves_taken < len(correct_moves):
            opponent_move = correct_moves[moves_taken]
            state.apply_action(opponent_move)
    
    time_taken = time.time() - start_time
    
    return {
        'solved': solved,
        'moves_taken': moves_taken,
        'correct_moves': len([x for x in move_accuracy if x]),
        'move_accuracy': move_accuracy,
        'time_taken': time_taken,
        'puzzle_id': puzzle['id'],
        'starting_color': puzzle['starting_color']
    }

def evaluate_bot_on_puzzles(game, bot_type, puzzles):
    """
    Evaluate a bot's performance on a set of puzzles.
    """
    bot = _init_bot(bot_type, game, 0)
    results = []
    
    for i, puzzle in enumerate(puzzles):
        if not FLAGS.quiet:
            _opt_print(f"\nEvaluating puzzle {i+1}/{len(puzzles)} "
                      f"(ID: {puzzle['id']}) with {bot_type}")
        
        try:
            result = evaluate_puzzle_solving(game, bot, puzzle)
            results.append(result)
            
            if not FLAGS.quiet:
                _opt_print(f"  Solved: {result['solved']}")
                _opt_print(f"  Moves taken: {result['moves_taken']}")
                _opt_print(f"  Correct moves: {result['correct_moves']}")
                _opt_print(f"  Time: {result['time_taken']:.2f}s")
                
        except Exception as e:
            _opt_print(f"Error evaluating puzzle {puzzle['id']}: {str(e)}")
            results.append({
                'puzzle_id': puzzle['id'],
                'solved': False,
                'moves_taken': 0,
                'correct_moves': 0,
                'move_accuracy': [],
                'time_taken': 0,
                'starting_color': puzzle['starting_color'],
                'error': str(e)
            })
    
    return results

def analyze_results(all_results):
    """Analyze and summarize results for all bots."""
    summary = {}
    
    for bot_type, results in all_results.items():
        puzzles_solved = sum(1 for r in results if r['solved'])
        total_correct_moves = sum(r['correct_moves'] for r in results)
        total_moves_taken = sum(r['moves_taken'] for r in results)
        avg_moves = float(np.mean([r['moves_taken'] for r in results]))
        avg_time = float(np.mean([r['time_taken'] for r in results]))
        move_accuracy = total_correct_moves / total_moves_taken if total_moves_taken > 0 else 0
        
        summary[bot_type] = {
            "total_puzzles": len(results),
            "puzzles_solved": puzzles_solved,
            "solve_rate": float(puzzles_solved / len(results)),
            "move_accuracy": float(move_accuracy),
            "avg_moves_per_puzzle": avg_moves,
            "avg_time_per_puzzle": avg_time,
            "total_correct_moves": total_correct_moves,
            "total_moves_attempted": total_moves_taken
        }
    
    return summary

def main(argv):
    game = pyspiel.load_game(FLAGS.game)
    puzzles = load_puzzles(FLAGS.puzzle_pgn, FLAGS.num_puzzles, game)
    
    all_results = {}
    for bot_type in _KNOWN_PLAYERS:
        results = evaluate_bot_on_puzzles(game, bot_type, puzzles)
        all_results[bot_type] = results
    
    summary = analyze_results(all_results)
    
    with open("puzzle_evaluation_results.json", "w") as f:
        json.dump({
            "puzzle_evaluation_summary": summary,
            "evaluation_config": {
                "num_puzzles": FLAGS.num_puzzles,
                "max_moves_per_puzzle": FLAGS.max_moves_per_puzzle,
                "timestamp": datetime.now().isoformat()
            }
        }, f, indent=2)

if __name__ == "__main__":
    app.run(main)