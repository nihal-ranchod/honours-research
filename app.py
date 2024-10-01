from flask import Flask, render_template, request, jsonify, Response
import pyspiel
import numpy as np
import random
import chess
import logging
import json

# Import necessary modules for MCTS
import mcts_algorithm as mcts
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "seed": None,
    "rollout_count": 1,
    "max_simulations": 1000,
    "uct_c": 2,
    "solve": True,
    "verbose": False
}

_KNOWN_PLAYERS = [
    "mcts",
    "random",
    "human",
    "mcts_trained_pgn",
    "mcts_trained_puzzle"
]

def _init_bot(bot_type, game, player_id):
    """Initializes a bot by type."""
    rng = np.random.RandomState(CONFIG["seed"])
    if bot_type == "mcts":
        evaluator = mcts.RandomRolloutEvaluator(CONFIG["rollout_count"], rng)
        return mcts.MCTSBot(
            game,
            CONFIG["uct_c"],
            CONFIG["max_simulations"],
            evaluator,
            random_state=rng,
            solve=CONFIG["solve"],
            verbose=CONFIG["verbose"])
    if bot_type == "mcts_trained_pgn":
        evaluator = mcts.RandomRolloutEvaluator(CONFIG["rollout_count"], rng)
        return mcts.MCTSWithTraining(
            game,
            CONFIG["uct_c"],
            CONFIG["max_simulations"],
            evaluator,
            training_data="PGN_Data/lichess_db_standard_rated_2013-01.pgn",
            random_state=rng,
            solve=CONFIG["solve"],
            verbose=CONFIG["verbose"])
    if bot_type == "mcts_trained_puzzle":
        evaluator = mcts.RandomRolloutEvaluator(CONFIG["rollout_count"], rng)
        return mcts.MCTSWithTraining(
            game,
            CONFIG["uct_c"],
            CONFIG["max_simulations"],
            evaluator,
            training_data="PGN_Data/lichess_db_puzzle_converted.pgn",
            random_state=rng,
            solve=CONFIG["solve"],
            verbose=CONFIG["verbose"])
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    if bot_type == "human":
        return human.HumanBot()
    raise ValueError("Invalid bot type: %s" % bot_type)

def openspiel_move_to_uci(move_string, board):
    """Convert OpenSpiel move string to UCI format."""
    # OpenSpiel seems to be providing moves in a simple format (e.g., "e4", "Nf3")
    # We need to find the corresponding UCI move on the current board
    for move in board.legal_moves:
        if board.san(move) == move_string or move.uci().startswith(move_string):
            return move.uci()
    return None

@app.route('/')
def index():
    return render_template('index.html', bots=_KNOWN_PLAYERS)

@app.route('/play', methods=['POST'])
def play():
    player1 = request.form['player1']
    player2 = request.form['player2']
    
    logger.info(f"Starting game: {player1} vs {player2}")
    
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()
    
    bot1 = _init_bot(player1, game, 0)
    bot2 = _init_bot(player2, game, 1)
    
    board = chess.Board()

    def generate_moves():
        while not state.is_terminal():
            if state.current_player() == 0:
                bot = bot1
            else:
                bot = bot2
            
            try:
                action = bot.step(state)
                move = state.action_to_string(state.current_player(), action)
                
                logger.debug(f"Raw move from bot: {move}")
                
                uci_move = openspiel_move_to_uci(move, board)
                if uci_move is None:
                    logger.error(f"Failed to convert move: {move}")
                    state.apply_action(action)
                    logger.debug(f"Applied action directly: {action}")
                    continue
                
                logger.debug(f"UCI move: {uci_move}")
                
                chess_move = chess.Move.from_uci(uci_move)
                if chess_move in board.legal_moves:
                    board.push(chess_move)
                    state.apply_action(action)
                    logger.debug(f"Move applied successfully")
                    yield json.dumps({"move": uci_move, "fen": board.fen()}) + "\n"
                else:
                    logger.error(f"Illegal move: {uci_move}")
                    state.apply_action(action)
                    logger.debug(f"Applied action directly: {action}")
            except Exception as e:
                logger.error(f"Error processing move: {str(e)}")
                logger.error(f"Current state: {state}")
                logger.error(f"Current board: {board.fen()}")
                break
        
        returns = state.returns()
        winner = "Draw"
        if returns[0] > returns[1]:
            winner = player1
        elif returns[1] > returns[0]:
            winner = player2
        
        logger.info(f"Game ended. Winner: {winner}")
        yield json.dumps({"game_over": True, "winner": winner, "final_fen": board.fen()}) + "\n"

    return Response(generate_moves(), content_type='application/json')

@app.route('/human_move', methods=['POST'])
def human_move():
    fen = request.form['fen']
    move = request.form['move']
    
    logger.debug(f"Human move: {move}, FEN: {fen}")
    
    board = chess.Board(fen)
    try:
        chess_move = chess.Move.from_uci(move)
        if chess_move in board.legal_moves:
            board.push(chess_move)
            logger.debug("Human move applied successfully")
            return jsonify({
                'success': True,
                'fen': board.fen(),
                'game_over': board.is_game_over(),
                'winner': get_winner(board)
            })
        else:
            logger.error(f"Illegal move: {move}")
            return jsonify({'success': False, 'error': 'Illegal move'})
    except ValueError:
        logger.error(f"Invalid move format: {move}")
        return jsonify({'success': False, 'error': 'Invalid move format'})

def get_winner(board):
    if board.is_checkmate():
        return 'white' if board.turn == chess.BLACK else 'black'
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
        return 'draw'
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)