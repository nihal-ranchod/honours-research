import chess
import chess.engine
from open_spiel.python.observation import make_observation
import pyspiel

class StockfishBot:
    """A bot that uses the Stockfish chess engine to make moves."""
    
    def __init__(self, player_id, stockfish_path, time_limit=0.01):
        """Initialize the bot.
        
        Args:
            player_id: The integer id of the player (0 or 1).
            stockfish_path: Path to the Stockfish engine executable.
            time_limit: Time limit for each move in seconds.
        """
        self.player_id = player_id
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.time_limit = time_limit
        self.engine.configure({"Skill Level": 1})

    def restart(self):
        """Restarts the bot to an initial state."""
        pass

    def inform_action(self, state, player_id, action):
        """Inform the bot about actions taken by other players."""
        pass

    def _convert_uci_to_san(self, board, uci_move):
        """Convert UCI move to SAN notation."""
        try:
            move = chess.Move.from_uci(uci_move)
            return board.san(move)
        except Exception as e:
            print(f"Error converting UCI to SAN: {e}")
            return None

    def step(self, state):
        """Returns the bot's move given the current state.
        
        Args:
            state: The current state of the game.
            
        Returns:
            An action from the legal actions list.
        """
        try:
            # Convert OpenSpiel state to chess.Board
            board = chess.Board(str(state))
            
            # Get Stockfish's best move
            result = self.engine.play(
                board,
                chess.engine.Limit(time=self.time_limit)
            )
            
            if result is None or result.move is None:
                print("Stockfish couldn't find a move")
                return None
            
            # Convert Stockfish's UCI move to SAN
            uci_move = result.move.uci()
            san_move = self._convert_uci_to_san(board, uci_move)
            
            # Get legal actions and their SAN representations
            legal_actions = state.legal_actions()
            legal_moves_dict = {
                state.action_to_string(state.current_player(), action): action 
                for action in legal_actions
            }
            
            # Debug output
            print(f"Stockfish UCI move: {uci_move}")
            print(f"Converted to SAN: {san_move}")
            print(f"Legal moves: {legal_moves_dict.keys()}")
            
            # Find matching action
            if san_move in legal_moves_dict:
                return legal_moves_dict[san_move]
            
            print(f"Failed to match Stockfish move {san_move} with legal moves")
            return None
            
        except Exception as e:
            print(f"Error in Stockfish bot: {str(e)}")
            return None

    def __del__(self):
        """Clean up the Stockfish engine when the bot is destroyed."""
        if hasattr(self, 'engine'):
            self.engine.quit()