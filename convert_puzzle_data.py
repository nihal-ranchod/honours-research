import csv
import chess
import chess.pgn
import os

def convert_puzzle_csv_to_pgn(csv_file, pgn_file, max_size_mb=80):
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    
    with open(csv_file, 'r') as csv_f, open(pgn_file, 'w') as pgn_f:
        reader = csv.DictReader(csv_f)
        puzzles_converted = 0
        
        for row in reader:
            fen = row['FEN']
            moves = row['Moves'].split()
            
            # Create a new game and set the starting position
            game = chess.pgn.Game()
            game.setup(fen)
            node = game

            # Add the moves
            board = game.board()
            for move in moves:
                board.push_san(move)
                node = node.add_variation(board.move_stack[-1])

            # Set some metadata
            game.headers["Event"] = "Puzzle"
            game.headers["Site"] = "Lichess"
            game.headers["Date"] = "????.??.??"
            game.headers["Round"] = "-"
            game.headers["White"] = "White"
            game.headers["Black"] = "Black"
            game.headers["Result"] = "*"
            game.headers["PuzzleId"] = row.get('PuzzleId', '')  # Add PuzzleId if available

            # Write the game to the PGN file
            exporter = chess.pgn.FileExporter(pgn_f)
            game.accept(exporter)
            puzzles_converted += 1

            # Check file size
            pgn_f.flush()  # Ensure all data is written to disk
            if os.path.getsize(pgn_file) >= max_size_bytes:
                print(f"Reached maximum file size of {max_size_mb} MB. Stopping conversion.")
                print(f"Converted {puzzles_converted} puzzles.")
                return puzzles_converted

    print(f"Finished converting all puzzles ({puzzles_converted}) without reaching size limit.")
    return puzzles_converted

def main():
    input_csv = "PGN_Data/lichess_db_puzzle.csv" 
    output_pgn = "PGN_Data/lichess_db_puzzle_converted.pgn"  
    max_size_mb = 80  # Maximum size of the output file in MB

    print(f"Converting {input_csv} to {output_pgn} (max size: {max_size_mb} MB)...")
    puzzles_converted = convert_puzzle_csv_to_pgn(input_csv, output_pgn, max_size_mb)
    print(f"Conversion complete! Converted {puzzles_converted} puzzles.")

if __name__ == "__main__":
    main()