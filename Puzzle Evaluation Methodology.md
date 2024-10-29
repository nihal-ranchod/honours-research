# Puzzle Evaluation Script Methodology

This document provides an overview of the methodology, structure, and parameters used in the chess puzzle evaluation script. This script is designed to evaluate the performance of the implemented chess-playing bots on a set of puzzles stored in PGN (Portable Game Notation) format. The evaluation assesses each bot's puzzle-solving capabilities by measuring metrics such as move accuracy, time taken, and overall puzzle-solving rate.

---

## Table of Contents
- [Overview](#overview)
- [Setup and Installation](#setup-and-installation)
- [Parameters](#parameters)
- [Methodology](#methodology)
  - [Bot Initialization](#bot-initialization)
  - [Puzzle Loading](#puzzle-loading)
  - [Puzzle Solving Evaluation](#puzzle-solving-evaluation)
  - [Result Analysis](#result-analysis)
- [Outputs](#outputs)
- [Usage](#usage)
- [Error Handling](#error-handling)

---

## Bot Setup and Initialization

The evaluation includes the following bots:

- **MCTS (Monte Carlo Tree Search)**: Plays using random rollouts and a UCT exploration constant.
- **MCTS Trained on PGN Data**: Uses Monte Carlo Tree Search but incorporates prior game data in PGN format for enhanced decision-making.
- **NFSP (Neural Fictitious Self-Play)**: A bot trained through neural fictitious self-play to improve its strategy.
- **Genetic Algorithm (GA)**: A bot trained via a genetic algorithm optimized through evolutionary strategies.
- **Stockfish**: A baseline bot using Stockfish playing at Skill Level 1.
- **Random**: A baseline bot that chooses moves randomly.

---

## Parameters

**Command-Line Flags**
- `game`: Game type (default: "chess").
- `puzzle_pgn`: Path to the puzzle PGN file.
- `num_puzzles`: Number of puzzles for evaluation.
- `max_moves_per_puzzle`: Maximum moves allowed per puzzle.
- `max_simulations`: Maximum simulations for MCTS.
- `solve`: Whether to use MCTS-Solver.
- `stockfish_path`: Path to the Stockfish executable.

## Methodology

Each bot uses a specific strategy to evaluate the next move, as configured through the respective parameters and command-line flags.

### Puzzle Loading
The `load_puzzles` function reads chess puzzles from a PGN file. Each puzzle is converted into a sequence of moves in a format compatible with `pyspiel`, allowing bots to interact with the puzzle directly.

1. **Loading PGN Data**: Puzzles are loaded from a specified PGN file using `chess.pgn`.
2. **Converting Moves**: Each move in the puzzle is converted to an action format that `pyspiel` understands. Correct moves are stored for evaluation.

### Puzzle Solving Evaluation
The core evaluation occurs in the `evaluate_puzzle_solving` function:
1. **Initialization**: The puzzle is initialized to its starting position.
2. **Bot Move Evaluation**: The bot iteratively makes moves, which are compared to the correct sequence of moves for the puzzle.
3. **Performance Metrics**:
   - `solved`: Boolean indicating if the bot solved the puzzle.
   - `moves_taken`: Number of moves taken by the bot.
   - `correct_moves`: Count of moves matching the puzzle's solution.
   - `move_accuracy`: List indicating move-by-move correctness.
   - `time_taken`: Time taken to solve the puzzle.

### Result Analysis
The `analyze_results` function aggregates performance metrics across puzzles:
- **Solve Rate**: Ratio of puzzles solved.
- **Move Accuracy**: Proportion of correct moves to total moves.
- **Average Moves per Puzzle**: Mean moves taken across puzzles.
- **Average Time per Puzzle**: Average time spent per puzzle.

These results are saved to a JSON file for further analysis.

---

## Outputs

1. **Result File**: Results are saved in `puzzle_evaluation_results.json`. This file contains:
   - Performance summary per bot.
   - Configuration details (e.g., number of puzzles, max moves per puzzle).
   - Timestamp of the evaluation.

2. **Console Logs**: Detailed logs (if `--quiet` is not set) display each bot's progress on each puzzle.
