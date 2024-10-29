# Chess Bot Tournament Evaluation Methodology

This describes the methodology used for evaluating and comparing the performance of the implemented chess bots, in a round-robin tournament format. The goal is to simulate a series of games between bot pairs and evaluate their relative performance using metrics such as Elo ratings, win rates, and average game lengths.

## Table of Contents
1. [Overview](#overview)
2. [Bot Setup and Initialization](#bot-setup-and-initialization)
3. [Tournament Structure](#tournament-structure)
4. [Game Simulation](#game-simulation)
5. [Result Processing and Elo Rating Calculation](#result-processing-and-elo-rating-calculation)
6. [Data Logging and Result Storage](#data-logging-and-result-storage)
7. [Execution and Configuration](#execution-and-configuration)
8. [Future Work](#future-work)

## Overview

The tournament involves playing a specified number of games between each pair of bots, allowing each bot to play both as White and Black to remove color bias. A round-robin format ensures that every bot competes against all other bots, and the results are stored in JSON format for analysis.

## Bot Setup and Initialization

The tournament includes the following bots:

- **MCTS (Monte Carlo Tree Search)**: Plays using random rollouts and a UCT exploration constant.
- **MCTS Trained on PGN Data**: Uses Monte Carlo Tree Search but incorporates prior game data in PGN format for enhanced decision-making.
- **NFSP (Neural Fictitious Self-Play)**: A bot trained through neural fictitious self-play to improve its strategy.
- **Genetic Algorithm (GA)**: A bot trained via a genetic algorithm optimized through evolutionary strategies.
- **Stockfish**: A baseline bot using Stockfish playing at Skill Level 1.
- **Random**: A baseline bot that chooses moves randomly.

Each bot is initialized based on a predefined `bot_type` identifier. The `_init_bot` function loads and configures each bot with relevant parameters, including game, player ID, and random seeds where applicable. Path configurations for data and model files (e.g., PGN data, genetic algorithm models) are predefined to ensure correct loading.

## Tournament Structure

The tournament is configured to play a set number of games (as specified by `num_games`) between each bot pair, with each bot playing both White and Black. For instance, if `num_games` is set to 50, each pair plays 25 games with one bot as White and 25 with the other as White.

**Bot Pairs and Known Players**: Bot types are defined in `_KNOWN_PLAYERS`, which is used to generate unique pairings, avoiding redundant matchups. If the number of games is not even, an error is raised.

## Game Simulation

Each game simulation proceeds as follows:

1. **Initial State**: The game board is initialized.
2. **Turn Iteration**:
   - **Chance Node**: If the current player is `CHANCE`, an action is chosen based on predefined probabilities for each possible outcome.
   - **Bot Action**: The current bot chooses a move based on its implemented strategy.
3. **Action Application**: The chosen move is applied to the game state, and the loop continues until a terminal state is reached.
4. **Game Termination**: The game ends when a terminal state is reached, at which point the result is determined:
   - **White Wins**: `result = 1`
   - **Black Wins**: `result = -1`
   - **Draw**: `result = 0`
5. **Logging Game Data**: Each game logs the moves, total game time, and result for post-game analysis.

Exception handling is included to catch and manage errors that may arise during the game (e.g., model loading issues or bot failures).

## Result Processing and Elo Rating Calculation

Each bot’s performance is tracked and evaluated using the following metrics:

- **Total Games**: The number of games played by each bot.
- **Wins as White and Black**: Counts of wins based on the bot's color.
- **Draws**: Total number of games ending in a draw.
- **Average Game Length**: Calculated as the total moves divided by the number of games played.
- **Elo Rating**: Updated after each game based on the Elo rating system, with the following steps:

### Elo Rating Calculation

1. **Score Calculation**: Each game’s outcome (White win, Black win, or draw) results in a score update for each bot (White score: 1, Black score: 0, Draw score: 0.5).
2. **Expected Score Calculation**: Based on the current Elo rating difference between the two bots:
   \[
   \text{expected\_white} = \frac{1}{1 + 10^{(-\text{rating\_diff} / 400)}}
   \]
   The Black expected score is \(1 - \text{expected\_white}\).
3. **Rating Update**: Using a configurable K-factor (default: 32), each bot’s Elo rating is updated as follows:
   - White: `elo_rating += K * (score - expected_score)`
   - Black: `elo_rating += K * (score - expected_score)`

This method provides a dynamic Elo rating system, allowing stronger bots to accumulate higher ratings over time.

## Data Logging and Result Storage

After each game, results are updated in a `results` dictionary with detailed metrics per bot. At the end of the tournament, the data is saved in JSON format, including:

- **Individual Bot Results**: Win counts, Elo ratings, average game length, and other relevant metrics.
- **Tournament Configuration**: Number of games, timestamp, and other configurable parameters for reproducibility.

