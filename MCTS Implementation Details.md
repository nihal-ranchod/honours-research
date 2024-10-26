# MCTS_with_PGN_Data: Enhanced Monte Carlo Tree Search for Chess with Historical Data

## Overview
The `MCTS_with_PGN_Data` class is an advanced Monte Carlo Tree Search (MCTS) bot designed to improve its decision-making capabilities by incorporating historical data from past chess games. It leverages patterns observed in previously played games to inform move selection, particularly in opening phases, making it more aggressive and effective against traditional MCTS approaches. 

This bot’s modifications allow it to:
- Reference commonly used moves from a library of historical games stored in PGN (Portable Game Notation) format.
- Utilize **progressive history** to bias move selection based on the success rates of moves in past games.
- Cache evaluations to optimize repeated state computations.

## Key Components

### 1. **EnhancedSearchNode Class**
The `EnhancedSearchNode` extends the base `SearchNode` by incorporating additional state information. This allows nodes to retain specific board states, enhancing data accessibility during tree traversal.

#### Parameters
- `state`: Stores the board state associated with each node for easier tracking of board configurations and improved move evaluation consistency.

### 2. **Initialization and Configuration**
The MCTS bot can be configured with various parameters to adjust exploration, caching, and opening moves from training data.

#### Parameters
- `uct_c`: The exploration parameter controlling the balance between exploration and exploitation in UCT (Upper Confidence Bound for Trees).
- `max_simulations`: Maximum number of simulations per move.
- `training_data`: PGN file containing historical games data.
- `cache_size` and `min_position_visits`: Parameters to manage caching and threshold for using historical data.

### 3. **Historical Data Integration**
The `_load_training_data` method parses historical chess games from the PGN file. This data feeds into:
- `position_stats`: Tracks move occurrences for each position to inform likelihood-based move selection.
- `opening_book`: Stores move sequences frequently played in openings.
- `progressive_history`: Tracks the performance (win, loss, draw) of each move to bias selection toward moves that historically performed well.

### 4. **Progressive History and Move Probability Adjustments**
The `progressive_history` dictionary is used to assign scores to moves based on their historical outcomes. The `_update_progressive_history` method adjusts move probabilities dynamically based on previous outcomes.

#### Aggressive Move Selection
The aggressive approach is achieved by boosting moves that have higher historical success rates. This is implemented as a **history bonus**, added to the UCT score during move selection to favor moves with strong past performance. This approach is especially effective in balancing exploration and exploitation, allowing the bot to seek moves that have proven successful while still exploring alternatives.

## Core Methods

### `_load_training_data(pgn_file)`
Loads historical games data from a PGN file. Moves from the first 15 moves are treated as opening book moves and receive priority during the opening phase of a game.

### `_get_state_key(state)` and `_get_state_key_from_board(board)`
Extracts a position key using board state representations (FEN format) to uniquely identify and retrieve specific positions across different games.

### `_get_prior_policy(state)`
Generates move probabilities that incorporate both legal move counts from historical data and adjustments from `progressive_history`. Moves frequently played in similar positions, especially those with high success rates, are weighted more heavily.

### `_select_child(node, state)`
Selects the next child node using both UCT value and progressive history data. The aggressive nature of move selection is emphasized here, where a history bonus is factored into the UCT score.

### `mcts_search(state)`
Runs the MCTS search process by expanding nodes based on prior policies, which are influenced by historical data.

### `_backup(visit_path, returns, solved)`
Backpropagates values along the search path, updating cumulative rewards, and checks if a node can be marked as “solved.” Progressive history is also updated here for moves with high returns.

### `step_with_policy(state)`
Evaluates a state using MCTS and historical data. If the state’s policy has already been computed and cached with a high evaluation score, it can be retrieved directly for efficiency. Otherwise, the bot continues with MCTS, updating the cache as necessary.

## Aggressive Move Selection Justification
The addition of historical data and aggressive prioritization of high-success moves creates an opening and midgame bias toward moves that have succeeded in similar games. In chess, early control often leads to a more favorable position, enabling stronger endgames. The bot’s aggression is especially helpful in:
1. **Achieving Dominance in Openings**: Leveraging opening book moves allows the bot to establish control early on.
2. **Reducing Exploration in High-Stakes Scenarios**: Moves that have proven effective historically are prioritized, reducing unnecessary exploration in familiar positions.
3. **Encouraging High-Efficiency Moves**: Moves with high win rates or draws from past data are favored, making the bot capable of closing out games more reliably.


## Conclusion
`MCTS_with_PGN_Data` is a powerful extension of traditional vanilla MCTS, integrating historical game data to make move selection more aggressive and controlled. By incorporating historical success rates, this bot can develop an opening advantage, minimize exploration in high-certainty situations, and effectively navigate both known and unknown positions with improved precision.

