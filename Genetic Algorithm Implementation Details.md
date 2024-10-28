
# Genetic Algorithm Chess Bot: Implementation and Methodology

## Overview
Implements a chess-playing bot using a genetic algorithm (GA) framework. The bot is designed to evaluate chess board positions and make moves by evolving a neural network model, `ChessNet`, that assesses the state of the board. The model is trained on chess games and/or puzzles to optimize its ability to evaluate positions, with fitness scores assigned based on performance. Through successive generations, the GA-based chess bot improves its evaluation capabilities.

## Contents
- [Architecture](#architecture)
- [Neural Network Model (`ChessNet`)](#neural-network-model-chessnet)
- [Genetic Algorithm Workflow](#genetic-algorithm-workflow)
- [Training on Chess Games and Puzzles](#training-on-chess-games-and-puzzles)
- [Hyperparameters and Settings](#hyperparameters-and-settings)
- [Logging and Visualization](#logging-and-visualization)
- [Saving and Loading Models](#saving-and-loading-models)

---

## Architecture
The bot is built upon two main components:
1. **Neural Network Model (`ChessNet`)**: A convolutional neural network (CNN) designed to evaluate chess positions, outputting a score that represents the board's favourability for the player.
2. **Genetic Algorithm (`GeneticChessBot`)**: A population-based optimization algorithm that trains and evolves the `ChessNet` models by selective breeding, mutation, and fitness-based survival.

## Neural Network Model (`ChessNet`)
The `ChessNet` model is a CNN with the following layers:
1. **Convolutional Layers**:
   - Three convolutional layers with ReLU activations.
   - Input: 12 channels (representing piece types and colors on an 8x8 board).
   - Filters: 64, 128, and 64 filters, respectively.
   - Padding: Ensures the spatial dimensions are preserved.

2. **Fully Connected Layers**:
   - A fully connected layer with 512 neurons and ReLU activation.
   - An output layer with a single neuron, applying a `tanh` activation, resulting in a score between -1 (disadvantageous position) and 1 (advantageous position).

3. **Data Representation**:
   - The board is represented as a tensor of size `12x8x8`, where each piece type and color has a dedicated channel.
   - The tensor encoding allows the CNN to learn piece-specific patterns and interactions on the board.

## Genetic Algorithm Workflow
The GA optimizes the `ChessNet` model across multiple generations. Each generation comprises:
1. **Population Initialization**:
   - A population of `ChessNet` models is generated with random initial weights.

2. **Fitness Evaluation**:
   - Each model in the population plays several games or solves puzzles, accumulating a score based on its performance.
   - Fitness is calculated as the model's ability to correctly evaluate board positions, matching the expected outcomes.

3. **Selection**:
   - Models with the highest fitness are selected to pass their traits to the next generation.
   - The top two performing models are preserved without modifications to maintain elite traits.

4. **Crossover**:
   - For each pair of selected parents, a child model is created.
   - Crossover swaps weights from each parent model probabilistically, creating a new `ChessNet` model that inherits traits from both parents.

5. **Mutation**:
   - After crossover, random mutations are applied to the child’s weights to maintain genetic diversity.
   - The mutation rate and mutation strength parameters control the extent of these changes, balancing between exploration and convergence.

6. **Population Replacement**:
   - The new generation consists of elite models, crossover offspring, and mutated children, forming a fresh population for the next generation.

## Training on Chess Games and Puzzles
The bot can be trained using two types of data:
- **Standard Games** (`train_on_pgn`): Uses a PGN file of complete games to improve position evaluation.
  - Each game is loaded, and the board is evaluated based on the final game result.
  - Fitness is calculated based on the alignment between predicted and actual game outcomes.
  
- **Puzzles** (`train_on_puzzles`): Uses a PGN file of puzzles (typically tactics).
  - For each puzzle, the bot evaluates the position before and after a key move, scoring fitness based on improvement.

The bot's fitness is determined by the accuracy of the position evaluations in relation to the expected outcome, encouraging the bot to learn effective board assessment heuristics.

### Key Functions
1. **`evaluate_position`**: Converts a board state to a tensor and predicts its score.
2. **`crossover`**: Performs genetic recombination between two parent networks.
3. **`mutate`**: Applies random noise to model weights based on mutation rate and strength.
4. **`save_best_genome`**: Saves the best-performing model of each generation for later use.

## Hyperparameters and Settings
Several hyperparameters influence the GA's performance:
- **Population Size**: Controls the number of models per generation.
- **Mutation Rate**: The probability that a weight in the model will mutate.
- **Mutation Strength**: Determines the scale of each mutation.
- **Games/Puzzles per Genome**: Number of games/puzzles each genome plays during evaluation.
- **Number of Generations**: Total iterations for the GA.
  
These hyperparameters are adjustable to experiment with convergence speed, exploration, and model quality.

## Saving and Loading Models
1. **Best Model Persistence**:
   - The best genome from each generation is preserved, enabling model persistence beyond the training session.
   - Models are saved as serialized objects using Python’s `pickle` module.

---

## Conclusion
This genetic algorithm-based chess bot leverages neural networks to assess and improve board evaluation heuristics. By iteratively evolving through fitness-based selection, crossover, and mutation, the bot enhances its position evaluation skills, becoming increasingly competitive. The combination of genetic algorithms and CNN-based position evaluation allows this bot to explore a vast solution space, uncovering non-trivial patterns in chess strategy.