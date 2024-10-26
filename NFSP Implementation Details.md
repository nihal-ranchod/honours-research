
# Neural Fictitious Self-Play Chess Bot: Implementation Details and Methodology

## Table of Contents
1. [Introduction](#introduction)
2. [Design Goals and Constraints](#design-goals-and-constraints)
3. [Neural Fictitious Self-Play (NFSP) Overview](#neural-fictitious-self-play-nfsp-overview)
4. [Implementation Details](#implementation-details)
   - [1. Chess Environment Setup](#1-chess-environment-setup)
   - [2. NFSP Agent Structure](#2-nfsp-agent-structure)
   - [3. Experience Replay](#3-experience-replay)
   - [4. Supervised and Reinforcement Learning Integration](#4-supervised-and-reinforcement-learning-integration)
   - [5. Aggressive Play Rationale](#5-aggressive-play-rationale)
5. [Training and Evaluation](#training-and-evaluation)
6. [Performance and Insights](#performance-and-insights)

---

## Introduction
The NFSP Chess Bot aims to tackle chess through a combination of supervised learning and reinforcement learning using neural fictitious self-play. Leveraging OpenSpiel, for multi-agent environments, the NFSP approach dynamically adjusts its strategy through a mix of memory-based self-play and exploration.

## Design Goals and Constraints
1. **Learning from Experience:** The bot should improve through experience, not relying on extensive domain-specific heuristics.
2. **Aggressive Playstyle:** Design the bot to maintain an aggressive approach, particularly in the mid-game, to avoid excessive passive strategies that may lead to stalemates and draws.

## Neural Fictitious Self-Play (NFSP) Overview
NFSP is a hybrid of supervised learning (SL) and reinforcement learning (RL) to approximate Nash Equilibrium in two-player zero-sum games. It employs:
1. **Supervised Learning (SL):** Tracks opponents’ strategies in experience replay buffers to learn approximate equilibria.
2. **Reinforcement Learning (RL):** Uses Deep Q-Learning (DQN) to train the agent based on its own experiences.

In this implementation, the agent learns both short-term tactics through SL and long-term strategic planning with RL, achieving a balance between opponent modeling and self-improvement.

---

## Implementation Details

### 1. Chess Environment Setup
- **Platform:** OpenSpiel framework for handling game states, moves, and transitions in chess.
- **Agent-Environment Loop:** The bot interacts with the chess environment by selecting actions (moves) based on its policy, receiving rewards, and updating its knowledge base.
- **Reward Function:** A custom reward function based on game outcomes, including victory, checkmate, and material gain, penalizing moves leading to disadvantages and prioritizing checkmate strategies.

### 2. NFSP Agent Structure
The NFSP agent architecture includes:
- **Policy Network:** A supervised learning network that approximates optimal moves based on historical data.
- **Q-Network (DQN):** A reinforcement learning network that updates based on temporal difference learning to adjust the bot's response to various chess positions.

**Neural Network Architecture**:
- **Policy Network:** A convolutional neural network (CNN) suited to the spatial nature of the chessboard, followed by fully connected layers.
- **Q-Network:** Similar CNN architecture but with separate output layers for Q-value predictions.

### 3. Experience Replay
NFSP utilizes two types of experience replay:
- **SL Replay Buffer:** Stores game states and corresponding actions observed from opponents. This buffer trains the policy network through supervised learning.
- **RL Replay Buffer:** Contains state-action-reward-next_state tuples for Q-network training through DQN.
  
Buffers store diverse experiences to enhance learning stability and strategy variance.

### 4. Supervised and Reinforcement Learning Integration
- **Supervised Learning (Policy Network):** The policy network observes and mimics the actions from self-play and high-level games (including data from engines like Stockfish), allowing the bot to learn effective opening and endgame moves.
- **Reinforcement Learning (Q-Network):** The Q-network focuses on maximizing future rewards, learning through exploration and exploitation. This network also facilitates **ε-greedy exploration** (ε=0.1), ensuring balanced move selection between known and experimental strategies.

By balancing these learning paradigms, the NFSP bot optimally adjusts its strategy to diverse opponents, learning tactical, positional, and endgame concepts.

### 5. Aggressive Play Rationale
The bot's aggressive approach is designed to overcome one of the common issues in self-play training—passive play leading to frequent draws. A key element of NFSP is its tendency to reach Nash equilibrium, which can sometimes result in overly conservative strategies. In chess, such equilibrium may lead to defensive play, stalling, and avoiding confrontation.

#### Aggressive Playstyle in Chess
The bot’s **aggressive moves** prioritize:
- **Material Advantage**: Target pieces with high value early in the game, increasing chances of victory.
- **Position Control**: Dominate the board’s center and pressure the opponent’s critical squares.
- **Checkmate Drive**: Focus on moves that set up checkmate sequences, prioritizing tactics over incremental material gain.

This approach reduces the likelihood of passive draw-prone strategies and increases win rates by forcing opponents to respond under pressure, thus facilitating a decisive outcome.

---

## Training and Evaluation
1. **Training Regimen**: The bot undergoes extensive training through self-play and simulated games against Stockfish and other scripted bots.
   - **Self-Play**: The agent plays against itself, iteratively improving by learning from both its own actions and reactions.
   - **Engine Benchmarking**: Competes with Stockfish for evaluation, reinforcing learning through challenging gameplay.

2. **Evaluation Metrics**: 
   - **Win-Draw-Loss Ratio**: Tracks performance to gauge the impact of aggressive playstyle.
   - **Exploitability**: Measures the bot's robustness in handling varying strategies.
   - **Reward Tracking**: Monitors cumulative rewards to verify consistent improvement.

3. **Training Duration**: The model is trained over multiple epochs, with regular validation checks to adjust for overfitting or lack of exploration.

## Insights

**Key Insights**:
- **Aggression Avoids Draws**: The aggressive style forces opponents into defensive positions, effectively reducing the frequency of drawn games.
- **Strategic Flexibility**: NFSP’s dual learning methods provide the bot with a diverse range of tactical responses, improving adaptability in the mid-game.
- **Balanced Improvement**: The SL-RL hybrid ensures that the bot can adopt a blend of long-term strategy and short-term tactics.
