<h1 align="center"> Exploring the Performance of Monte Carlo Tree Search, Genetic Algorithm, and Neural Fictitious Self-Play in Tournament-Style Chess and Puzzle-Solving within the OpenSpiel Framework. </h1>

<h2 align="center"> School of Computer Science & Applied Mathematics: University of the Witwatersrand </h2>

<h2 align="center"> Supervised by Dr. Branden Ingram and Dr. Pravesh Ranchod </h2>

This reasearch project is built using [OpenSpiel](https://github.com/google-deepmind/open_spiel): A collection of environments and algorithms for research in general reinforcement learning and search/planning in games.

### Research Paper
The final Research Paper can be found here: [Research Report](./Research Report.pdf)

### Dependencies 
```bash
pip install chess
pip install tensorflow
pip install chess.com
pip install pandas
pip install ches.pgn
pip install pyspiel
pip install tqdm
pip install torch
pip install numpy
pip install matlplotlib
```

### To run a single game of Chess using any of the implemented agents:

1. Bots:
- Random Agents: `random`
- Human Player: `human`
- Vanilla MCTS: `mcts`
- MCTS leveraging historical PGN Chess Data: `mcts_trained_pgn`
- MCTS leveraging historical PGN Puzzle Data: `mcts_trained_puzzle`
- Genetic Algorithm Agent: `ga`
- Neural Fictitous Self-Play Agent: `nfsp`
- Baseline Stockfish engine playing at level 5 (Elo rating: 1400-1500): `stockfish`

    - White player: `bot1`
    - Black player: `bot2`

```bash
python3 play_chess.py --player1=bot1 --player2=bot2
```

2. Example Case:
```bash
python3 play_chess.py --player1=mcts --player2=mcts_trained_pgn
```

### Note: 
In order to obtain the Neural Fictitious Self-Play Model the model needs be trained as the saved model is too large to track through git.
```bash
python3 nfsp_algorithm.py
```
