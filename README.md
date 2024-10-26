<h1 align="center"> Investigating the Optimal AI Algorithm for Developing Expert-Level Chess Gameplay within the OpenSpiel Framework. </h1>

<h2 align="center"> School of Computer Science & Applied Mathematics: University of the Witwatersrand </h2>

<h2 align="center"> Supervised by Dr. Branden Ingram and Dr. Pravesh Ranchod </h2>

This reasearch project is built using [OpenSpiel](https://github.com/google-deepmind/open_spiel): A Framework for Reinforcement Learning in Games.

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

### To run a single game of Monte Carlo Tree Search Variants:

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