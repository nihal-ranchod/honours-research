import numpy as np
import pyspiel
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import mcts
from collections import deque
import random

class NFSPBot:
    def __init__(self, game, player_id, model_path="nfsp_model.pth"):
        self.game = game
        self.player_id = player_id
        self.model_path = model_path
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.rewards = []
        self.losses = []
        self.training_data = []  # Store state-action pairs for training
        self.replay_buffer = deque(maxlen=10000)
        self.target_model = self.build_model()
        self.target_model.load_state_dict(self.model.state_dict())

    def build_model(self):
        # A simple feedforward neural network for NFSP
        input_size = self.game.observation_tensor_size()
        output_size = self.game.num_distinct_actions()
        model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        return model

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state, epsilon=0.1):
        # Epsilon-greedy action selection
        legal_actions = state.legal_actions()
        if np.random.rand() < epsilon:
            return np.random.choice(legal_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.observation_tensor(self.player_id))
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                q_values = self.model(state_tensor)
                
                # Filter Q-values for legal actions only
                legal_q_values = q_values[0, legal_actions]
                return legal_actions[torch.argmax(legal_q_values).item()]

    def inform_action(self, state, player_id, action):
        # Store state-action pairs for training
        self.training_data.append((state.observation_tensor(player_id), action))
        self.replay_buffer.append((state.observation_tensor(player_id), action))

    def train(self):
        # Train the model using collected training data
        if len(self.training_data) == 0:
            return

        states, actions = zip(*self.training_data)
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)

        # Dummy target Q-values (for simplicity)
        target_q_values = torch.zeros(len(actions))

        self.optimizer.zero_grad()
        q_values = self.model(states_tensor)
        loss = self.loss_fn(q_values.gather(1, actions_tensor.unsqueeze(1)), target_q_values.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.training_data.clear()  # Clear after training

        if len(self.replay_buffer) % 1000 == 0:  # Update target network periodically
            self.update_target_network()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

    def plot_learning_progress(self):
        plt.plot(self.losses)
        plt.xlabel('Training Iterations')
        plt.ylabel('Loss')
        plt.title('NFSP Learning Progress')
        plt.savefig('nfsp_learning_progress.png')
        plt.show()

    def step(self, state):
        action = self.select_action(state)
        return action

    def restart(self):
        self.training_data.clear()  # Clear data for new game

# Example usage
if __name__ == "__main__":
    game = pyspiel.load_game("chess")  # Load chess game from OpenSpiel
    nfsp_bot = NFSPBot(game, player_id=0)  # Create NFSP bot for player 0
    # Create MCTS bot for player 1
    mcts_bot = mcts.MCTSBot(
        game,
        uct_c=2,
        max_simulations=1000,
        evaluator=mcts.RandomRolloutEvaluator(1, np.random.RandomState()),
        random_state=np.random.RandomState(),
        solve=True
    ) 
    #random_bot = uniform_random.UniformRandomBot(1, np.random.RandomState())  # Create random bot for player 1

    num_games = 100  # Number of games to train the NFSP bot
    for _ in range(num_games):
        state = game.new_initial_state()  # Start a new game
        while not state.is_terminal():
            if state.current_player() == nfsp_bot.player_id:
                action = nfsp_bot.step(state)  # NFSP bot's action
            else:
                action = mcts_bot.step(state)  # Random bot's action

            nfsp_bot.inform_action(state, state.current_player(), action)  # Inform NFSP bot of the action taken
            state.apply_action(action)  # Apply the action to the game state

        nfsp_bot.train()  # Train the NFSP bot after each game

    nfsp_bot.save_model()  # Save the model after training
    nfsp_bot.plot_learning_progress()  # Plot learning progress
