import os
import numpy as np
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms import nfsp
from open_spiel.python import rl_environment

tf.disable_v2_behavior()

class NFSPAgent:
    def __init__(self, game, player_id, hidden_layers_sizes=[128, 128], reservoir_buffer_capacity=2e6, anticipatory_param=0.1, batch_size=128, rl_learning_rate=0.01, sl_learning_rate=0.01, min_buffer_size_to_learn=1000, learn_every=64, optimizer_str="adam", epsilon_decay_duration=int(1e6), epsilon_start=0.06, epsilon_end=0.001):
        self.game = game
        self.player_id = player_id
        self.hidden_layers_sizes = hidden_layers_sizes
        self.reservoir_buffer_capacity = reservoir_buffer_capacity
        self.anticipatory_param = anticipatory_param
        self.batch_size = batch_size
        self.rl_learning_rate = rl_learning_rate
        self.sl_learning_rate = sl_learning_rate
        self.min_buffer_size_to_learn = min_buffer_size_to_learn
        self.learn_every = learn_every
        self.optimizer_str = optimizer_str
        self.epsilon_decay_duration = epsilon_decay_duration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        self.env = rl_environment.Environment(game)
        self.info_state_size = self.env.observation_spec()["info_state"][0]
        self.num_actions = self.env.action_spec()["num_actions"]

        self.sess = tf.Session()
        self._init_agent()

    def _init_agent(self):
        self.agent = nfsp.NFSP(
            session=self.sess,
            player_id=self.player_id,
            state_representation_size=self.info_state_size,
            num_actions=self.num_actions,
            hidden_layers_sizes=self.hidden_layers_sizes,
            reservoir_buffer_capacity=self.reservoir_buffer_capacity,
            anticipatory_param=self.anticipatory_param,
            batch_size=self.batch_size,
            rl_learning_rate=self.rl_learning_rate,
            sl_learning_rate=self.sl_learning_rate,
            min_buffer_size_to_learn=self.min_buffer_size_to_learn,
            learn_every=self.learn_every,
            optimizer_str=self.optimizer_str,
            epsilon_decay_duration=self.epsilon_decay_duration,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end
        )
        self.sess.run(tf.global_variables_initializer())

    def step(self, time_step):
        return self.agent.step(time_step)

    def action_to_string(self, action):
        return self.env.get_state.action_to_string(self.player_id, action)

    def string_to_action(self, action_str):
        return self.env.get_state.string_to_action(self.player_id, action_str)

    def save(self, file_path):
        self.agent.save(file_path)

    def restore(self, file_path):
        self.agent.restore(file_path)

    def train(self, num_episodes=10000):
        returns = []
        for ep in range(num_episodes):
            time_step = self.env.reset()
            episode_return = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == self.player_id:
                    agent_output = self.step(time_step)
                    action = self.string_to_action(self.action_to_string(agent_output.action))
                else:
                    action = np.random.choice(time_step.observations["legal_actions"][player_id])
                time_step = self.env.step([action])
                episode_return += time_step.rewards[self.player_id]
            returns.append(episode_return)
            if (ep + 1) % 100 == 0:
                avg_return = np.mean(returns[-100:])
                print(f"Episode {ep + 1}: Average Return: {avg_return}")
        return returns

    def __del__(self):
        self.sess.close()