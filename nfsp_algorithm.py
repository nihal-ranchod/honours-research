import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pyspiel
from collections import deque
import random
import os
import csv

# CSV logging setup
csv_file = 'nfsp_agent_training.csv'
csv_columns = ['step', 'state', 'action', 'reward', 'next_state', 'done', 'value_loss', 'policy_loss']
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()

class NFSPAgent:
    def __init__(self, game, policy_network, value_network, 
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, 
                 learning_rate=1e-2, discount_factor=0.995, 
                 replay_buffer_size=100000, batch_size=64):
        self.game = game
        self.policy_network = policy_network
        self.value_network = value_network
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.current_state = None

        # TensorBoard setup
        self.summary_writer = tf.summary.create_file_writer('logs/nfsp_agent')

        # Step counter
        self.step_counter = 0

    def choose_action(self, state):
        legal_actions = state.legal_actions(state.current_player())
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)
        
        state_tensor = tf.convert_to_tensor([self._state_to_features(state)], dtype=tf.float32)
        policy = self.policy_network(state_tensor)
        action_probs = tf.nn.softmax(policy[0]).numpy()
        
        total_actions = max(legal_actions) + 1
        if len(action_probs) < total_actions:
            action_probs = np.pad(action_probs, (0, total_actions - len(action_probs)), 'constant')
        
        legal_action_probs = np.array([action_probs[action] for action in legal_actions])
        
        sum_probs = legal_action_probs.sum()
        if sum_probs == 0:
            legal_action_probs = np.ones_like(legal_action_probs) / len(legal_action_probs)
        else:
            legal_action_probs /= sum_probs
        
        return np.random.choice(legal_actions, p=legal_action_probs)

    def step(self, state):
        return self.choose_action(state)

    def train(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        for s, a, r, ns, d in batch:
            self._train_on_batch(s, a, r, ns, d)

        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def _train_on_batch(self, state, action, reward, next_state, done):
        state_features = self._state_to_features(state)
        next_state_features = self._state_to_features(next_state)

        with tf.profiler.experimental.Trace('train', step_num=self.optimizer.iterations, _r=1):
            with tf.GradientTape() as tape:
                policy = self.policy_network(tf.convert_to_tensor([state_features], dtype=tf.float32))
                value = self.value_network(tf.convert_to_tensor([state_features], dtype=tf.float32))
                next_value = self.value_network(tf.convert_to_tensor([next_state_features], dtype=tf.float32))
                target_value = reward + (1.0 - done) * self.discount_factor * next_value
                value_loss = tf.reduce_mean(tf.square(value - target_value))
                policy_loss = -tf.reduce_mean(tf.math.log(tf.gather(policy[0], action)) * (target_value - value))

            grads = tape.gradient(value_loss + policy_loss, self.policy_network.trainable_variables + self.value_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables + self.value_network.trainable_variables))

        # Log metrics to TensorBoard
        with self.summary_writer.as_default():
            tf.summary.scalar('value_loss', value_loss, step=self.optimizer.iterations)
            tf.summary.scalar('policy_loss', policy_loss, step=self.optimizer.iterations)
            tf.summary.scalar('reward', reward, step=self.optimizer.iterations)

        # Log metrics to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            writer.writerow({
                'step': self.step_counter,
                'state': self._state_to_features(state).tolist(),
                'action': action,
                'reward': reward,
                'next_state': self._state_to_features(next_state).tolist(),
                'done': done,
                'value_loss': value_loss.numpy(),
                'policy_loss': policy_loss.numpy()
            })

        self.step_counter += 1

    def _state_to_features(self, state):
        return np.array(state.observation_tensor())

    @staticmethod
    def build_network(input_shape, output_shape):
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_shape)
        ])
        return model

    def inform_action(self, state, player_id, action):
        pass

    def restart(self):
        self.current_state = None

    def evaluate(self, num_episodes=10):
        total_reward = 0
        for _ in range(num_episodes):
            state = self.game.new_initial_state()
            done = False
            while not done:
                action = self.choose_action(state)
                state.apply_action(action)
                reward = state.rewards()[state.current_player()]
                total_reward += reward
                done = state.is_terminal()
        avg_reward = total_reward / num_episodes
        return avg_reward

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore_checkpoint(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
        else:
            print("No checkpoint found, starting from scratch.")