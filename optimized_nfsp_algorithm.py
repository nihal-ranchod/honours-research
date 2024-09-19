
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pyspiel

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.position = 0

    def store(self, experience):
        if len(self.buffer) < self.size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

class NFSPAgent:
    def __init__(self, game, state_shape, num_actions, 
                 epsilon=0.1, learning_rate=1e-3, discount_factor=0.99, 
                 replay_buffer_size=50000, batch_size=32, epsilon_decay=0.995, min_epsilon=0.01):
        
        self.game = game
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size

        # Policy Network
        self.policy_network = self.build_network(state_shape, num_actions)
        self.target_policy_network = self.build_network(state_shape, num_actions)  # Target network for stability
        self.target_policy_network.set_weights(self.policy_network.get_weights())

        # Value Network
        self.value_network = self.build_network(state_shape, num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.update_target_steps = 100  # Number of steps to update target network

    def build_network(self, state_shape, num_actions):
        model = tf.keras.Sequential([
            layers.Input(shape=state_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_actions)
        ])
        return model

    def choose_action(self, state):
        # Exploration phase (choose random action with probability epsilon)
        if np.random.rand() < self.epsilon:
            legal_actions = state.legal_actions(state.current_player())
            return np.random.choice(legal_actions)

        # Convert state to tensor for the neural network input
        state_tensor = tf.convert_to_tensor([self._state_to_features(state)], dtype=tf.float32)
        action_logits = self.policy_network(state_tensor)
        action_probs = tf.nn.softmax(action_logits).numpy()[0]  # Get the probabilities for actions

        # Initialize mask with zeros, size equal to the number of actions
        mask = np.zeros(self.num_actions)
        
        # Get legal actions and ensure they are within the range of possible actions
        legal_actions = state.legal_actions(state.current_player())
        legal_actions = [action for action in legal_actions if action < self.num_actions]  # Ensure valid indices

        # Apply mask for legal actions
        mask[legal_actions] = 1.0

        # Apply the mask to the action probabilities
        masked_probs = mask * action_probs

        # Check if all masked probabilities are zero or if there are NaNs
        if np.sum(masked_probs) == 0 or np.isnan(masked_probs).any():
            # Assign uniform probability over legal actions if there's an issue
            masked_probs = mask / np.sum(mask)

        # Normalize the masked probabilities to ensure they sum to 1
        masked_probs /= np.sum(masked_probs)

        # Now sample an action based on the masked probabilities
        return np.random.choice(range(self.num_actions), p=masked_probs)


    def store_experience(self, experience):
        self.replay_buffer.store(experience)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data to sample

        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Value network target calculation
        next_q_values = self.target_policy_network(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards + self.discount_factor * max_next_q_values * (1 - dones)

        with tf.GradientTape() as tape:
            q_values = tf.reduce_sum(self.value_network(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.reduce_mean((target_q_values - q_values) ** 2)

        grads = tape.gradient(loss, self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.value_network.trainable_variables))

        # Update epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_policy_network.set_weights(self.policy_network.get_weights())

    def _state_to_features(self, state):
        return np.array(state.observation_tensor(), dtype=np.float32)
    
    def inform_action(self, state, player_id, action):
        """
        This method will inform the game about the action taken by the NFSP agent.
        """
        return self.choose_action(state)

    def step(self, state):
        """
        Take a step by choosing an action given the current game state.
        """
        return self.choose_action(state)