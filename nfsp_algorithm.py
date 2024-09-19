import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pyspiel

class NFSPAgent:
    def __init__(self, game, policy_network, value_network, 
                 epsilon=0.1, learning_rate=1e-3, discount_factor=0.99, 
                 replay_buffer_size=10000, batch_size=32):
        self.game = game
        self.policy_network = policy_network
        self.value_network = value_network
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.replay_buffer = []

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.current_state = None

    def choose_action(self, state):
        legal_actions = state.legal_actions(state.current_player())
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)
        
        state_tensor = tf.convert_to_tensor([self._state_to_features(state)], dtype=tf.float32)
        policy = self.policy_network(state_tensor)
        action_probs = tf.nn.softmax(policy[0]).numpy()
        
        # Ensure action_probs has the same length as the total number of actions
        total_actions = max(legal_actions) + 1
        if len(action_probs) < total_actions:
            action_probs = np.pad(action_probs, (0, total_actions - len(action_probs)), 'constant')
        
        # Filter action probabilities for legal actions
        legal_action_probs = [action_probs[action] for action in legal_actions]
        legal_action_probs = np.array(legal_action_probs)
        
        # Check if the sum of legal_action_probs is zero
        sum_probs = legal_action_probs.sum()
        if sum_probs == 0:
            # If all probabilities are zero, assign equal probability to each legal action
            legal_action_probs = np.ones_like(legal_action_probs) / len(legal_action_probs)
        else:
            legal_action_probs /= sum_probs  # Normalize to sum to 1
        
        # Debugging prints
        print(f"Legal actions: {legal_actions}")
        print(f"Filtered action probabilities: {legal_action_probs}")
        print(f"Sizes - Legal actions: {len(legal_actions)}, Filtered action probabilities: {len(legal_action_probs)}")
        
        if len(legal_actions) != len(legal_action_probs):
            raise ValueError("Mismatch between the number of legal actions and filtered action probabilities")
        
        return np.random.choice(legal_actions, p=legal_action_probs)

    def step(self, state):
        """Choose an action based on the current state."""
        return self.choose_action(state)

    def train(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = np.random.choice(self.replay_buffer, self.batch_size)
        for s, a, r, ns, d in batch:
            self._train_on_batch(s, a, r, ns, d)

    def _train_on_batch(self, state, action, reward, next_state, done):
        state_features = self._state_to_features(state)
        next_state_features = self._state_to_features(next_state)

        with tf.GradientTape() as tape:
            policy = self.policy_network(tf.convert_to_tensor([state_features], dtype=tf.float32))
            value = self.value_network(tf.convert_to_tensor([state_features], dtype=tf.float32))
            next_value = self.value_network(tf.convert_to_tensor([next_state_features], dtype=tf.float32))
            target_value = reward + (1.0 - done) * self.discount_factor * next_value
            value_loss = tf.reduce_mean(tf.square(value - target_value))
            policy_loss = -tf.reduce_mean(tf.math.log(tf.gather(policy[0], action)) * (target_value - value))

        grads = tape.gradient(value_loss + policy_loss, self.policy_network.trainable_variables + self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables + self.value_network.trainable_variables))

    def _state_to_features(self, state):
        """Convert game state to feature vector."""
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
        """Update the agent with the action taken by the other player."""
        # No specific implementation is required for this method if the agent
        # does not use this information for updating its state.
        pass

    def restart(self):
        """Reset the agent’s state or clear any stored information between games."""
        self.current_state = None
