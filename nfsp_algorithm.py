import os
import numpy as np
import tensorflow.compat.v1 as tf
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import nfsp
import pyspiel

tf.disable_v2_behavior()

class NFSPAgent:
    def __init__(self, game, player_id, hidden_layers_sizes=[256, 256], reservoir_buffer_capacity=1e6, anticipatory_param=0.1):
        self.game = game
        self.player_id = player_id
        self.hidden_layers_sizes = hidden_layers_sizes
        self.reservoir_buffer_capacity = reservoir_buffer_capacity
        self.anticipatory_param = anticipatory_param
        self.learning_rate = 0.001
        self.batch_size = 128
        self.target_network_update_freq = 1000
        self.min_buffer_size_to_learn = 1000
        self._init_agent()

    def _init_agent(self):
        try:
            self.agent = nfsp.NFSP(
                session=self.sess,
                player_id=self.player_id,
                state_representation_size=self.info_state_size,
                num_actions=self.num_actions,
                hidden_layers_sizes=self.hidden_layers_sizes,
                reservoir_buffer_capacity=self.reservoir_buffer_capacity,
                anticipatory_param=self.anticipatory_param,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                min_buffer_size_to_learn=self.min_buffer_size_to_learn,
                target_network_update_freq=self.target_network_update_freq,
                optimizer_str="adam"
            )
        except Exception as e:
            print(f"Error initializing NFSP agent: {e}")
            raise

    def step(self, time_step_or_state):
        try:
            if isinstance(time_step_or_state, rl_environment.TimeStep):
                return self.agent.step(time_step_or_state)
            elif isinstance(time_step_or_state, pyspiel.State):
                info_state = time_step_or_state.observation_tensor(self.player_id)
                legal_actions = time_step_or_state.legal_actions(self.player_id)
                
                print(f"Info state shape: {np.array(info_state).shape}")
                print(f"Legal actions: {legal_actions}")
                
                # Create a more complete dummy TimeStep object
                dummy_time_step = rl_environment.TimeStep(
                    observations={
                        "info_state": [info_state],
                        "legal_actions": [legal_actions],
                        "current_player": self.player_id,
                        "serialized_state": time_step_or_state.serialize()
                    },
                    rewards=[0.0],  # Add a dummy reward
                    discounts=[1.0],  # Add a dummy discount
                    step_type=rl_environment.StepType.MID
                )

                if np.random.rand() < self.anticipatory_param:
                    # Use best response policy (RL agent)
                    rl_step = self.agent._rl_agent.step(dummy_time_step)
                    action = rl_step.action
                else:
                    # Use average policy (SL policy)
                    action_and_probs = self.agent._act(info_state, legal_actions)
                    print(f"Action and probabilities type: {type(action_and_probs)}")
                    print(f"Action and probabilities content: {action_and_probs}")
                    
                    if isinstance(action_and_probs, tuple) and len(action_and_probs) == 2:
                        action, probs = action_and_probs
                        print(f"Action: {action}")
                        print(f"Probabilities type: {type(probs)}")
                        print(f"Probabilities content: {probs}")
                        
                        if isinstance(probs, np.ndarray):
                            if len(probs.shape) > 1:
                                print(f"Probabilities shape before flatten: {probs.shape}")
                                probs = probs.flatten()
                            
                            print(f"Probabilities shape after processing: {probs.shape}")
                            print(f"Probabilities after processing: {probs}")
                            
                            # Normalize probabilities
                            probs = probs / np.sum(probs)
                            
                            # Choose action based on probabilities
                            action = np.random.choice(len(probs), p=probs)
                    else:
                        raise ValueError(f"Unexpected return type from _act: {type(action_and_probs)}")
                
                # Ensure action is valid
                if action is None or action not in legal_actions:
                    print(f"Invalid action {action}, choosing random legal action")
                    action = np.random.choice(legal_actions)
                
                print(f"Chosen action: {action}")
                return action
            else:
                raise ValueError(f"Unexpected input type: {type(time_step_or_state)}")
        except Exception as e:
            print(f"Error in step method: {e}")
            raise

    def action_to_string(self, action):
        return self.env.get_state.action_to_string(self.player_id, action)

    def string_to_action(self, action_str):
        return self.env.get_state.string_to_action(self.player_id, action_str)

    def save(self, file_path):
        self.agent.save(file_path)

    def restore(self, file_path):
        self.agent.restore(file_path)

    def train(self, num_episodes=100000, eval_every=1000, num_eval_episodes=1000):
        """
        Train the NFSP agent with improved process and evaluation.
        
        Args:
            num_episodes (int): Total number of training episodes.
            eval_every (int): Evaluate the agent every `eval_every` episodes.
            num_eval_episodes (int): Number of episodes to use for evaluation.
        
        Returns:
            list: Average returns during training.
        """
        returns = []
        eval_returns = []

        for ep in range(num_episodes):
            time_step = self.env.reset()
            episode_return = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == self.player_id:
                    agent_output = self.step(time_step)
                    action = agent_output.action
                else:
                    action = np.random.choice(time_step.observations["legal_actions"][player_id])
                time_step = self.env.step([action])
                episode_return += time_step.rewards[self.player_id]

            # Add episode return to the list
            returns.append(episode_return)

            # Evaluate the agent periodically
            if (ep + 1) % eval_every == 0:
                eval_return = self.evaluate(num_eval_episodes)
                eval_returns.append(eval_return)
                print(f"Episode {ep + 1}: Eval Return: {eval_return:.2f}, "
                      f"Training Return: {np.mean(returns[-eval_every:]):.2f}")

        return returns, eval_returns

    def evaluate(self, num_episodes):
        """
        Evaluate the NFSP agent against a random opponent.
        
        Args:
            num_episodes (int): Number of episodes to evaluate.
        
        Returns:
            float: Average return of the NFSP agent.
        """
        total_return = 0
        for _ in range(num_episodes):
            time_step = self.env.reset()
            episode_return = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == self.player_id:
                    agent_output = self.step(time_step)
                    action = agent_output.action
                else:
                    action = np.random.choice(time_step.observations["legal_actions"][player_id])
                time_step = self.env.step([action])
                episode_return += time_step.rewards[self.player_id]
            total_return += episode_return
        return total_return / num_episodes

    def __del__(self):
        if hasattr(self, 'sess'):
            self.sess.close()

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions."""
        action_string = state.action_to_string(player_id, action)
        print(f"Player {player_id} took action: {action_string}")

    def restart(self):
        """Reset the agent's state for a new game."""
        # Reset any necessary state variables
        # For example, you might want to clear any game-specific memory
        if hasattr(self.agent, '_prev_timestep'):
            self.agent._prev_timestep = None
        if hasattr(self.agent, '_prev_action'):
            self.agent._prev_action = None