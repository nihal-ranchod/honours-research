"""Monte-Carlo Tree Search algorithm for game play."""

import math
import time
from collections import defaultdict
import numpy as np
import pyspiel
import chess.pgn


class Evaluator(object):
  """Abstract class representing an evaluation function for a game.

  The evaluation function takes in an intermediate state in the game and returns
  an evaluation of that state, which should correlate with chances of winning
  the game. It returns the evaluation from all player's perspectives.
  """

  def evaluate(self, state):
    """Returns evaluation on given state."""
    raise NotImplementedError

  def prior(self, state):
    """Returns a probability for each legal action in the given state."""
    raise NotImplementedError


class RandomRolloutEvaluator(Evaluator):
  """A simple evaluator doing random rollouts.

  This evaluator returns the average outcome of playing random actions from the
  given state until the end of the game.  n_rollouts is the number of random
  outcomes to be considered.
  """

  def __init__(self, n_rollouts=1, random_state=None):
    self.n_rollouts = n_rollouts
    self._random_state = random_state or np.random.RandomState()

  def evaluate(self, state):
    """Returns evaluation on given state."""
    result = None
    for _ in range(self.n_rollouts):
      working_state = state.clone()
      while not working_state.is_terminal():
        if working_state.is_chance_node():
          outcomes = working_state.chance_outcomes()
          action_list, prob_list = zip(*outcomes)
          action = self._random_state.choice(action_list, p=prob_list)
        else:
          action = self._random_state.choice(working_state.legal_actions())
        working_state.apply_action(action)
      returns = np.array(working_state.returns())
      result = returns if result is None else result + returns

    return result / self.n_rollouts

  def prior(self, state):
    """Returns equal probability for all actions."""
    if state.is_chance_node():
      return state.chance_outcomes()
    else:
      legal_actions = state.legal_actions(state.current_player())
      return [(action, 1.0 / len(legal_actions)) for action in legal_actions]


class SearchNode(object):
  """A node in the search tree.

  A SearchNode represents a state and possible continuations from it. Each child
  represents a possible action, and the expected result from doing so.

  Attributes:
    action: The action from the parent node's perspective. Not important for the
      root node, as the actions that lead to it are in the past.
    player: Which player made this action.
    prior: A prior probability for how likely this action will be selected.
    explore_count: How many times this node was explored.
    total_reward: The sum of rewards of rollouts through this node, from the
      parent node's perspective. The average reward of this node is
      `total_reward / explore_count`
    outcome: The rewards for all players if this is a terminal node or the
      subtree has been proven, otherwise None.
    children: A list of SearchNodes representing the possible actions from this
      node, along with their expected rewards.
  """
  __slots__ = [
      "action",
      "player",
      "prior",
      "explore_count",
      "total_reward",
      "outcome",
      "children",
  ]

  def __init__(self, action, player, prior):
    self.action = action
    self.player = player
    self.prior = prior
    self.explore_count = 0
    self.total_reward = 0.0
    self.outcome = None
    self.children = []

  def uct_value(self, parent_explore_count, uct_c):
    """Returns the UCT value of child."""
    if self.outcome is not None:
      return self.outcome[self.player]

    if self.explore_count == 0:
      return float("inf")

    return self.total_reward / self.explore_count + uct_c * math.sqrt(
        math.log(parent_explore_count) / self.explore_count)

  def puct_value(self, parent_explore_count, uct_c):
    """Returns the PUCT value of child."""
    if self.outcome is not None:
      return self.outcome[self.player]

    return ((self.explore_count and self.total_reward / self.explore_count) +
            uct_c * self.prior * math.sqrt(parent_explore_count) /
            (self.explore_count + 1))

  def sort_key(self):
    """Returns the best action from this node, either proven or most visited.

    This ordering leads to choosing:
    - Highest proven score > 0 over anything else, including a promising but
      unproven action.
    - A proven draw only if it has higher exploration than others that are
      uncertain, or the others are losses.
    - Uncertain action with most exploration over loss of any difficulty
    - Hardest loss if everything is a loss
    - Highest expected reward if explore counts are equal (unlikely).
    - Longest win, if multiple are proven (unlikely due to early stopping).
    """
    return (0 if self.outcome is None else self.outcome[self.player],
            self.explore_count, self.total_reward)

  def best_child(self):
    """Returns the best child in order of the sort key."""
    return max(self.children, key=SearchNode.sort_key)

  def children_str(self, state=None):
    """Returns the string representation of this node's children.

    They are ordered based on the sort key, so order of being chosen to play.

    Args:
      state: A `pyspiel.State` object, to be used to convert the action id into
        a human readable format. If None, the action integer id is used.
    """
    return "\n".join([
        c.to_str(state)
        for c in reversed(sorted(self.children, key=SearchNode.sort_key))
    ])

  def to_str(self, state=None):
    """Returns the string representation of this node.

    Args:
      state: A `pyspiel.State` object, to be used to convert the action id into
        a human readable format. If None, the action integer id is used.
    """
    action = (
        state.action_to_string(state.current_player(), self.action)
        if state and self.action is not None else str(self.action))
    return ("{:>6}: player: {}, prior: {:5.3f}, value: {:6.3f}, sims: {:5d}, "
            "outcome: {}, {:3d} children").format(
                action, self.player, self.prior, self.explore_count and
                self.total_reward / self.explore_count, self.explore_count,
                ("{:4.1f}".format(self.outcome[self.player])
                 if self.outcome else "none"), len(self.children))

  def __str__(self):
    return self.to_str(None)


class MCTSBot(pyspiel.Bot):
  """Bot that uses Monte-Carlo Tree Search algorithm."""

  def __init__(self,
               game,
               uct_c,
               max_simulations,
               evaluator,
               solve=True,
               random_state=None,
               child_selection_fn=SearchNode.uct_value,
               dirichlet_noise=None,
               verbose=False,
               dont_return_chance_node=False):
    """Initializes a MCTS Search algorithm in the form of a bot.

    In multiplayer games, or non-zero-sum games, the players will play the
    greedy strategy.

    Args:
      game: A pyspiel.Game to play.
      uct_c: The exploration constant for UCT.
      max_simulations: How many iterations of MCTS to perform. Each simulation
        will result in one call to the evaluator. Memory usage should grow
        linearly with simulations * branching factor. How many nodes in the
        search tree should be evaluated. This is correlated with memory size and
        tree depth.
      evaluator: A `Evaluator` object to use to evaluate a leaf node.
      solve: Whether to back up solved states.
      random_state: An optional numpy RandomState to make it deterministic.
      child_selection_fn: A function to select the child in the descent phase.
        The default is UCT.
      dirichlet_noise: A tuple of (epsilon, alpha) for adding dirichlet noise to
        the policy at the root. This is from the alpha-zero paper.
      verbose: Whether to print information about the search tree before
        returning the action. Useful for confirming the search is working
        sensibly.
      dont_return_chance_node: If true, do not stop expanding at chance nodes.
        Enabled for AlphaZero.

    Raises:
      ValueError: if the game type isn't supported.
    """
    pyspiel.Bot.__init__(self)
    # Check that the game satisfies the conditions for this MCTS implemention.
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
      raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("Game must have sequential turns.")

    self._game = game
    self.uct_c = uct_c
    self.max_simulations = max_simulations
    self.evaluator = evaluator
    self.verbose = verbose
    self.solve = solve
    self.max_utility = game.max_utility()
    self._dirichlet_noise = dirichlet_noise
    self._random_state = random_state or np.random.RandomState()
    self._child_selection_fn = child_selection_fn
    self.dont_return_chance_node = dont_return_chance_node

  def restart_at(self, state):
    pass

  def step_with_policy(self, state):
    """Returns bot's policy and action at given state."""
    t1 = time.time()
    root = self.mcts_search(state)

    best = root.best_child()

    if self.verbose:
      seconds = time.time() - t1
      print("Finished {} sims in {:.3f} secs, {:.1f} sims/s".format(
          root.explore_count, seconds, root.explore_count / seconds))
      print("Root:")
      print(root.to_str(state))
      print("Children:")
      print(root.children_str(state))
      if best.children:
        chosen_state = state.clone()
        chosen_state.apply_action(best.action)
        print("Children of chosen:")
        print(best.children_str(chosen_state))

    mcts_action = best.action

    policy = [(action, (1.0 if action == mcts_action else 0.0))
              for action in state.legal_actions(state.current_player())]

    return policy, mcts_action

  def step(self, state):
    return self.step_with_policy(state)[1]

  def _apply_tree_policy(self, root, state):
    """Applies the UCT policy to play the game until reaching a leaf node.

    A leaf node is defined as a node that is terminal or has not been evaluated
    yet. If it reaches a node that has been evaluated before but hasn't been
    expanded, then expand it's children and continue.

    Args:
      root: The root node in the search tree.
      state: The state of the game at the root node.

    Returns:
      visit_path: A list of nodes descending from the root node to a leaf node.
      working_state: The state of the game at the leaf node.
    """
    visit_path = [root]
    working_state = state.clone()
    current_node = root
    while (not working_state.is_terminal() and
           current_node.explore_count > 0) or (
               working_state.is_chance_node() and self.dont_return_chance_node):
      if not current_node.children:
        # For a new node, initialize its state, then choose a child as normal.
        legal_actions = self.evaluator.prior(working_state)
        if current_node is root and self._dirichlet_noise:
          epsilon, alpha = self._dirichlet_noise
          noise = self._random_state.dirichlet([alpha] * len(legal_actions))
          legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                           for (a, p), n in zip(legal_actions, noise)]
        # Reduce bias from move generation order.
        self._random_state.shuffle(legal_actions)
        player = working_state.current_player()
        current_node.children = [
            SearchNode(action, player, prior) for action, prior in legal_actions
        ]

      if working_state.is_chance_node():
        # For chance nodes, rollout according to chance node's probability
        # distribution
        outcomes = working_state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = self._random_state.choice(action_list, p=prob_list)
        chosen_child = next(
            c for c in current_node.children if c.action == action)
      else:
        # Otherwise choose node with largest UCT value
        chosen_child = max(
            current_node.children,
            key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                c, current_node.explore_count, self.uct_c))

      working_state.apply_action(chosen_child.action)
      current_node = chosen_child
      visit_path.append(current_node)

    return visit_path, working_state

  def mcts_search(self, state):
    """A vanilla Monte-Carlo Tree Search algorithm.

    This algorithm searches the game tree from the given state.
    At the leaf, the evaluator is called if the game state is not terminal.
    A total of max_simulations states are explored.

    At every node, the algorithm chooses the action with the highest PUCT value,
    defined as: `Q/N + c * prior * sqrt(parent_N) / N`, where Q is the total
    reward after the action, and N is the number of times the action was
    explored in this position. The input parameter c controls the balance
    between exploration and exploitation; higher values of c encourage
    exploration of under-explored nodes. Unseen actions are always explored
    first.

    At the end of the search, the chosen action is the action that has been
    explored most often. This is the action that is returned.

    This implementation supports sequential n-player games, with or without
    chance nodes. All players maximize their own reward and ignore the other
    players' rewards. This corresponds to max^n for n-player games. It is the
    norm for zero-sum games, but doesn't have any special handling for
    non-zero-sum games. It doesn't have any special handling for imperfect
    information games.

    The implementation also supports backing up solved states, i.e. MCTS-Solver.
    The implementation is general in that it is based on a max^n backup (each
    player greedily chooses their maximum among proven children values, or there
    exists one child whose proven value is game.max_utility()), so it will work
    for multiplayer, general-sum, and arbitrary payoff games (not just win/loss/
    draw games). Also chance nodes are considered proven only if all children
    have the same value.

    Some references:
    - Sturtevant, An Analysis of UCT in Multi-Player Games,  2008,
      https://web.cs.du.edu/~sturtevant/papers/multi-player_UCT.pdf
    - Nijssen, Monte-Carlo Tree Search for Multi-Player Games, 2013,
      https://project.dke.maastrichtuniversity.nl/games/files/phd/Nijssen_thesis.pdf
    - Silver, AlphaGo Zero: Starting from scratch, 2017
      https://deepmind.com/blog/article/alphago-zero-starting-scratch
    - Winands, Bjornsson, and Saito, "Monte-Carlo Tree Search Solver", 2008.
      https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf

    Arguments:
      state: pyspiel.State object, state to search from

    Returns:
      The most visited move from the root node.
    """
    root = SearchNode(None, state.current_player(), 1)
    for _ in range(self.max_simulations):
      visit_path, working_state = self._apply_tree_policy(root, state)
      if working_state.is_terminal():
        returns = working_state.returns()
        visit_path[-1].outcome = returns
        solved = self.solve
      else:
        returns = self.evaluator.evaluate(working_state)
        solved = False

      while visit_path:
        # For chance nodes, walk up the tree to find the decision-maker.
        decision_node_idx = -1
        while visit_path[decision_node_idx].player == pyspiel.PlayerId.CHANCE:
          decision_node_idx -= 1
        # Chance node targets are for the respective decision-maker.
        target_return = returns[visit_path[decision_node_idx].player]
        node = visit_path.pop()
        node.total_reward += target_return
        node.explore_count += 1

        if solved and node.children:
          player = node.children[0].player
          if player == pyspiel.PlayerId.CHANCE:
            # Only back up chance nodes if all have the same outcome.
            # An alternative would be to back up the weighted average of
            # outcomes if all children are solved, but that is less clear.
            outcome = node.children[0].outcome
            if (outcome is not None and
                all(np.array_equal(c.outcome, outcome) for c in node.children)):
              node.outcome = outcome
            else:
              solved = False
          else:
            # If any have max utility (won?), or all children are solved,
            # choose the one best for the player choosing.
            best = None
            all_solved = True
            for child in node.children:
              if child.outcome is None:
                all_solved = False
              elif best is None or child.outcome[player] > best.outcome[player]:
                best = child
            if (best is not None and
                (all_solved or best.outcome[player] == self.max_utility)):
              node.outcome = best.outcome
            else:
              solved = False
      if root.outcome is not None:
        break

    return root

class EnhancedSearchNode(SearchNode):
    """Extended SearchNode that can store state information."""
    
    __slots__ = SearchNode.__slots__ + ['state']
    
    def __init__(self, action, player, prior):
        super().__init__(action, player, prior)
        self.state = None

class MCTS_with_PGN_Data(MCTSBot):
    """Enhanced MCTS algorithm that incorporates learning from historical chess games."""

    def __init__(
        self,
        game,
        uct_c,
        max_simulations,
        evaluator,
        training_data,
        random_state=None,
        solve=False,
        verbose=False,
        cache_size=1000000,
        min_position_visits=5
    ):
        super().__init__(
            game,
            uct_c,
            max_simulations, 
            evaluator,
            random_state=random_state,
            solve=solve,
            verbose=verbose
        )
        
        self.position_stats = defaultdict(lambda: defaultdict(int))
        self.eval_cache = {}
        self.opening_book = {}
        self.progressive_history = defaultdict(lambda: defaultdict(float))
        self.cache_size = cache_size
        self.min_position_visits = min_position_visits
        
        self._load_training_data(training_data)

    def _load_training_data(self, pgn_file):
        """Load and process historical games from PGN file."""
        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                board = game.board()
                outcome = game.headers.get("Result", "*")
                
                for move in game.mainline_moves():
                    position_key = self._get_state_key_from_board(board)
                    
                    self.position_stats[position_key][move.uci()] += 1
                    
                    if board.fullmove_number <= 15:
                        if position_key not in self.opening_book:
                            self.opening_book[position_key] = []
                        self.opening_book[position_key].append(move.uci())
                    
                    if outcome == "1-0":
                        self._update_progressive_history(move.uci(), 1.0)
                    elif outcome == "0-1":
                        self._update_progressive_history(move.uci(), 0.0)
                    elif outcome == "1/2-1/2":
                        self._update_progressive_history(move.uci(), 0.5)
                        
                    board.push(move)

    def _get_state_key_from_board(self, board):
        """Get position key from a python-chess board."""
        return board.fen().split(' ')[0]

    def _get_state_key(self, state):
        """Get position key from an OpenSpiel state."""
        return state.observation_string(state.current_player())

    def _update_progressive_history(self, move, outcome, learning_rate=0.1):
        """Update the progressive history scores for a move."""
        current_value = self.progressive_history['all'][move]
        self.progressive_history['all'][move] = (
            current_value + learning_rate * (outcome - current_value)
        )

    def _get_prior_policy(self, state):
        """Get enhanced prior move probabilities using historical data."""
        legal_actions = state.legal_actions()
        state_key = self._get_state_key(state)
        
        prior_probs = np.ones(len(legal_actions)) / len(legal_actions)
        
        if state_key in self.opening_book:
            book_moves = set(self.opening_book[state_key])
            for i, action in enumerate(legal_actions):
                move_uci = self._action_to_uci(state, action)
                if move_uci in book_moves:
                    prior_probs[i] *= 2.0
                    
        pos_total = sum(self.position_stats[state_key].values())
        if pos_total >= self.min_position_visits:
            for i, action in enumerate(legal_actions):
                move_uci = self._action_to_uci(state, action)
                count = self.position_stats[state_key][move_uci]
                if count > 0:
                    prior_probs[i] *= (count / pos_total)
                    
        for i, action in enumerate(legal_actions):
            move_uci = self._action_to_uci(state, action)
            history_score = self.progressive_history['all'][move_uci]
            if history_score > 0:
                prior_probs[i] *= (1.0 + history_score)
                
        prior_sum = prior_probs.sum()
        if prior_sum > 0:
            prior_probs /= prior_sum
        else:
            prior_probs = np.ones(len(legal_actions)) / len(legal_actions)
        
        return list(zip(legal_actions, prior_probs))

    def _action_to_uci(self, state, action):
        """Convert OpenSpiel action to UCI format."""
        return state.action_to_string(state.current_player(), action)

    def _select_child(self, node, state):
        """Enhanced child selection using both UCT and historical data."""
        if not node.children:
            return None
            
        best_value = float("-inf")
        best_child = None
        
        for child in node.children:
            uct_score = child.uct_value(node.explore_count, self.uct_c)
            
            move_uci = self._action_to_uci(state, child.action)
            history_bonus = self.progressive_history['all'][move_uci]
            
            total_score = uct_score + 0.1 * history_bonus
            
            if total_score > best_value:
                best_value = total_score
                best_child = child
                
        return best_child

    def step_with_policy(self, state):
        """Enhanced version of step_with_policy using historical data."""
        state_key = self._get_state_key(state)
        if state_key in self.eval_cache:
            cached_policy, cached_value = self.eval_cache[state_key]
            if cached_value > 0.9:
                return cached_policy, max(cached_policy, key=lambda x: x[1])[0]
        
        root = self.mcts_search(state)
        
        best_child = root.best_child()
        mcts_action = best_child.action
        
        total_visits = sum(c.explore_count for c in root.children)
        policy = [
            (c.action, c.explore_count / total_visits)
            for c in root.children
        ]
        
        if len(self.eval_cache) < self.cache_size:
            self.eval_cache[state_key] = (policy, best_child.total_reward / best_child.explore_count)
            
        return policy, mcts_action

    def mcts_search(self, state):
        """Enhanced MCTS search incorporating historical data."""
        root = EnhancedSearchNode(None, state.current_player(), 1)
        root.state = state.clone()
        
        prior_policy = self._get_prior_policy(state)
        root.children = [
            EnhancedSearchNode(action, state.current_player(), prior)
            for action, prior in prior_policy
        ]
        
        for _ in range(self.max_simulations):
            visit_path, working_state = self._apply_tree_policy(root, state)
            if working_state.is_terminal():
                returns = working_state.returns()
                visit_path[-1].outcome = returns
                solved = self.solve
            else:
                returns = self.evaluator.evaluate(working_state)
                solved = False
                
            self._backup(visit_path, returns, solved)
            
            if root.outcome is not None:
                break
                
        return root

    def _backup(self, visit_path, returns, solved):
        """Enhanced backup procedure incorporating progressive history updates."""
        while visit_path:
            node = visit_path.pop()
            
            node.total_reward += returns[node.player]
            node.explore_count += 1
            
            if solved and node.children:
                if self._check_solved_state(node, returns):
                    node.outcome = returns
                else:
                    solved = False
                    
            if node.action is not None and node.state is not None:
                move_uci = self._action_to_uci(node.state, node.action)
                self._update_progressive_history(
                    move_uci,
                    node.total_reward / node.explore_count
                )

    def _check_solved_state(self, node, returns):
        """Check if a node should be marked as solved."""
        if not node.children:
            return True
            
        player = node.player
        if player == pyspiel.PlayerId.CHANCE:
            first_outcome = node.children[0].outcome
            return all(
                child.outcome is not None and
                np.array_equal(child.outcome, first_outcome)
                for child in node.children
            )
        else:
            best_outcome = None
            all_solved = True
            
            for child in node.children:
                if child.outcome is None:
                    all_solved = False
                elif (best_outcome is None or 
                      child.outcome[player] > best_outcome[player]):
                    best_outcome = child.outcome
                    
            return (best_outcome is not None and
                    (all_solved or best_outcome[player] == self.max_utility))

    def _apply_tree_policy(self, root, state):
        """Apply the UCT policy to play the game until reaching a leaf node."""
        visit_path = [root]
        working_state = state.clone()
        current_node = root
        
        while (not working_state.is_terminal() and
               current_node.explore_count > 0):
            
            if not current_node.children:
                legal_actions = self.evaluator.prior(working_state)
                player = working_state.current_player()
                current_node.children = [
                    EnhancedSearchNode(action, player, prior) 
                    for action, prior in legal_actions
                ]

            if working_state.is_chance_node():
                outcomes = working_state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = self._random_state.choice(action_list, p=prob_list)
                chosen_child = next(c for c in current_node.children if c.action == action)
            else:
                chosen_child = self._select_child(current_node, working_state)
                if chosen_child is None:
                    break

            chosen_child.state = working_state.clone()
            working_state.apply_action(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_state