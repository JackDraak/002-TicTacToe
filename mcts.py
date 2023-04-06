import copy
import time
import math


class MCTS:
    '''
    Class MCTS:
        A class that implements the Monte Carlo Tree Search algorithm for playing games.

    Attributes:
        exploration_param (float): The exploration parameter used in the UCB1 formula.
        time_limit (float): The maximum time in seconds that the algorithm is allowed to run.
        
    Methods:
        backpropagate(node, reward):
            Propagates the results of a simulation back up the tree.
            
        choose_action(game, player=None, num_simulations=1000):
            Chooses the best action to take based on the Monte Carlo Tree Search algorithm.
            
        expand_node(node):
            Expands the given node by adding children.
            
        select_node(node):
            Selects the best node to explore based on the UCB1 formula.
            
        run_simulation(root, player):
            Runs a simulation from the given node and updates the tree with the results.
            
        simulate(node):
            Simulates a game from the given node to the end and returns the result.
            
        ucb1(node):
            Calculates the UCB1 score for the given node.
    '''
    def __init__(self, exploration_param=1, time_limit=1):
        self.exploration_param = exploration_param
        self.time_limit = time_limit
    
    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.wins += reward
            node = node.parent
            reward = -reward

    def choose_action(self, game, player=None, num_simulations=1000):
        root = Node(game=game, parent=None)
        start_time = time.time()
        while (time.time() - start_time) < self.time_limit:
            self.run_simulation(root, game.current_player)
        if not root.children:
            raise RuntimeError("No available actions in the current game state.")
        best_node = max(root.children, key=lambda node: node.visits)
        return best_node.action

    def expand_node(self, node):
        node.expand()

    def select_node(self, node):
        while node.children:
            node = max(node.children, key=lambda child: self.ucb1(child))
        return node
    
    def run_simulation(self, root, player):
        node = self.select_node(root)
        self.expand_node(node)
        reward = self.simulate(node)
        self.backpropagate(node, reward)

    def simulate(self, node):
        game = copy.deepcopy(node.game)
        while not game.is_draw() and not game.is_winner(game.current_player):
            game.random_play(game.current_player)
        if game.is_draw():
            return 0
        elif game.is_winner(node.game.current_player):
            return -1
        else:
            return 1

    def ucb1(self, node):
        if node.visits == 0:
            return float('inf')
        return (node.wins / node.visits) + self.exploration_param * math.sqrt(math.log(node.parent.visits) / node.visits)
  

class Node:
    """
    Class Node:
        A class that represents a node in the MCTS search tree.

    Attributes:
        game (Game): The game state corresponding to this node.
        parent (Node): The parent node of this node in the search tree.
        action (int): The action that led to this node.
        children (list): A list of child nodes of this node.
        wins (int): The number of times this node was part of a winning simulation.
        visits (int): The total number of times this node was visited during simulations.

    Methods:
        expand():
            Expands the node by generating child nodes corresponding to all possible actions from the current game state.
    """
    def __init__(self, game, parent=None, action=None):
        self.game = game
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0

    def expand(self):
        """
        Method expand():
            Expands the node by generating child nodes corresponding to all possible actions from the current game state.
        """
        if not self.children:
            for action in self.game.get_actions():
                new_game = copy.deepcopy(self.game)
                player = new_game.current_player
                new_game.play(action // 3, action % 3, player)
                new_node = Node(new_game, parent=self, action=action)
                self.children.append(new_node)

if __name__ == '__main__':
    print(MCTS.__doc__)
    print()
    print(Node.__doc__)
    print(Node.expand.__doc__)
