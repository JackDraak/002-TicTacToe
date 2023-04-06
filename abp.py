import copy

class AlphaBetaPruning:  
    '''
    Class AlphaBetaPrunung:
        A class that implements the Alpha-Beta Pruning algorithm for playing Tic Tac Toe.

    Attributes:
        depth_limit (int): The maximum depth that the search algorithm can reach.
        
    Methods:
        alpha_beta_search(game, player, depth): 
            Conducts an alpha-beta search to determine the best move for the given player.
            
        choose_action(game, player):
            Chooses the best action to take for the given player using alpha-beta search.
            
        evaluate(game, player):
            Evaluates the given game board for the specified player and returns a score.
            
        get_actions(game):
            Returns a list of all possible actions that can be taken in the given game state.
            
        max_value(game, player, alpha, beta, depth):
            Conducts a max-value search to determine the highest possible score for the given player.
            
        min_value(game, player, alpha, beta, depth):
            Conducts a min-value search to determine the lowest possible score for the given player.
    '''
    def __init__(self, depth_limit=5):
        self.depth_limit = depth_limit

    def alpha_beta_search(self, game, player, depth):
        return self.max_value(game, player, -float('inf'), float('inf'), depth)

    def choose_action(self, game, player):
        _, action = self.alpha_beta_search(game, player, self.depth_limit)
        return action

    def evaluate(self, game, player):
        if game.is_winner(player):
            return 1
        elif game.is_winner('X' if player == 'O' else 'O'):
            return -1
        else:
            return 0

    def get_actions(self, game):
        return [i for i in range(9) if game.get_board()[i // 3][i % 3] == ' ']

    def max_value(self, game, player, alpha, beta, depth):
        if depth == 0 or game.is_winner(player) or game.is_winner('X' if player == 'O' else 'O') or game.is_draw():
            return self.evaluate(game, player), None
        value = -float('inf')
        best_action = None
        for action in self.get_actions(game):
            new_game = copy.deepcopy(game)
            new_game.play(action // 3, action % 3, player)
            temp_value, _ = self.min_value(new_game, 'X' if player == 'O' else 'O', alpha, beta, depth - 1)
            if temp_value > value:
                value = temp_value
                best_action = action
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_action

    def min_value(self, game, player, alpha, beta, depth):
        if depth == 0 or game.is_winner(player) or game.is_winner('X' if player == 'O' else 'O') or game.is_draw():
            return self.evaluate(game, player), None
        value = float('inf')
        best_action = None
        for action in self.get_actions(game):
            new_game = copy.deepcopy(game)
            new_game.play(action // 3, action % 3, player)
            temp_value, _ = self.max_value(new_game, 'X' if player == 'O' else 'O', alpha, beta, depth - 1)
            if temp_value < value:
                value = temp_value
                best_action = action
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_action
    
    
if __name__ == '__main__':
    print(AlphaBetaPruning.__doc__)
