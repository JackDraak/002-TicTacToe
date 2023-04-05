import copy
import json
import math
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

MODEL_PATH = "model.pth"
MODEL_METADATA_PATH = "model_data.json"
BOARD_SIZE = 3
SEPARATOR_LINE = '-' * 5
REWARD_X_WIN = 50
REWARD_O_WIN = -50
INVALID_MOVE_PENALTY = -5
MEMORY_SIZE = 2000

'''
The RL agent that learns to play the game using Q-Learning algorithm with a Deep Neural Network. 
It has methods for choosing an action, remembering the state, action, reward, next_state, and 
done tuple, and updating the model's weights through backpropagation.
'''
class Agent:
    def __init__(self, model, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64):
        self.model = model.to(device)
        self.target_model = Model().to(device) 
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=MEMORY_SIZE)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            return random.choice([i for i in range(9) if state[i // 3][i % 3] == ' '])
        else:
            state = torch.tensor(self._preprocess_state(state), dtype=torch.float32).to(device)
            q_values = self.model(state)
            state_reshaped = state.view(3, 3)
            valid_q_values = [(i, q_values[i].item()) for i in range(9) if state_reshaped[i // 3][i % 3] == 0]
            return max(valid_q_values, key=lambda x: x[1])[0]

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.tensor(self._preprocess_state(state), dtype=torch.float32).to(device)
            next_state = torch.tensor(self._preprocess_state(next_state), dtype=torch.float32).to(device)
            target = reward
            if not done:
                target += self.discount_factor * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state)
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f.detach())
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _preprocess_state(self, state):
        return [1 if cell == 'X' else -1 if cell == 'O' else 0 for row in state for cell in row]

'''
i.e. select the lowest numbered cell for the next play, dumb as a rock and 
doesn't even need all these layers of logic to accomplish....
'''
class AlphaBetaPruning:  
    def __init__(self, depth_limit=3):
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


class Game:
    def __init__(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'X'

    def __str__(self):
        return "\n".join(['|'.join(row) for row in self.board])

    def get_actions(self):
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    actions.append(i * 3 + j)
        return actions

    def get_board(self):
        return self.board

    def is_draw(self):
        return all([cell != ' ' for row in self.board for cell in row])

    def is_winner(self, player):
        for row in self.board:
            if all([cell == player for cell in row]):
                return True
        for col in range(3):
            if all([self.board[row][col] == player for row in range(3)]):
                return True
        if all([self.board[i][i] == player for i in range(3)]) or all([self.board[i][2 - i] == player for i in range(3)]):
            return True
        return False

    def play(self, row, col, player):
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            self.current_player = 'O' if player == 'X' else 'X'
            if self.is_winner(player):
                return self.board, 25 if player == 'X' else -25, True
            if self.is_draw():
                return self.board, 0, True
            return self.board, 0, False
        return self.board, INVALID_MOVE_PENALTY, False

    def print_board(self):
        print()
        for i, row in enumerate(self.board):
            formatted_row = [str((i * BOARD_SIZE) + j + 1) if cell == ' ' else cell for j, cell in enumerate(row)]
            print('|'.join(formatted_row))
            if i < BOARD_SIZE - 1:
                print(SEPARATOR_LINE)
                
    def random_play(self, player):
        valid_moves = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if self.board[i][j] == ' ']
        row, col = random.choice(valid_moves)
        return self.play(row, col, player)

'''
This is an opponent class that uses the Monte Carlo Tree Search algorithm to choose the next best move.
'''
class MCTS:
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

'''
This is the Deep Neural Network model used by the Agent.
'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.model(x)
  
'''
A node in the Monte Carlo Tree used by the MCTS algorithm.
'''
class Node:
    def __init__(self, game, parent=None, action=None):
        self.game = game
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0

    def expand(self):
        if not self.children:
            for action in self.game.get_actions():
                new_game = copy.deepcopy(self.game)
                player = new_game.current_player
                new_game.play(action // 3, action % 3, player)
                new_node = Node(new_game, parent=self, action=action)
                self.children.append(new_node)

    
def human_vs_ai(agent, opponent_type):
    game = Game()
    player = 'X'
    done = False
    opponent = AlphaBetaPruning() if opponent_type == 'ab_prune' else None
    while not done:
        game.print_board()
        if player == 'X':
            action = int(input("Enter your move (1-9): ")) - 1
            _, _, done = game.play(action // 3, action % 3, player)
            if done:
                break
        else:
            if opponent_type == 'ab_prune':                 # alpha beta pruning
                action = opponent.choose_action(game, 'O')
            elif opponent_type == 'mcts':                   # monte carlo tree search
                opponent = MCTS()
                action = opponent.choose_action(game)
            elif opponent_type == "rnn":                    # reinforcement learning neural network
                action = agent.choose_action(game.get_board())
                _, _, done = game.play(action // 3, action % 3, player)
            else:
                print("soemthing unexpected happened, sorry")
        player = 'O' if player == 'X' else 'X'
    game.print_board()
    if game.is_winner('X'):
        print("Congratulations! You won!")
    elif game.is_winner('O'):
        print(f"{opponent_type.capitalize()} has won.")
    else:
        print("It's a draw!")
    
def load_or_create_model(model_path, metadata_path):
    if os.path.exists(model_path):
        model = Model()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with open(metadata_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
    else:  # Save initial metadata when creating a new model
        model = Model()
        metadata = {"name": "TicTacBot", "episodes_trained": 0, "epsilon": 1.0} 
        with open(metadata_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
    return model, metadata

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def save_model_metadata(metadata, metadata_path):
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

def train_ai_agent(agent, metadata, episodes=500):
    for episode in range(episodes):
        game = Game()
        state = game.get_board()
        done = False
        steps = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = game.play(action // 3, action % 3, 'X')
            reward *= REWARD_X_WIN
            if not done:
                _, ai_reward, ai_done = game.random_play('O')
                ai_reward *= REWARD_O_WIN
                reward -= ai_reward
                done = ai_done
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            steps += 1
            agent.learn()
        if (episode + 1) % 50 == 0:
            agent.update_target_model()
            print(f"Episode: {episode + metadata['episodes_trained'] + 1}, Epsilon: {agent.epsilon:.7f}")
    metadata['episodes_trained'] += episodes
    metadata['epsilon'] = agent.epsilon
    return agent, metadata

def main():
    model, metadata = load_or_create_model(MODEL_PATH, MODEL_METADATA_PATH)
    agent = Agent(model, epsilon=metadata["epsilon"])
    while True:
        print()
        print(f"Agent {metadata['name']} (Episodes Trained: {metadata['episodes_trained']}, Epsilon: {metadata['epsilon']:.7f})")
        print("Select an option:")
        print("1. Train the TicTacBot agent for N*100 episodes")
        print("2. Play against an AI opponent...")
        print("3. Reset the RLNN-TicTacBot agent model (NOT REVERSIBLE)")
        print("4. Quit")
        option = int(input("Command: "))
        if option == 1:
            n_sets = int(input("Enter the number of sets of 100 episodes to train: "))
            episodes = n_sets * 100
            trained_agent, metadata = train_ai_agent(agent, metadata, episodes)
            save_model(agent.model, MODEL_PATH)
            save_model_metadata(metadata, MODEL_METADATA_PATH)
        elif option == 2:
            print("Choose your opponent:")
            print("1. The Grocer (ABP) Alpha Beta Pruning")
            print("2. Tree Master (AB-MCTS) Monte Carlo Tree")
            print("3. Tic Tac Bot (RLA-NN) Neural Network")
            opponent_option = int(input("Command: "))
            if opponent_option == 1:
                human_vs_ai(agent, 'ab_prune')
            elif opponent_option == 2:
                human_vs_ai(agent, 'mcts')
            elif opponent_option == 3:
                human_vs_ai(agent, "rnn")
            else:
                print("Invalid option. Please try again.")
        elif option == 3:
            if input("Are you certain you would like to wipe the trained RLNN model? (y for yes) ") == 'y':
                try:
                    os.remove(MODEL_PATH)
                except: print("- no model present")
                # reset the metadata dictionary
                metadata = {"name": "TicTacBot", "episodes_trained": 0, "epsilon": 1.0}
                save_model_metadata(metadata, MODEL_METADATA_PATH)
                model, metadata = load_or_create_model(MODEL_PATH, MODEL_METADATA_PATH)
                agent = Agent(model, epsilon=metadata["epsilon"])
            else:
                model, metadata = load_or_create_model(MODEL_PATH, MODEL_METADATA_PATH)
                agent = Agent(model, epsilon=metadata["epsilon"])
        elif option == 4:
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
