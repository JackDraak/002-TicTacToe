import json
import os
import random
import torch
from abp import AlphaBetaPruning
from agent import Agent, Model
from mcts import MCTS

MODEL_PATH = "model.pth"
MODEL_METADATA_PATH = "model_data.json"
BOARD_SIZE = 3
SEPARATOR_LINE = '-' * 5
REWARD_X_WIN = 50
REWARD_O_WIN = -50
INVALID_MOVE_PENALTY = -5

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
