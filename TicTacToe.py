import json
import os
import random
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


class Game:
    def __init__(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'X'

    def play(self, row, col, player):
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            if self.is_winner(player):
                return self.board, 25 if player == 'X' else -25, True
            if self.is_draw():
                return self.board, 0, True
            return self.board, 0, False
        return self.board, INVALID_MOVE_PENALTY, False

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

    def is_draw(self):
        return all([cell != ' ' for row in self.board for cell in row])

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

    def get_board(self):
        return self.board

    def __str__(self):
        return "\n".join(['|'.join(row) for row in self.board])


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
    

def human_vs_ai(agent):
    game = Game()
    player = 'X'
    done = False

    while not done:
        game.print_board()

        if player == 'X':
            action = int(input("Enter your move (1-9): ")) - 1
            _, _, done = game.play(action // 3, action % 3, player)
            if done:
                break
        else:
            action = agent.choose_action(game.get_board())
            _, _, done = game.play(action // 3, action % 3, player)

        player = 'O' if player == 'X' else 'X'

    game.print_board()

    if game.is_winner('X'):
        print("Congratulations! You won!")
    elif game.is_winner('O'):
        print("TicTacBot has won.")
    else:
        print("It's a draw!")
    
def load_or_create_model(model_path, metadata_path):
    if os.path.exists(model_path):
        model = Model()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with open(metadata_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
    else:
        model = Model()
        metadata = {"name": "TicTacBot", "episodes_trained": 0, "epsilon": 1.0}
        
        # Save initial metadata when creating a new model
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
        print("2. Play against TicTacBot")
        print("3. Reset the TicTacBot agent")
        print("4. Quit")
        option = int(input("Command: "))

        if option == 1:
            n_sets = int(input("Enter the number of sets of 100 episodes to train: "))
            episodes = n_sets * 100
            trained_agent, metadata = train_ai_agent(agent, metadata, episodes)
            save_model(agent.model, MODEL_PATH)
            save_model_metadata(metadata, MODEL_METADATA_PATH)
        elif option == 2:
            human_vs_ai(agent)
        elif option == 3:
            os.remove(MODEL_PATH)
            os.remove(MODEL_METADATA_PATH)
            model, metadata = load_or_create_model(MODEL_PATH, MODEL_METADATA_PATH)
            agent = Agent(model, epsilon=metadata["epsilon"])
        elif option == 4:
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
