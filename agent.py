import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
MEMORY_SIZE = 2000

class Agent:
    '''
    class Agent:
        A class that implements an agent for playing Tic Tac Toe using Q-learning.

    Attributes:
        model (nn.Module): A PyTorch neural network used to approximate the Q-function.
        learning_rate (float): The learning rate used by the Adam optimizer for updating the model weights.
        discount_factor (float): The discount factor used for future rewards in the Q-value update.
        epsilon (float): The probability of choosing a random action during training.
        epsilon_min (float): The minimum value of epsilon during training.
        epsilon_decay (float): The decay rate of epsilon after each training episode.
        batch_size (int): The number of transitions sampled from the replay buffer for each training step.
        memory (deque): A replay buffer for storing past state, action, reward, next state, and done tuples.

    Methods:
        update_target_model():
            Copies the weights of the model to the target model used for calculating target Q-values.
        
        remember(state, action, reward, next_state, done):
            Adds a new transition to the replay buffer.
        
        choose_action(state):
            Chooses an action to take for the given state using an epsilon-greedy policy.
        
        learn():
            Updates the model weights using a batch of transitions sampled from the replay buffer.
        
        _preprocess_state(state):
            Converts the state representation from a list of lists to a flattened list of -1, 0, or 1 values.
    '''
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
    """
    Class Model:
        A neural network model that predicts the Q-values for each action given a state.

    Attributes:
        model (nn.Sequential): The neural network model.

    Methods:
        forward(x): Predicts the Q-values for each action given the input state x.
    """
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
        """
        Method forward(x):
            Predicts the Q-values for each action given the input state x.

        Args:
            x (torch.Tensor): A tensor representing the input state.

        Returns:
            A tensor of Q-values for each action given the input state.
        """
        return self.model(x)

    
if __name__ == '__main__':
    print(Agent.__doc__)
    print()
    print(Model.__doc__)
    print(Model.forward.__doc__)
