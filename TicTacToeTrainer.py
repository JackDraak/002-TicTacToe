import numpy as np
import torch
from TicTacToe import Game
import agent
import mcts

def rnn_vs_mcts(rnn_agent, mcts_agent):
    game = Game()
    player = 'X'
    done = False
    while not done:
        game.print_board()
        if player == 'X':
            action = rnn_agent.choose_action(game.get_board())
            _, _, done = game.play(action // 3, action % 3, player)
        else:
            action = mcts_agent.choose_action(game)
            _, _, done = game.play(action // 3, action % 3, player)
        if done:
            break
        player = 'O' if player == 'X' else 'X'
    
    game.print_board()
    if game.is_winner('X'):
        print("RNN agent has won.")
    elif game.is_winner('O'):
        print("MCTS agent has won.")
    else:
        print("It's a draw!")

class TicTacToeTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def calc_rewards(self, game, player):
        if game.is_winner(player):
            return 1
        elif game.is_winner('X' if player == 'O' else 'O'):
            return -1
        else:
            return 0
        
    def train(self, episodes, eval_interval):
        for episode in range(1, episodes + 1):
            rnn_agent = agent.Agent(self.model)  # Create the RNN agent
            rnn_player = 'X'
            mcts_player = 'O'
            game = Game()
            done = False
            states, actions, rewards = [], [], []

            while not done:
                state = game.get_board()

                if game.player == rnn_player:
                    action = rnn_agent.choose_action(state)  # Call choose_action on rnn_agent
                    _, _, done = game.play(action // 3, action % 3, rnn_player)
                else:
                    mcts_agent = mcts.MCTS()
                    action = mcts_agent.choose_action(game)
                    _, _, done = game.play(action // 3, action % 3, mcts_player)

                states.append(state)
                actions.append(action)
                rewards.append(self.calc_rewards(game, rnn_player))

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)

            pred_actions = self.model(states)
            target_actions = pred_actions.clone().detach()
            target_actions[np.arange(len(actions)), actions] = rewards

            loss = self.loss_fn(pred_actions, target_actions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
            if episode % eval_interval == 0:
                print(f"Episode: {episode}, Loss: {loss.item()}")
                rnn_agent = agent.Agent(self.model)
                mcts_agent = mcts.MCTS()
                rnn_vs_mcts(rnn_agent, mcts_agent)

def play_100_episodes():
    # Create the model and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # Create and train the TicTacToeTrainer
    trainer = TicTacToeTrainer(model, optimizer, loss_fn)
    trainer.train(episodes=100, eval_interval=20)



if __name__ == "__main__":
    # Create the RNN and MCTS agents
    model = agent.Model()
    rnn_agent = agent.Agent(model)
    mcts_agent = mcts.MCTS()

    # Print agent roles
    print("RNN agent is playing as 'X'")
    print("MCTS agent is playing as 'O'")

    # Simulate one complete game between MCTS and RNN-agent
    rnn_vs_mcts(rnn_agent, mcts_agent)
