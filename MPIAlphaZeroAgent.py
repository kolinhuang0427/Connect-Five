import numpy as np
print(np.__version__)
import time

import torch
print(torch.__version__)
print("Is CUDA available:", torch.cuda.is_available())

import torch.nn as nn
import torch.nn.functional as F
from send_email import send_email
import random
import math
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank (0 or 1 in our case)
size = comm.Get_size()  # Total number of processes

class ConnectFive:
    def __init__(self):
        self.row_count = 16
        self.column_count = 16
        self.win_length = 5
        self.action_size = self.row_count * self.column_count

    def __str__(self):
        return "ConnectFive"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        
        def count_consecutive(row, col, d_row, d_col):
            count = 0
            r, c = row, col
            while 0 <= r < self.row_count and 0 <= c < self.column_count and state[r, c] == player:
                count += 1
                r += d_row
                c += d_col
            return count

        # Check horizontal, vertical, and two diagonal directions
        return (
            count_consecutive(row, column, 0, 1) + count_consecutive(row, column, 0, -1) - 1 >= self.win_length
            or count_consecutive(row, column, 1, 0) + count_consecutive(row, column, -1, 0) - 1 >= self.win_length
            or count_consecutive(row, column, 1, 1) + count_consecutive(row, column, -1, -1) - 1 >= self.win_length
            or count_consecutive(row, column, 1, -1) + count_consecutive(row, column, -1, 1) - 1 >= self.win_length
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state
    
    def change_perspective(self, state, player):
        return state * player

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
        
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)
        
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    node.backpropagate(value)
                    
                else:
                    spg.node = node
                    
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
                    
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()
                
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)

class AlphaZeroParallel:
    def __init__(self, model, optimizer, scheduler, game, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        
    def selfPlay(self):
        return_memory = []
        player = 1
        total_games = self.args['num_parallel_games']
        
        # Split games across processes
        local_num_games = total_games // size
        spGames = [SPG(self.game) for _ in range(local_num_games)]
        
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)
            
            # Perform MCTS search locally
            self.mcts.search(neutral_states, spGames)
            
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
                    
            player = self.game.get_opponent(player)
        
        # Gather results from all processes
        all_return_memory = comm.gather(return_memory, root=0)
    
        if rank == 0:
            # Combine data from all processes
            all_return_memory = sum(all_return_memory, [])
            result = all_return_memory  # Only rank 0 returns the combined memory
        else:
            result = []  # Other ranks return an empty list
        
        # Synchronize all processes
        comm.barrier()
        
        return result

                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            try:
                sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            except:
                 sample = memory[batchIdx:batchIdx+self.args['batch_size']]
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                start_time = time.time()
                memory += self.selfPlay()
                time_used = time.time() - start_time
                if rank == 0:
                    print(f"Game {selfPlay_iteration * self.args['num_parallel_games']}-{(selfPlay_iteration+1) * self.args['num_parallel_games']} Time Elapsed: {time_used:.4f}")
                    send_email(f"Game {selfPlay_iteration * self.args['num_parallel_games']}-{(selfPlay_iteration+1) * self.args['num_parallel_games']} Time Elapsed: {time_used:.4f}")
            
            if rank == 0:
                start_time = time.time()
                self.model.train()
                for epoch in range(self.args['num_epochs']):
                    self.train(memory)
                time_used = time.time() - start_time
                weights = self.model.state_dict()
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}_{timestamp}.pt")
                torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}_{timestamp}.pt")
                torch.save(self.scheduler.state_dict(), f"scheduler_{iteration}_{self.game}_{timestamp}.pt")
                print(f"iteration {iteration} done! Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB. Time Elapsed in training: {time_used:.4f}.")
                send_email(f"iteration {iteration} done! Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB. Time Elapsed in training: {time_used:.4f}.")
            else:
                weights = None
            weights = comm.bcast(weights, root=0)
            self.model.load_state_dict(weights)

class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

def main():
    game = ConnectFive()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 9, 128, device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    model_checkpoint_path = "model_5_ConnectFive_20241214-182158.pt"
    optimizer_checkpoint_path = "optimizer_5_ConnectFive_20241214-182158.pt"
    scheduler_checkpoint_path = "scheduler_5_ConnectFive_20241214-182158.pt"

    model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True))
    optimizer.load_state_dict(torch.load(optimizer_checkpoint_path, weights_only=True))
    scheduler.load_state_dict(torch.load(scheduler_checkpoint_path, weights_only=True))

    args = {
        'C': 2,
        'num_searches': 800,
        'num_iterations': 20,
        'num_selfPlay_iterations': 1000,
        'num_parallel_games': 250,
        'num_epochs': 5,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.03
    }

    alphaZero = AlphaZeroParallel(model,optimizer, scheduler, game, args)
    alphaZero.learn()

if __name__ == "__main__":
    main()