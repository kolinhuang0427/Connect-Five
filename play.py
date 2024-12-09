from connect_five import *
from miniMaxAgent import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        


def play():
    gameState = GameState()
    ai = MinimaxAgent()
    '''
    if len(sys.argv) > 1:
    # Print command-line arguments
        print("Command-line arguments:", sys.argv[1:])
    else:
        print("No command-line arguments provided.")
    '''
    while gameState.isLose() == False and gameState.isWin() == False:
        user_input = input("What is your move? enter x,y where 0<=x,y<=15:")
        a = user_input.split(',')
        try:
            user_input = int(a[0].strip()), int(a[1].strip())
            if not (0 <= user_input[0] <= 15 and 0 <= user_input[1] <= 15):
                raise ValueError
        except (ValueError, IndexError):
            print("Invalid input, please enter x,y where 0<=x,y<=15.")
            continue

        print("your move is ",user_input)
        while not isinstance(user_input,tuple): 
            user_input = input("Enter x,y where 0<=x,y<=15:")
        gameState = gameState.generateSuccessor(-1,user_input)
        if gameState.isWin() : 
            print("You won haha")
            break
        action = ai.getAction(gameState)
        print("the ai's move is:", action)
        gameState = gameState.generateSuccessor(1,action)
        gameState.data.printGrid()
        if gameState.isLose() : 
            print("You lose haha")
            break
        

#play()

def botmatch():
    game = ConnectFive()
    player = -1
    gameState = GameState()
    minimax = MinimaxAgent()

    args = {
        'C': 2,
        'num_searches': 800,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load("model_0_ConnectFive_20241209-085238.pt", map_location=device))
    model.eval()

    mcts = MCTS(game, args, model)

    while True:
        
        if player == -1:
            neutral_state = np.array(gameState.data.data)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
            print("alphazero action:", action)
            row = (action-1) // game.column_count
            column = (action-1) % game.column_count
            print("alphazero action:", (column, row))
            gameState = gameState.generateSuccessor(-1, (column, row))
            gameState.data.printGrid()
            if gameState.isWin() : 
                print("ALPHAZERO")
                break
        else:
            action = minimax.getAction(gameState)
            print("minimax action:", action)
            gameState = gameState.generateSuccessor(1,action)
            gameState.data.printGrid()
            if gameState.isLose() : 
                print("Minimax haha")
                break
        player = game.get_opponent(player)
botmatch()