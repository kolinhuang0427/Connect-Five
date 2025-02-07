from connect_five import *
from miniMaxAgent import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from MPIAlphaZeroAgent import ResNet, Node
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
                    torch.tensor(self.game.get_encoded_state(node.state), 
                                 device=self.model.device).unsqueeze(0)
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
        

class human:
    def __init__(self):
        pass

    def __str__(self):
        return "human"

    def getAction(self, gameState):
        while True:
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
            
            action = user_input[0]*16 + user_input[1]
            return action

class alphaZeroPlayer:
    def __init__(self, model_path):
        game = ConnectFive()
        args = {
            'C': 2,
            'num_searches': 800,
            'dirichlet_epsilon': 0.,
            'dirichlet_alpha': 0.03
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNet(game, 9, 128, device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        self.mcts = MCTS(game, args, model)

    def __str__(self):
        return "alphazero"

    def getAction(self, gameState):
        mcts_probs = self.mcts.search(gameState)
        action = np.argmax(mcts_probs)
        return action

def print_game(gameState):
    gameState = np.vstack([np.arange(16), gameState])
    gameState = np.hstack([np.append(0, np.arange(16)).reshape(17,1), gameState])
    print(gameState.astype(int))
    return 

def play():
    game = ConnectFive()
    gameState = game.get_initial_state()

    player1 = human()
    player1 = alphaZeroPlayer("model_5_ConnectFive_20241214-182158.pt")
    #player2 = alphaZeroPlayer("")
    player2 = MinimaxAgent()

    player_index = -1
    while True:
        
        if player_index == -1:
            action = player1.getAction(gameState)

            print("Player1 action:", action//16, action%16)
            gameState = game.get_next_state(gameState, action, -1)
            print_game(gameState)
            win = game.check_win(gameState, action)
            if win:
                print(f"Player1 {player1} Wins!")
                break
        
        else:
            action = player2.getAction(gameState)

            print("Player2 action:", action//16, action%16)
            gameState = game.get_next_state(gameState, action, 1)
            print_game(gameState)
            win = game.check_win(gameState, action)
            if win:
                print(f"Player2 {player2} Wins!")
                break

        player_index = game.get_opponent(player_index)

if __name__ == '__main__':
    play()