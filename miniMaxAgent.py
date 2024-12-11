from connect_five import *
import numpy as np
from multiprocessing import Pool
def parallel_worker(args):
            gameState, action, agent = args
            child_state = gameState.copy()
            child_state = agent.game.get_next_state(child_state, action, 1)
            value = agent.minimizer(child_state, agent.depth - 1, float("-inf"), float("inf"))
            return value, action

class MinimaxAgent():
    def __init__(self) :
        self.depth = 2
        self.game = ConnectFive()
        
        self.size = self.game.size

        self.left = self.size//2 - 1
        self.right = self.size//2 + 1
        self.top = self.size//2 - 1
        self.bottom = self.size//2 + 1
    def count_consecutive_values(self, gameState, target):
        height, width = gameState.shape

        def check_consecutive(x, y, dx, dy, value):
            count = 0
            while 0 <= x < width and 0 <= y < height:
                if gameState[x][y] == value:
                    count += 1
                    if count == target:
                        return True
                else:
                    return False
                x += dx
                y += dy
            return False

        countFree = countBounded = 0 
        for i in range(width):
            for j in range(height):
                if gameState[i][j] != 0:
                    value = gameState[i][j]
                    if value == -1: value1 = -1.1
                    else: value1 = 1
                    # Check vertically
                    if j <= height - target:
                        if check_consecutive(i, j, 0, 1,value):
                            if j > 0 and j + target < height: # not at borders
                                if gameState[i][j-1] == 0 and gameState[i][j+target] == 0: countFree += value1
                                elif gameState[i][j-1] == 0 or gameState[i][j+target] == 0: countBounded += value1
                            else: 
                                if j == 0: 
                                    if gameState[i][j+target] == 0: countBounded += value1
                                else:
                                    if gameState[i][j-1] == 0: countBounded += value1
                    # Check horizontally
                    if i <= width - target:
                        if check_consecutive(i, j, 1, 0,value):
                            if i > 0 and i + target < width:
                                if gameState[i-1][j] == 0 and gameState[i+target][j] == 0: countFree += value1
                                elif gameState[i-1][j] == 0 or gameState[i+target][j] == 0: countBounded += value1
                            else: 
                                if i == 0: 
                                    if gameState[i+target][j] == 0: countBounded+= value1
                                else:
                                    if gameState[i-1][j] == 0: countBounded+= value1
                    # Check diagonally (top-left to bottom-right)
                    if i <= width - target and j <= height - target:
                        if check_consecutive(i, j, 1, 1,value):
                            if i > 0 and j > 0 and i + target < width and j + target < height:
                                if gameState[i-1][j-1] == 0 and gameState[i+target][j+target] == 0: countFree += value1
                                elif gameState[i-1][j-1] == 0 or gameState[i+target][j+target] == 0: countBounded += value1
                            elif i + target < width - 1 and j + target < height - 1: 
                                if gameState[i+target][j+target] == 0: countBounded += value1
                            elif i > 0 and j > 0:
                                if gameState[i-1][j-1] == 0: countBounded += value1
                    # Check diagonally (top-right to bottom-left)
                    if i - target >= 0 and j <= height - target:
                        if check_consecutive(i, j, -1, 1,value):
                            if i + 1 < width and j > 0 and i - target > 0 and j + target < height:
                                if gameState[i+1][j-1] == 0 and gameState[i-target][j+target] == 0: countFree += value1
                                elif gameState[i+1][j-1] == 0 or gameState[i-target][j+target] == 0: countBounded += value1
                            elif i + 1 < width and j > 0: 
                                if gameState[i+1][j-1] == 0: countBounded += value1
                            elif i - target > 0 and j + target < height:
                                if gameState[i-target][j+target] == 0: countBounded += value1
        return countFree, countBounded
    
    def evaluationFunction1(self, gameState):
        res = 0
        for target_length in [5, 4, 3, 2]:
            free, bounded = self.count_consecutive_values(gameState, target_length)
            if target_length == 4 and free != 0:  
                res += free * 1e308 # Very high penalty for allowing four in a row
            else:
                res += free*((target_length-1)**2) * 10**target_length + bounded*(target_length-1) * 10**(target_length-2)

            if target_length == 5 and (free != 0 or bounded != 0):  
                res += free * float("inf")
                return res
        return res
    
    def minimizer (self, gameState, depth, a, b) :
        v = float("inf")
        if self.game.check_win(gameState, 1, 1): float("inf")

        values = []
        actions = self.game.get_valid_moves(gameState)
        indices = np.where(actions == 1)[0]
        if len(indices) == 0: return self.evaluationFunction1(gameState)
        indices = indices[(indices//self.size >= self.left) & (indices//self.size < self.right) & (indices%self.size >= self.top) & (indices%self.size < self.bottom)]

        actionsAndValues = []
        for action in indices:
            child_state = gameState.copy()
            child_state = self.game.get_next_state(child_state, action, -1)
            stateValue = self.evaluationFunction1(child_state)
            actionsAndValues.append((stateValue, child_state))
        
        sortedPairs = sorted(actionsAndValues, key=lambda tup: tup[0])
        for i in range(len(sortedPairs)):
            value, state = sortedPairs[i]
            nextValue, temp = self.maximizer(state, depth - 1, a, b)
            values.append(nextValue)
            sorted_list = sorted(values)
            # Get the top five highest values
            top_five = sorted_list[:3]
            # Calculate the average of the top five values
            average_top_five = sum(top_five) / len(top_five)
            v = min(v, nextValue)
            if v <= a: return v
            b = min(b, v)
        return v

    def maximizer (self, gameState, depth, a, b) :
        v = float("-inf")
        if self.game.check_win(gameState, 1, -1): float("-inf"), None
        if depth == 0 : return self.evaluationFunction1(gameState), None
        
        best = float("-inf")
        actions = self.game.get_valid_moves(gameState)
        indices = np.where(actions == 1)[0]
        if len(indices) == 0: return self.evaluationFunction1(gameState), None
        indices = indices[(indices//self.size >= self.left) & (indices//self.size < self.right) & (indices%self.size >= self.top) & (indices%self.size < self.bottom)]
        
        actionsAndValues = []
        for action in indices:
            child_state = gameState.copy()
            child_state = self.game.get_next_state(child_state, action, 1)
            stateValue = self.evaluationFunction1(child_state)
            if stateValue >= b:
                return stateValue, action  # prune
            actionsAndValues.append((stateValue, child_state, action))
        
        sortedPairs = sorted(actionsAndValues, key=lambda tup: tup[0], reverse=True)
        theaction = sortedPairs[0][2]
        possibleact = []
        for i in range(len(sortedPairs)):
            _, state, action = sortedPairs[i]
            nextValue = self.minimizer(state, depth, a, b)
            v = max (v, nextValue)
            if depth == self.depth: 
                possibleact += [(action, nextValue)]
            if v >= b : 
                return v, action
            a = max (a,v)
            if nextValue > best:
                best = nextValue
                theaction = action
        return v, theaction
        
    def getAction(self, gameState):

        actions = self.game.get_valid_moves(gameState).reshape(16,16)
        indices = np.where(actions != 1)
        self.top = max(min(indices[1]) - 2, 0)
        self.bottom = min(max(indices[1]) + 2, self.size)
        self.left = max(min(indices[0]) - 2, 0)
        self.right = min(max(indices[0]) + 2, self.size)

        actions = self.game.get_valid_moves(gameState)
        indices = np.where(actions == 1)[0]
        indices = indices[
            (indices // self.size >= self.left) &
            (indices // self.size < self.right) &
            (indices % self.size >= self.top) &
            (indices % self.size < self.bottom)
        ]

        args = [(gameState, action, self) for action in indices]
        # Parallel evaluation of root-level moves
        with Pool() as pool:
            results = pool.map(parallel_worker, args)

        # Find the best action
        best_value, best_action = max(results, key=lambda x: x[0])
        return best_action