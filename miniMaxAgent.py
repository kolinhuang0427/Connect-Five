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

    def __str__(self):
        return "minimax"
    
    def count_consecutive_values(self, gameState):
        height, width = gameState.shape

        def check_consecutive(x, y, dx, dy, value):
            count = 0
            while 0 <= x < width and 0 <= y < height:
                if gameState[x][y] == value:
                    count += 1
                else:
                    return count
                x += dx
                y += dy
            return count
    
        indices = np.where(gameState != 0)
        indices_list = list(zip(indices[0], indices[1]))
        
        countFree = {x: 0 for x in np.arange(1,6,1)}
        countBounded = {x: 0 for x in np.arange(1,6,1)}
        for i, j in indices_list:
            value = gameState[i][j]
            if value == -1: value1 = -1.1
            else: value1 = 1

            # Check vertically
            count = check_consecutive(i, j, 0, 1,value)
            if j > 0 and j + count < height: # not at borders
                if gameState[i][j-1] == 0 and gameState[i][j+count] == 0: countFree[count] += value1
                elif gameState[i][j-1] == 0 or gameState[i][j+count] == 0: countBounded[count] += value1
            else: 
                if j == 0: 
                    if gameState[i][j+count] == 0: countBounded[count]+= value1
                else:
                    if gameState[i][j-1] == 0: countBounded[count] += value1

            # Check horizontally
            count = check_consecutive(i, j, 1, 0,value)
            if i > 0 and i + count < width:
                if gameState[i-1][j] == gameState[i+count][j] == 0: countFree[count] += value1
                elif gameState[i-1][j] == 0 or gameState[i+count][j] == 0: countBounded[count] += value1
            else: 
                if i == 0: 
                    if gameState[i+count][j] == 0: countBounded[count] += value1
                else:
                    if gameState[i-1][j] == 0: countBounded[count] += value1

            # Check diagonally (top-left to bottom-right)
            count = check_consecutive(i, j, 1, 1,value)
            if i > 0 and j > 0 and i + count < width and j + count < height:
                if gameState[i-1][j-1] == gameState[i+count][j+count] == 0: countFree[count] += value1
                elif gameState[i-1][j-1] == 0 or gameState[i+count][j+count] == 0: countBounded[count] += value1
            elif i + count < width - 1 and j + count < height - 1: 
                if gameState[i+count][j+count] == 0: countBounded[count] += value1
            elif i > 0 and j > 0:
                if gameState[i-1][j-1] == 0: countBounded[count] += value1

            # Check diagonally (top-right to bottom-left)
            count = check_consecutive(i, j, -1, 1,value)
            if i + 1 < width and j > 0 and i - count > 0 and j + count < height:
                if gameState[i+1][j-1] == gameState[i-count][j+count] == 0: countFree[count] += value1
                elif gameState[i+1][j-1] == 0 or gameState[i-count][j+count] == 0: countBounded[count] += value1
            elif i + 1 < width and j > 0: 
                if gameState[i+1][j-1] == 0: countBounded[count] += value1
            elif i - count > 0 and j + count < height:
                if gameState[i-count][j+count] == 0: countBounded[count] += value1
        return countFree, countBounded
    
    def evaluationFunction(self, gameState):
        res = 0
        free, bounded = self.count_consecutive_values(gameState)

        for key in free.keys():
            if key != 1:
                res += free[key] * 16**key
        for key in bounded.keys():
            if key != 1:
                res += bounded[key] * 16**key
        
        if free[5] != 0 or bounded[5] != 0:  
            res += free[5] * float("inf")
            res += bounded[5] * float("inf")

        return res
    
    def minimizer (self, gameState, depth, a, b) :
        v = float("inf")
        if self.game.check_win(gameState, 1, 1): float("inf")

        values = []
        actions = self.game.get_valid_moves(gameState)
        indices = np.where(actions == 1)[0]
        if len(indices) == 0: return self.evaluationFunction(gameState)
        indices = indices[(indices//self.size >= self.left) & (indices//self.size < self.right) 
                          & (indices%self.size >= self.top) & (indices%self.size < self.bottom)]

        actionsAndValues = []
        for action in indices:
            child_state = gameState.copy()
            child_state = self.game.get_next_state(child_state, action, -1)
            stateValue = self.evaluationFunction(child_state)
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
        if depth == 0 : return self.evaluationFunction(gameState), None
        
        best = float("-inf")
        actions = self.game.get_valid_moves(gameState)
        indices = np.where(actions == 1)[0]
        if len(indices) == 0: return self.evaluationFunction(gameState), None
        indices = indices[(indices//self.size >= self.left) & (indices//self.size < self.right) 
                          & (indices%self.size >= self.top) & (indices%self.size < self.bottom)]
        
        actionsAndValues = []
        for action in indices:
            child_state = gameState.copy()
            child_state = self.game.get_next_state(child_state, action, 1)
            stateValue = self.evaluationFunction(child_state)
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