from connect_five import *

class MinimaxAgent():
    def __init__(self) :
        self.depth = 2
    def getAction(self, gameState):
        def count_consecutive_values(matrix, target):
            width = len(matrix)
            height = len(matrix[0])

            def check_consecutive(x, y, dx, dy, value):
                count = 0
                while 0 <= x < width and 0 <= y < height:
                    if matrix[x][y] == value:
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
                    if matrix[i][j] != 0:
                        value = matrix[i][j]
                        if value == -1: value1 = -1.1
                        else: value1 = 1
                        # Check vertically
                        if j <= height - target:
                            if check_consecutive(i, j, 0, 1,value):
                                if j > 0 and j + target < height: # not at borders
                                    if matrix[i][j-1] == 0 and matrix[i][j+target] == 0: countFree += value1
                                    elif matrix[i][j-1] == 0 or matrix[i][j+target] == 0: countBounded += value1
                                else: 
                                    if j == 0: 
                                        if matrix[i][j+target] == 0: countBounded += value1
                                    else:
                                        if matrix[i][j-1] == 0: countBounded += value1
                        # Check horizontally
                        if i <= width - target:
                            if check_consecutive(i, j, 1, 0,value):
                                if i > 0 and i + target < width:
                                    if matrix[i-1][j] == 0 and matrix[i+target][j] == 0: countFree += value1
                                    elif matrix[i-1][j] == 0 or matrix[i+target][j] == 0: countBounded += value1
                                else: 
                                    if i == 0: 
                                        if matrix[i+target][j] == 0: countBounded+= value1
                                    else:
                                        if matrix[i-1][j] == 0: countBounded+= value1
                        # Check diagonally (top-left to bottom-right)
                        if i <= width - target and j <= height - target:
                            if check_consecutive(i, j, 1, 1,value):
                                if i > 0 and j > 0 and i + target < width and j + target < height:
                                    if matrix[i-1][j-1] == 0 and matrix[i+target][j+target] == 0: countFree += value1
                                    elif matrix[i-1][j-1] == 0 or matrix[i+target][j+target] == 0: countBounded += value1
                                elif i + target < width - 1 and j + target < height - 1: 
                                    if matrix[i+target][j+target] == 0: countBounded += value1
                                elif i > 0 and j > 0:
                                    if matrix[i-1][j-1] == 0: countBounded += value1
                        # Check diagonally (top-right to bottom-left)
                        if i - target >= 0 and j <= height - target:
                            if check_consecutive(i, j, -1, 1,value):
                                if i + 1 < width and j > 0 and i - target > 0 and j + target < height:
                                    if matrix[i+1][j-1] == 0 and matrix[i-target][j+target] == 0: countFree += value1
                                    elif matrix[i+1][j-1] == 0 or matrix[i-target][j+target] == 0: countBounded += value1
                                elif i + 1 < width and j > 0: 
                                    if matrix[i+1][j-1] == 0: countBounded += value1
                                elif i - target > 0 and j + target < height:
                                    if matrix[i-target][j+target] == 0: countBounded += value1
            return countFree, countBounded

        def evaluationFunction1(currentGameState):
            matrix = currentGameState.data.data
            if currentGameState.isWin(): return float("-inf")
            if currentGameState.isLose(): return float("inf")
            res = 0
            for target_length in [5, 4, 3, 2]:
                free, bounded = count_consecutive_values(matrix, target_length)
                if target_length == 4 and free != 0:  
                    res += free * 10000000000000 # Very high penalty for allowing four in a row
                else:
                    res += free*((target_length-1)**2) * 10**target_length + bounded*(target_length-1) * 10**(target_length-2)

                if target_length == 5 and (free != 0 or bounded != 0):  
                    res += free * float("inf") # Very high penalty for allowing four in a row
                    return res
            return res

        def maximizer (gameState, depth, a, b) :
            v = float("-inf")
            if gameState.isWin() or gameState.isLose(): return evaluationFunction1(gameState), None
            if depth == 0 : return evaluationFunction1(gameState), None
            best = float("-inf")
            actions = gameState.getLegalActions(1)
            if actions == []: return evaluationFunction1(gameState), None
            actionsAndValues = []
            for action in actions:
                nextState = gameState.generateSuccessor(1,action)
                stateValue = evaluationFunction1(nextState)
                if stateValue >= b:
                    return stateValue, action  # prune
                actionsAndValues.append((stateValue, nextState, action))
            sortedPairs = sorted(actionsAndValues, key=lambda tup: tup[0], reverse=True)
            theaction = sortedPairs[0][2]
            possibleact = []
            for i in range(len(sortedPairs)):
                value, state, action = sortedPairs[i]
                nextValue = minimizer(state, depth, a, b)
                v = max (v, nextValue)
                if depth == self.depth: 
                    #state.data.printGrid()
                    #print("this state's value is", value)
                    possibleact += [(action, nextValue)]
                if v >= b : 
                    if depth == self.depth: print("PRUNED, The set of considered actions were:", possibleact, "The AI chose", action, "with value", v)
                    return v, action
                a = max (a,v)
                if nextValue > best:
                    best = nextValue
                    theaction = action
            if depth == self.depth: print("NORMAL, The set of considered actions were:", possibleact, "The AI chose", theaction, "with value", v)
            return v, theaction
        
        def minimizer (gameState, depth, a, b) :
            v = float("inf")
            if gameState.isWin() or gameState.isLose(): return evaluationFunction1(gameState)

            values = []
            actions = gameState.getLegalActions(-1)
            if actions == []: return evaluationFunction1(gameState)
            actionsAndValues = []
            for action in actions:
                nextState = gameState.generateSuccessor(-1,action)
                stateValue = evaluationFunction1(nextState)
                actionsAndValues.append((stateValue, nextState))
            sortedPairs = sorted(actionsAndValues, key=lambda tup: tup[0])
            for i in range(len(sortedPairs)):
                value, state = sortedPairs[i]
                nextValue, temp = maximizer(state, depth - 1, a, b)
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
        
        value, action = maximizer(gameState, self.depth, float("-inf"), float("inf"))
        print("the value for this action is", value)
        return action