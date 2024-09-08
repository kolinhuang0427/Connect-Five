"""AI agentIndex = 1"""
"""Player agentIndex = -1"""
"""isLose records if the player has lost (AI has won)"""
"""An action is a tuple of the place an agent can place a piece"""
import copy
import numpy as np
class Grid :

    def __init__(self, width=16, height=16, initialValue=0) -> None:
        self.left = width//2
        self.right = width//2
        self.top = height//2
        self.bottom = height//2
        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]
        self.isLose = False
        self.isWin = False
  
    def __eq__(self, __value: object) -> bool:
        if __value == None:
            return False
        return self.data == __value.data
    
    def _cellIndexToPosition(self, index):
        x = index // self.height
        y = index % self.height
        return x, y
    
    def checkWinLose(self, last_move):
        if last_move == None : return
        x, y = last_move
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)] # horizontal, vertical, and diagonals
        for dx, dy in directions:
            count_ai, count_player = 0, 0
            for i in range(-4, 5):  # check 5 in a row in each direction
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.data[nx][ny] == 1:
                        count_ai += 1
                        if count_ai == 5:
                            self.isLose = True
                            return
                    else:
                        count_ai = 0
                    if self.data[nx][ny] == -1:
                        count_player += 1
                        if count_player == 5:
                            self.isWin = True
                            return
                    else:
                        count_player = 0


    def act (self, x, y, agentIndex):
        if self.data[x][y] != 0:
            raise Exception("This cell already has a piece")
        self.data[x][y] = agentIndex
        if x <= self.left and x >= 1 : self.left = x - 1
        else: self.left = min (x, self.left)
        if x >= self.right - 1 and x < self.width - 1: self.right = x + 2
        else: self.right = max (x+1, self.right)
        if y <= self.top and y >= 1 : self.top = y - 1
        else: self.top = min (y, self.top)
        if y >= self.bottom - 1 and y < self.height - 1: self.bottom = y + 2
        else: self.bottom = max (y+1, self.bottom)

    def getValue (self, x, y):
        return self.data[x][y]

    def printGrid(self):
        print("  0 1 2 3 4 5 6 7 8 9 a b c d e f")
        for y in range(self.height) :
            temp = hex(y)
            temp = temp[-1]
            temp += ' '
            for x in range(self.width) :
                if self.data[x][y] == 1: state = 'A'
                elif self.data[x][y] == -1: state = 'P'
                else: state = '_'
                temp += state
                temp += ' '
            print(str(temp))

class GameState :

    def __init__(self, prevState = None) :
        if prevState != None :
            self.data = Grid()
            for y in range(prevState.data.height):
                for x in range(prevState.data.width):
                    if prevState.data.getValue(x,y) == 1:
                        self.data.act(x,y,1)
                    if prevState.data.getValue(x,y) == -1:
                        self.data.act(x,y,-1)
            self.lastMove = prevState.lastMove
        else:
            self.data = Grid()
            self.lastMove = None
        self.ai = AI()
        self.player = Player()
        

    def getLegalActions (self, agentIndex = 1) :
        if self.isWin() or self.isLose():
            return []
        
        if agentIndex == 1: #AI moving
            return self.ai.getLegalActions(self)
        else:
            assert(agentIndex == -1)
            return self.player.getLegalActions(self)
    
    def generateSuccessor(self, agentIndex, action) :

        if self.isWin() or self.isLose():
            raise Exception('Can\'t generate a successor of a terminal state.')
        
        state = GameState(self)

        if agentIndex == 1: #Ai is Moving
            state.ai.applyAction(state, action)
        else : state.player.applyAction(state,action)
        state.lastMove = action
        return state

    def isLose(self):
        self.data.checkWinLose(self.lastMove)
        return self.data.isLose
    
    def isWin(self):
        self.data.checkWinLose(self.lastMove)
        return self.data.isWin

class AI:

    def getLegalActions(self, state) :
        """Given a game state, returns the list of possible actions by the AI(maximizer)"""
        res = []
        ##print(state.data.top, state.data.bottom, state.data.left, state.data.right, state.data.data)
        for y in range(state.data.top, state.data.bottom):
            for x in range(state.data.left, state.data.right):
                if state.data.data[x][y] == 0 : res.append((x,y))
        return res
    
    def applyAction(self, state, action) :
        #print(action)
        x,y = action
        if state.data.getValue(x,y) != 0: raise Exception("Illegal action" + str(action))
        state.data.act(x,y,1)

class Player:

    def getLegalActions(self, state) :
        """Given a game state, returns the list of possible actions by the Player(minimizer)"""
        res = []
        for y in range(state.data.top, state.data.bottom):
            for x in range(state.data.left, state.data.right):
                if state.data.data[x][y] == 0 : res.append((x,y))
        return res

    def applyAction(self, state, action) :
        #print(action)
        (x,y) = action
        if state.data.getValue(x,y) != 0: raise Exception("Illegal action" + str(action))
        state.data.act(x,y,-1)

'''
testGameState = GameState()
testAI = AI()
testPlayer = Player()
#print(testGameState.data.isWin)
testPlayer.applyAction(testGameState,(8,8))
#print(testGameState.data.isWin)
testPlayer.applyAction(testGameState,(9,9))
testPlayer.applyAction(testGameState,(10,10))
testPlayer.applyAction(testGameState,(7,7))
testGameState1 = testGameState.generateSuccessor(-1,(6,6))

##print(testGameState.data.data)
'''
