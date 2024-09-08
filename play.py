from connect_five import *
from miniMaxAgent import *

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
        

play()