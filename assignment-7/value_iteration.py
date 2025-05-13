#Siddharth Bhagvagar
#1001986088
import numpy as np

def up(util,i,j,block_states):
    up = 0.0
    row,col = util.shape

    if (i-1,j) not in block_states and i-1 >= 0:
        up += 0.8 * util[i-1][j]
    else:
        up += 0.8 * util[i][j]
    
    if (i ,j-1) not in block_states and j-1 >= 0:
        up += 0.1 * util[i][j-1]
    else:
        up += 0.1 * util[i][j]
    
    if (i ,j+1) not in block_states and j+1 < col:
        up += 0.1 * util[i][j+1]
    else:
        up += 0.1 * util[i][j]
    return up

def down(util,i,j,block_states):
    down = 0.0
    row,col = util.shape
    if (i+1,j) not in block_states and i+1 < row:
        down += 0.8 * util[i+1][j]
    else:
        down += 0.8 * util[i][j]

    if (i ,j-1) not in block_states and j-1 >= 0:
        down += 0.1 * util[i][j-1]
    else:
        down += 0.1 * util[i][j]
    
    if (i ,j+1) not in block_states and j+1 < col:
        down += 0.1 * util[i][j+1]
    else:
        down += 0.1 * util[i][j]
    return down

def left(util,i,j,block_states):
    left = 0.0
    row,col = util.shape

    if (i,j-1) not in block_states and j-1 >= 0:
        left += 0.8 * util[i][j-1]
    else:
        left += 0.8 * util[i][j]
    
    if (i-1 ,j) not in block_states and i-1 >= 0:
        left += 0.1 * util[i-1][j]
    else:
        left += 0.1 * util[i][j]
    
    if (i+1 ,j) not in block_states and i+1 < row:
        left += 0.1 * util[i+1][j]
    else:
        left += 0.1 * util[i][j]

    return left

def right(util,i,j,block_states):
    right = 0.0
    row,col = util.shape

    if (i,j+1) not in block_states and j+1 < col:
        right += 0.8 * util[i][j+1]
    else:
        right += 0.8 * util[i][j]

    if (i-1 ,j) not in block_states and i-1 >= 0:
        right += 0.1 * util[i-1][j]
    else:
        right += 0.1 * util[i][j]
    
    if (i+1 ,j) not in block_states and i+1 < row:
        right += 0.1 * util[i+1][j]
    else:
        right += 0.1 * util[i][j]
    return right

def value_iteration(environment_file, non_terminal_reward, gamma, K):
        
    world = np.loadtxt(environment_file,delimiter=',', dtype=str)
    
    rows = world.shape[0]
    cols = world.shape[1]

    utilities = np.zeros((world.shape[0], world.shape[1]),dtype=float)
    policy = np.full((world.shape[0], world.shape[1]),'',dtype=str)
    
    block_states = set()
    terminal_state = set()

    
    for i in range(rows):
        for j in range(cols):
            grid = world[i][j]
            if grid == 'X':
                block_states.add((i, j))
                utilities[i][j] = 0
                policy[i][j] = 'X'
            elif grid != '.':
                terminal_state.add((i, j))
                utilities[i][j] = float(grid)
                policy[i][j] = 'o'
  
    for k in range(K):
        next_utilities = np.copy(utilities)
        
        for i in range(0,rows,1):
            for j in range(0,cols,1):
                if (i, j) not in terminal_state and (i, j) not in block_states:

                    action_up = up(utilities,i,j,block_states)
                    # print(action_up)
                    action_down = down(utilities,i,j,block_states)
                    
                    action_left = left(utilities,i,j,block_states)
                    
                    action_right = right(utilities,i,j,block_states)

                    max_utility = max(action_up,action_down,action_left,action_right)
                    if max_utility == action_up:
                        policy[i][j] = "^"
                    elif max_utility == action_down:
                        policy[i][j] = "v"
                    elif max_utility == action_left:
                        policy[i][j] = "<"
                    elif max_utility == action_right:
                        policy[i][j] = ">"
                    next_utilities[i][j] = non_terminal_reward + gamma * max_utility
        
        utilities = next_utilities

    print("utilities:")
    for i in range(0,rows,1):
        for j in range(0,cols,1):
            print(f"{utilities[i][j]:6.3f} ",end=" ")
        print("\n")
    print("policy:")
    for i in range(0,rows,1):
        for j in range(0,cols,1):
            print(f"{policy[i][j]:6.3s} ",end=" ")
        print("\n")
    