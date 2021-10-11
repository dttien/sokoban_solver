
import functools
import sys
from typing import Tuple
import pygame
import string
import queue
import copy
import inspect
import heapq, random

class SokobanState:
    
    def __init__(self,matrix: list):
        self.queue = queue.LifoQueue()
        self.matrix = []
        self.matrix = copy.deepcopy(matrix)
        self.box = []
        self.dock = []
        self.width = 0 # width of the matrix
        self.height = 0 # height of the matrix
        x = 0
        y = 0
        # Store initial position of box and dock
        for row in self.matrix:
            for pos in row:
                if pos == '$' or pos == '*':
                    self.box.append([x,y])
                if pos == '.' or pos == '*' or pos == '+':
                    self.dock.append([x,y])
                x = x + 1
            y = y + 1
            x = 0

    def is_valid_value(self,char):
        if ( char == ' ' or #floor
            char == '#' or #wall
            char == '@' or #worker on floor
            char == '.' or #dock
            char == '*' or #box on dock
            char == '$' or #box
            char == '+' ): #worker on dock
            return True
        else:
            return False

    def check_deadlock(self):
        """
        This method return True if the current state is a deadlock, otherwise it return False
        """
        for box in self.box:
            if (self.matrix[box[1]][box[0] - 1] == '#' and self.matrix[box[1] - 1][box[0]] == '#') or \
                (self.matrix[box[1]][box[0] - 1] == '#' and self.matrix[box[1] + 1][box[0]] == '#') or \
                (self.matrix[box[1]][box[0] + 1] == '#' and self.matrix[box[1] - 1][box[0]] == '#') or \
                (self.matrix[box[1]][box[0] + 1] == '#' and self.matrix[box[1] + 1][box[0]] == '#') or \
                (self.matrix[box[1]][box[0]] == '$' and (self.matrix[box[1]][box[0] + 1] == '$' or self.matrix[box[1]][box[0] + 1] == '*') and \
                    (self.matrix[box[1] + 1][box[0]] == '$' or self.matrix[box[1] + 1][box[0]] == '*') and \
                    (self.matrix[box[1] + 1][box[0] + 1] == '$' or self.matrix[box[1] + 1][box[0] + 1] == '*')) or \
                (self.matrix[box[1]][box[0]] == '*' and (self.matrix[box[1]][box[0] + 1] == '$' or self.matrix[box[1] + 1][box[0]] == '$' or \
                    self.matrix[box[1] + 1][box[0] + 1] == '$')):
                return True
        
        # first line
        num_box_l1 = 0
        num_dock_l1 = 0
        # last line
        num_box_ll = 0 
        num_dock_ll = 0
        # left most
        num_box_l = 0
        num_dock_l = 0
        # right most
        num_box_r = 0
        num_dock_r = 0
        for box in self.box:
            if box[1] == 1:
                num_box_l1 = num_box_l1 + 1
            if box[1] == self.height - 2:
                num_box_ll = num_box_ll + 1
            if box[0] == 1:
                num_box_l = num_box_l + 1
            if box[0] == self.width - 2:
                num_box_r = num_box_r + 1
        for dock in self.dock:
            if dock[1] == 1: 
                num_dock_l1 = num_dock_l1 + 1
            if dock[1] == self.height - 2:
                num_dock_ll = num_dock_ll + 1
            if dock[0] == 1:
                num_dock_l = num_dock_l + 1
            if dock[0] == self.width - 2:
                num_dock_r = num_dock_r + 1
            
        if num_box_l1 > 0 and num_box_l1 > num_dock_l1: return True
        elif num_box_ll > 0 and num_box_ll > num_dock_ll: return True
        elif num_box_l > 0 and num_box_l > num_dock_l: return True
        elif num_box_r > 0 and num_box_r > num_dock_r: return True

        return False

    def load_size(self):
        self.height = len(self.matrix)
        
        for row in self.matrix:
            if len(row) > self.width:
                self.width = len(row)
        return (self.width * 32, self.height * 32)

    def get_matrix(self):
        return self.matrix

    def print_matrix(self):
        for row in self.matrix:
            for char in row:
                sys.stdout.write(char)
                sys.stdout.flush()
            sys.stdout.write('\n')

    def get_content(self,x,y):
        return self.matrix[y][x]

    def set_content(self,x,y,content):
        if self.is_valid_value(content):
            self.matrix[y][x] = content
        else:
            print ("ERROR: Value '"+content+"' to be added is not valid")

    def worker(self):
        x = 0
        y = 0
        for row in self.matrix:
            for pos in row:
                if pos == '@' or pos == '+':
                    return (x, y, pos)
                else:
                    x = x + 1
            y = y + 1
            x = 0

    def can_move(self,x,y):
        return self.get_content(self.worker()[0]+x,self.worker()[1]+y) not in ['#','*','$']

    def next(self,x,y):
        return self.get_content(self.worker()[0]+x,self.worker()[1]+y)

    def can_push(self,x,y):
        return (self.next(x,y) in ['*','$'] and self.next(x+x,y+y) in [' ','.'])


    def legalMoves(seft):
        """
          Returns a list of legal moves from the current state.

        Moves consist of moving the worker space up, down, left or right.
        These are encoded as 'up', 'down', 'left' and 'right' respectively.
        """
        moves = []

        if seft.can_move(0, -1) or seft.can_push(0, -1):
            moves.append('up')
        if seft.can_move(0, 1) or seft.can_push(0, 1):
            moves.append('down')
        if seft.can_move(-1, 0) or seft.can_push(-1, 0):
            moves.append('left')
        if seft.can_move(1, 0) or seft.can_push(1, 0):
            moves.append('right')
        return moves

    def result(self, moves):
        """
            Return a new updated game with provided moves

            NOTE: This function *does not* change the current object.  Instead,
            it returns a new object.
        """
        new_game = SokobanState(self.matrix)
        if moves == 'up':
            new_game.move(0, -1, True)
            return new_game
        elif moves == 'down':
            new_game.move(0, 1, True)
            return new_game
        elif moves == 'left':
            new_game.move(-1, 0, True)
            return new_game
        elif moves == 'right':
            new_game.move(1, 0, True)
            return new_game
        else:
            raise "Illegal move!"

    def is_completed(self):
        for row in self.matrix:
            for cell in row:
                if cell == '$':
                    return False
        return True

    def move_box(self,x,y,a,b):
        #  (x,y) -> move to do
        #  (a,b) -> box to move
        for box in self.box:
            if box[0] == x and box[1] == y:
                box[0] = x + a
                box[1] = y + b
        
        current_box = self.get_content(x,y)
        future_box = self.get_content(x+a,y+b)
        if current_box == '$' and future_box == ' ':
            self.set_content(x+a,y+b,'$')
            self.set_content(x,y,' ')
        elif current_box == '$' and future_box == '.':
            self.set_content(x+a,y+b,'*')
            self.set_content(x,y,' ')
        elif current_box == '*' and future_box == ' ':
            self.set_content(x+a,y+b,'$')
            self.set_content(x,y,'.')
        elif current_box == '*' and future_box == '.':
            self.set_content(x+a,y+b,'*')
            self.set_content(x,y,'.')

    def unmove(self):
        if not self.queue.empty():
            movement = self.queue.get()
            if movement[2]:
                current = self.worker()
                self.move(movement[0] * -1,movement[1] * -1, False)
                self.move_box(current[0]+movement[0],current[1]+movement[1],movement[0] * -1,movement[1] * -1)
            else:
                self.move(movement[0] * -1,movement[1] * -1, False)

    def move(self,x,y,save):
        if self.can_move(x,y):
            current = self.worker()
            future = self.next(x,y)
            if current[2] == '@' and future == ' ':
                self.set_content(current[0]+x,current[1]+y,'@')
                self.set_content(current[0],current[1],' ')
                if save: self.queue.put((x,y,False))
            elif current[2] == '@' and future == '.':
                self.set_content(current[0]+x,current[1]+y,'+')
                self.set_content(current[0],current[1],' ')
                if save: self.queue.put((x,y,False))
            elif current[2] == '+' and future == ' ':
                self.set_content(current[0]+x,current[1]+y,'@')
                self.set_content(current[0],current[1],'.')
                if save: self.queue.put((x,y,False))
            elif current[2] == '+' and future == '.':
                self.set_content(current[0]+x,current[1]+y,'+')
                self.set_content(current[0],current[1],'.')
                if save: self.queue.put((x,y,False))
        elif self.can_push(x,y):
            current = self.worker()
            future = self.next(x,y)
            future_box = self.next(x+x,y+y)
            if current[2] == '@' and future == '$' and future_box == ' ':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],' ')
                self.set_content(current[0]+x,current[1]+y,'@')
                if save: self.queue.put((x,y,True))
            elif current[2] == '@' and future == '$' and future_box == '.':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],' ')
                self.set_content(current[0]+x,current[1]+y,'@')
                if save: self.queue.put((x,y,True))
            elif current[2] == '@' and future == '*' and future_box == ' ':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],' ')
                self.set_content(current[0]+x,current[1]+y,'+')
                if save: self.queue.put((x,y,True))
            elif current[2] == '@' and future == '*' and future_box == '.':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],' ')
                self.set_content(current[0]+x,current[1]+y,'+')
                if save: self.queue.put((x,y,True))
            if current[2] == '+' and future == '$' and future_box == ' ':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],'.')
                self.set_content(current[0]+x,current[1]+y,'@')
                if save: self.queue.put((x,y,True))
            elif current[2] == '+' and future == '$' and future_box == '.':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],'.')
                self.set_content(current[0]+x,current[1]+y,'+')
                if save: self.queue.put((x,y,True))
            elif current[2] == '+' and future == '*' and future_box == ' ':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],'.')
                self.set_content(current[0]+x,current[1]+y,'+')
                if save: self.queue.put((x,y,True))
            elif current[2] == '+' and future == '*' and future_box == '.':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],'.')
                self.set_content(current[0]+x,current[1]+y,'+')
                if save: self.queue.put((x,y,True))
            

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        raiseNotDefined()

class SokobanSearchProblem(SearchProblem):
    """
      Implementation of a SearchProblem for the Sokoban domain

      Each state is represented by an instance of an game.
    """
    def __init__(self,sokoban):
        "Creates a new EightPuzzleSearchProblem which stores search information."
        self.sokoban = sokoban

    def getStartState(self):
        return self.sokoban

    def isGoalState(self,state):
        return state.is_completed()

    def getSuccessors(self,state):
        """
          Returns list of (successor, action, stepCost) pairs where
          each succesor is either left, right, up, or down
          from the original state and the cost is 1.0 for each
        """
        succ = []
        for a in state.legalMoves():
            successor = state.result(a)
            succ.append((successor, a, 1))
            # if successor.check_deadlock():
            #     print(successor.matrix)
        return succ

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0

class PriorityQueue:
    
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

def breadthFirstSearch(problem):
    # init frontier including the start state:
    frontier = Queue()
    # a frontier item include current state and movements to get there from start state
    frontier.push((problem.getStartState(), []))
    # init empty movements list and explored nodes list
    moves = []
    explored = []
    while not frontier.isEmpty():
        (state, path) = frontier.pop()
        # check if the state is goal
        if problem.isGoalState(state):
            moves = path
            break 
        # check if the state is not visited
        elif state not in explored:
            explored.append(state)
            for child in problem.getSuccessors(state):
                if child[0].check_deadlock() == False:
                    # add the new move
                    newPath = path + [child[1]]
                    # create new state
                    newState = (child[0], newPath)
                    frontier.push(newState)
    return moves

def nullHeuristic(state, problem=None):
    
    return 0

def Heuristic(state, problem):
    dist_sum = 0
    for box in state.box:
        min_dist = 2**31
        for storage in state.dock:
            # Calculate manhattan distance between box and storage point
            new_dist = abs(box[0] - storage[0]) + abs(box[1] - storage[1])
            if new_dist < min_dist:
                min_dist = new_dist
        dist_sum += min_dist
    return dist_sum

def aStarSearch(problem, heuristic=nullHeuristic):
    # Search the node that has the lowest combined cost and heuristic first.

    # init frontier including the start state:
    frontier = PriorityQueue()
    # a frontier item include current state and movements to get there from start state
    startNode = problem.getStartState()
    frontier.push(startNode, 0)

    # init empty movements list and explored nodes list
    moves = []
    explored = []
    # movement dictionary to each node
    move_dict = {str(startNode): []}

    while not frontier.isEmpty():
        state = frontier.pop()
        path = move_dict[str(state)]
        # check if the state is goal
        if problem.isGoalState(state):
            #print(move_dict)
            moves = path
            break
        
        # state is explored
        explored.append(state)
        # print(state, end=' ')
        # print(move_dict)

        for child in problem.getSuccessors(state):
            # path to child node
            newPath = path + [child[1]]
            newState = child[0]
            # if the child is not explored yet
            if child[0] not in explored:
                # add / update frontier
                frontier.update(newState, problem.getCostOfActions(newPath)+ heuristic(newState, problem))
                # add / update path movement dictionary
                if str(newState) in move_dict.keys() and problem.getCostOfActions(move_dict[str(newState)]) > problem.getCostOfActions(newPath):
                    move_dict[str(newState)] = newPath
                elif str(newState) not in move_dict.keys():
                    move_dict[str(newState)] = newPath
                # print(child)
                # print(heuristic(newState, problem))

    return moves


def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)

def is_valid_value(char):
    if ( char == ' ' or #floor
        char == '#' or #wall
        char == '@' or #worker on floor
        char == '.' or #dock
        char == '*' or #box on dock
        char == '$' or #box
        char == '+' ): #worker on dock
        return True
    else:
        return False

def read_level(filename,level):
    matrix = []
#        if level < 1 or level > 50:
    if level < 1:
        print ("ERROR: Level "+str(level)+" is out of range")
        sys.exit(1)
    else:
        file = open(filename,'r')
        level_found = False
        for line in file:
            row = []
            if not level_found:
                if  "Level "+str(level) == line.strip():
                    level_found = True
            else:
                if line.strip() != "":
                    row = []
                    for c in line:
                        if c != '\n' and is_valid_value(c):
                            row.append(c)
                        elif c == '\n': #jump to next row when newline
                            continue
                        else:
                            print ("ERROR: Level "+str(level)+" has invalid value "+c)
                            sys.exit(1)
                    matrix.append(row)
                else:
                    return matrix
    

def print_game(matrix,screen):
    screen.fill(background)
    x = 0
    y = 0
    for row in matrix:
        for char in row:
            if char == ' ': #floor
                screen.blit(floor,(x,y))
            elif char == '#': #wall
                screen.blit(wall,(x,y))
            elif char == '@': #worker on floor
                screen.blit(worker,(x,y))
            elif char == '.': #dock
                screen.blit(docker,(x,y))
            elif char == '*': #box on dock
                screen.blit(box_docked,(x,y))
            elif char == '$': #box
                screen.blit(box,(x,y))
            elif char == '+': #worker on dock
                screen.blit(worker_docked,(x,y))
            x = x + 32
        x = 0
        y = y + 32


def get_key():
    while 1:
        event = pygame.event.poll()
        if event.type == pygame.KEYDOWN:
            return event.key
        else:
            pass

def display_box(screen, message):
    "Print a message in a box in the middle of the screen"
    fontobject = pygame.font.Font(None,18)
    pygame.draw.rect(screen, (0,0,0),
                    ((screen.get_width() / 2) - 100,
                        (screen.get_height() / 2) - 10,
                        200,20), 0)
    pygame.draw.rect(screen, (255,255,255),
                    ((screen.get_width() / 2) - 102,
                        (screen.get_height() / 2) - 12,
                        204,24), 1)
    if len(message) != 0:
        screen.blit(fontobject.render(message, 1, (255,255,255)),
                    ((screen.get_width() / 2) - 100, (screen.get_height() / 2) - 10))
    pygame.display.flip()

def display_end(screen):
    message = "Level Completed"
    fontobject = pygame.font.Font(None,18)
    pygame.draw.rect(screen, (0,0,0),
                   ((screen.get_width() / 2) - 100,
                    (screen.get_height() / 2) - 10,
                    200,20), 0)
    pygame.draw.rect(screen, (255,255,255),
                   ((screen.get_width() / 2) - 102,
                    (screen.get_height() / 2) - 12,
                    204,24), 1)
    screen.blit(fontobject.render(message, 1, (255,255,255)),
                ((screen.get_width() / 2) - 100, (screen.get_height() / 2) - 10))
    pygame.display.flip()


def ask(screen, question):
    "ask(screen, question) -> answer"
    pygame.font.init()
    current_string = []
    display_box(screen, question + ": " + "".join(current_string))
    while 1:
        inkey = get_key()
        if inkey == pygame.K_BACKSPACE:
            current_string = current_string[0:-1]
        elif inkey == pygame.K_RETURN:
            break
        elif inkey == pygame.K_MINUS:
            current_string.append("_")
        elif inkey <= 127:
            current_string.append(chr(inkey))
        display_box(screen, question + ": " + "".join(current_string))
    return "".join(current_string)

def start_game():
    start = pygame.display.set_mode((320,240))
    level = int(ask(start,"Select Level"))
    if level > 0:
        return level
    else:
        print ("ERROR: Invalid Level: "+str(level))
        sys.exit(2)

if __name__ == '__main__':
    wall = pygame.image.load('images/wall.png')
    floor = pygame.image.load('images/floor.png')
    box = pygame.image.load('images/box.png')
    box_docked = pygame.image.load('images/box_docked.png')
    worker = pygame.image.load('images/worker.png')
    worker_docked = pygame.image.load('images/worker_dock.png')
    docker = pygame.image.load('images/dock.png')
    background = 255, 226, 191
    pygame.init()

    level = start_game()
    matrix = read_level('levels',level)
    game = SokobanState(matrix)
    size = game.load_size()
    #print(game.check_deadlock())
    screen = pygame.display.set_mode(size)

    problem = SokobanSearchProblem(game)
    solution_path = breadthFirstSearch(problem)
    print('BrFS found a path of %d moves: %s' % (len(solution_path), str(solution_path)))
    # solution_path = aStarSearch(problem, Heuristic)
    # print('A* found a path of %d moves: %s' % (len(solution_path), str(solution_path)))

    while 1:
        if game.is_completed(): display_end(screen)
        print_game(game.get_matrix(),screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: game.move(0,-1, True)
                elif event.key == pygame.K_DOWN: game.move(0,1, True)
                elif event.key == pygame.K_LEFT: game.move(-1,0, True)
                elif event.key == pygame.K_RIGHT: game.move(1,0, True)
                elif event.key == pygame.K_q: sys.exit(0)
                elif event.key == pygame.K_d: game.unmove()
        pygame.display.update()
