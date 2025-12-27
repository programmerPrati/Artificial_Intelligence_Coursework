# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack
from util import Queue
from util import PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # Set up stack and get start state
    stack = Stack()
    start_state = problem.getStartState()

    # Push start state with empty path (no path for root) on stack
    stack.push((start_state, []))

    # To keep track of visited
    visited = set()

    while not stack.isEmpty():
        state, path = stack.pop()
        #print("------------------")
        #print("This is the state:", state)
        #print("This is the path:", path)
        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            for nextState, action, _ in problem.getSuccessors(state):
                #print("This is new state:", state)
                #print("This is new action:", action)
                if nextState not in visited:
                    stack.push((nextState, path + [action]))

    return

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # Set up queue and get start state
    queue = Queue()
    start_state = problem.getStartState()

    # Push start state with empty path on stack
    queue.push((start_state, []))

    # To keep track of visited
    visited = set()

    while not queue.isEmpty():
        state, path = queue.pop()
        #print("------------------")
        #print("This is the state:", state)
        #print("This is the path:", path)
        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            for nextState, action, _ in problem.getSuccessors(state):
                #print("This is new state:", state)
                #print("This is new action:", action)
                if nextState not in visited:
                    queue.push((nextState, path + [action]))

    return

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    # Set up priority queue and get start state
    pQueue = PriorityQueue()
    start_state = problem.getStartState()

    # Push start state with empty actions and 0 cost
    pQueue.push((start_state, [], 0), 0)

    # To keep track of visited
    visited = set()

    while not pQueue.isEmpty():
        state, path, cost = pQueue.pop()
        #print("------------------")
        #print("This is the state:", state)
        #print("This is the path:", path)
        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            # get successors returns weight for the path as well
            for nextState, action, weight in problem.getSuccessors(state):
                #print("This is new state:", state)
                #print("This is new action:", action)
                # print("This is new weight:", weight)
                if nextState not in visited:
                    pQueue.push((nextState, path + [action], cost + weight), cost + weight)

    return

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # Set up priority queue and get start state
    starQueue = PriorityQueue()
    start_state = problem.getStartState()

    heuristicCost = heuristic(start_state, problem)
    cost = 0
    # Push start state with empty path on stack
    #starQueue.push((start_state, [], 0 + heuristicCost), (0 + heuristicCost))
    #starQueue.push((start_state, [], cost, heuristicCost), (heuristicCost))
    starQueue.push((start_state, [], cost), (cost + heuristicCost))

    # To keep track of visited
    visited = set()

    while not starQueue.isEmpty():
        state, path, cost = starQueue.pop()
        #print("------------------")
        #print("This is the state:", state)
        #print("This is the path:", path)
        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            for nextState, action, weight in problem.getSuccessors(state):
                #print("This is new state:", state)
                #print("This is new action:", action)
                if nextState not in visited:
                    newCost = cost + weight # weight is the new edge, which we don't need
                    h = heuristic(nextState, problem)
                    
                    starQueue.push((nextState, path + [action], newCost), newCost + h)

    return


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch