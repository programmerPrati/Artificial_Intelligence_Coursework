# multiAgents.py
# --------------
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

# Ryhor Pryslopski u1385181
# Pratibodh Solanki u1363835


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        if action == Directions.STOP: # NO STOP
            score -= 1

        foodList = newFood.asList()

        if foodList:
            distancesToFood = [manhattanDistance(newPos, food) for food in foodList]
            minFoodDist = min(distancesToFood)
            score += 5.0 / minFoodDist

        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            distToGhost = manhattanDistance(newPos, ghostPos)

            if newScaredTimes[i] > 2: # Edible
                if distToGhost > 0:
                    score += 5.0 / distToGhost
            else: # Non-Edible
                if distToGhost <= 1:
                    score -= 1000
                else:
                    score -= 2.0 / distToGhost

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax(state, agentIndex, depth):
            # Finish
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0: # Pacman turn
                return max_value(state, agentIndex, depth)
            else: # Ghost turn
                return min_value(state, agentIndex, depth)

        def max_value(state, agentIndex, depth):
            v = float('-inf')
            actions = state.getLegalActions(agentIndex)

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = minimax(successor, 1, depth)  # Go to first ghost (index 1)
                v = max(v, value)
            return v

        def min_value(state, agentIndex, depth):
            v = float('inf')
            actions = state.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            isLastGhost = (nextAgent == gameState.getNumAgents())

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                if isLastGhost:
                    value = minimax(successor, 0, depth + 1)  # Back to Pacman
                else:
                    value = minimax(successor, nextAgent, depth)  # Next ghost
                v = min(v, value)
            return v

        # Choose best action from current state
        bestScore = float('-inf')
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            score = minimax(successor, 1, 0)  # Start from ghost index 1, depth 0
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alpha_beta(state, agentIndex, depth, alpha, beta):
            # Finish
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman turn
                return max_value(state, agentIndex, depth, alpha, beta)
            else:  # Ghost turn
                return min_value(state, agentIndex, depth, alpha, beta)

        def max_value(state, agentIndex, depth, alpha, beta):
            v = float('-inf')
            actions = state.getLegalActions(agentIndex)
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = alpha_beta(successor, 1, depth, alpha, beta)
                v = max(v, value)
                if v > beta:  # prune
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, agentIndex, depth, alpha, beta):
            v = float('inf')
            actions = state.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            isLastGhost = (nextAgent == gameState.getNumAgents())

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                if isLastGhost:
                    value = alpha_beta(successor, 0, depth + 1, alpha, beta)
                else:
                    value = alpha_beta(successor, nextAgent, depth, alpha, beta)
                v = min(v, value)
                if v < alpha:  # prune
                    return v
                beta = min(beta, v)
            return v

        alpha = float('-inf')
        beta = float('inf')
        bestScore = float('-inf')
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = alpha_beta(successor, 1, 0, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(agentIndex, depth, gameState):
            # End on game-over or depth lim
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # MAX
                return max(expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action))
                           for action in legalActions)
            else:  # Chance
                values = [expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action))
                          for action in legalActions]
                return sum(values) / len(values)  # Uniform prob

        # Root
        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            score = expectimax(1, 0, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    I combine
    - Game score
    - Inverse of distance to closest food
    - Penalty for distance to active ghosts
    - Reward for being close to scared ghosts
    - Number of remaining food and big dots
    - Bonus for eating capsules

    In theory it avoids dangerous moves, chases scared ghosts
    when its better than more profitable, and eats  food
    """

    # Basic info
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # Closest food
    foodList = foodGrid.asList()
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]
    minFoodDist = min(foodDistances) if foodDistances else 1

    # Ghost handling
    ghostPenalty = 0
    scaredBonus = 0
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        distToGhost = manhattanDistance(pacmanPos, ghostPos)
        if ghost.scaredTimer > 0:
            # Encourage approaching scared ghosts
            scaredBonus += 200 / (distToGhost + 1)
        else:
            if distToGhost <= 1:
                ghostPenalty += 500
            elif distToGhost <= 2:
                ghostPenalty += 200
            elif distToGhost <= 3:
                ghostPenalty += 100

    # Capsule and food penalty (fewer is better)
    capsulePenalty = 50 * len(capsules)
    foodLeftPenalty = 10 * len(foodList)

    final_score = (
        score
        + 10 / minFoodDist
        + scaredBonus
        - ghostPenalty
        - capsulePenalty
        - foodLeftPenalty
    )

    return final_score

# Abbreviation
better = betterEvaluationFunction
