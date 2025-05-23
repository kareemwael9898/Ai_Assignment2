# multiAgents.py
# --------------


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        # Start with the base score
        score = successorGameState.getScore()

        # Eat big capsules
        capsules = successorGameState.getCapsules()
        # If there are capsules, prefer to eat them
        if capsules:
            minCapsuleDistance = min(manhattanDistance(newPos, capsule) for capsule in capsules)
            score += 100.0 / (minCapsuleDistance + 1)
        else:
            score += 100
            
        # Compute distances to all food pellets
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min(manhattanDistance(newPos, food) for food in foodList)
            score += 10.0 / (minFoodDistance + 1)  # Prefer closer food
        else:
            score += 100  # No food left is excellent

        # Evaluate ghost proximity
        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            if newScaredTimes[i] > 0:
                # Encourage chasing scared ghosts
                score += 20.0 / (distance + 1)
            else:
                if distance < 2:
                    # Strongly penalize being too close to active ghosts
                    score -= 100

        # Discourage stopping
        if action == Directions.STOP:
            score -= 10

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        numAgents = gameState.getNumAgents()
        evalFn = self.evaluationFunction
        depthLimit = self.depth

        def maxValue(state, depth, agentIndex):
            # Base case: terminal state or depth limit
            if state.isWin() or state.isLose() or depth == depthLimit:
                return evalFn(state)

            v = -float('inf')
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return evalFn(state)

            for action in legalActions:
                if action == Directions.STOP:
                    continue
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, minValue(successor, depth, agentIndex + 1))
            return v

        def minValue(state, depth, agentIndex):
            # Base case: terminal state or depth limit
            if state.isWin() or state.isLose() or depth == depthLimit:
                return evalFn(state)

            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return evalFn(state)

            for action in legalActions:
                if action == Directions.STOP:
                    continue
                successor = state.generateSuccessor(agentIndex, action)

                if agentIndex == numAgents - 1:
                    # All agents have moved; increase depth
                    v = min(v, maxValue(successor, depth + 1, 0))
                else:
                    v = min(v, minValue(successor, depth, agentIndex + 1))
            return v

        # Evaluate all legal actions for Pacman
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP

        # Remove STOP to avoid unnecessary moves
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        # Find best action using minimax
        bestAction = max(
            legalActions,
            key=lambda action: minValue(
                gameState.generateSuccessor(0, action), 0, 1
            )
        )

        return bestAction
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()
        evalFn = self.evaluationFunction
        maxDepth = self.depth

        def maxValue(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == maxDepth:
                return evalFn(state)

            v = -float('inf')
            for action in state.getLegalActions(0):
                if action == Directions.STOP:
                    continue
                nextState = state.generateSuccessor(0, action)
                v = max(v, minValue(nextState, depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == maxDepth:
                return evalFn(state)

            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                if action == Directions.STOP:
                    continue
                nextState = state.generateSuccessor(agentIndex, action)

                if agentIndex == numAgents - 1:
                    v = min(v, maxValue(nextState, depth + 1, alpha, beta))
                else:
                    v = min(v, minValue(nextState, depth, agentIndex + 1, alpha, beta))

                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        # Find the best action
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP

        bestAction = None
        bestScore = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for action in legalActions:
            if action == Directions.STOP:
                continue
            nextState = gameState.generateSuccessor(0, action)
            score = minValue(nextState, 0, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        numAgents = gameState.getNumAgents()
        evalFn = self.evaluationFunction
        maxDepth = self.depth

        def maxValue(state, depth):
            if state.isWin() or state.isLose() or depth == maxDepth:
                return evalFn(state)

            v = -float('inf')
            for action in state.getLegalActions(0):
                if action == Directions.STOP:
                    continue
                nextState = state.generateSuccessor(0, action)
                v = max(v, expValue(nextState, depth, 1))
            return v

        def expValue(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return evalFn(state)

            v = 0
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return evalFn(state)

            probability = 1.0 / len(legalActions)
            for action in legalActions:
                if action == Directions.STOP:
                    continue
                nextState = state.generateSuccessor(agentIndex, action)

                if agentIndex == numAgents - 1:
                    v += probability * maxValue(nextState, depth + 1)
                else:
                    v += probability * expValue(nextState, depth, agentIndex + 1)
            return v

        # Find the best action
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP

        bestAction = None
        bestScore = -float('inf')

        for action in legalActions:
            if action == Directions.STOP:
                continue
            nextState = gameState.generateSuccessor(0, action)
            score = expValue(nextState, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function considers the following:
    - The current score of the game.
    - The distance to the nearest food pellet (closer is better).
    - The distance to the nearest ghost (farther is better unless the ghost is scared).
    - The number of remaining food pellets (fewer is better).
    - The number of remaining capsules (fewer is better).
    - Encourages eating scared ghosts when possible.

    The function combines these factors into a single score to guide Pacman's decisions.
    """
    # Extract useful information from the current game state
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # Initialize evaluation score with the current game score
    evaluationScore = score

    # Compute distances to all food pellets
    foodList = food.asList()
    if foodList:
        minFoodDistance = min(manhattanDistance(pacmanPos, food) for food in foodList)
        evaluationScore += 10.0 / (minFoodDistance + 1)  # Prefer closer food
    else:
        evaluationScore += 100  # No food left is excellent

    # Evaluate ghost proximity
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        distance = manhattanDistance(pacmanPos, ghostPos)
        if ghost.scaredTimer > 0:
            # Encourage chasing scared ghosts
            evaluationScore += 20.0 / (distance + 1)
        else:
            if distance < 2:
                # Strongly penalize being too close to active ghosts
                evaluationScore -= 100

    # Encourage eating capsules
    if capsules:
        minCapsuleDistance = min(manhattanDistance(pacmanPos, capsule) for capsule in capsules)
        evaluationScore += 50.0 / (minCapsuleDistance + 1)
    else:
        evaluationScore += 25  # No capsules left is good

    # Penalize having more food left
    evaluationScore -= 4 * len(foodList)

    # Penalize having more capsules left
    evaluationScore -= 10 * len(capsules)

    return evaluationScore

# Abbreviation
better = betterEvaluationFunction
