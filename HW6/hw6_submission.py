from util import manhattanDistance
from game import Directions
import random, util
import queue

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


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

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    def Vminimax(gameState, agentIndex, currentDepth):
      #Terminating Conditions
      if gameState.isWin() or gameState.isLose():
        return gameState.getScore()
      elif currentDepth == 0:
        return self.evaluationFunction(gameState)

      if agentIndex == 0: #Pacman
          maxVminimax = float("-inf")
          for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            maxVminimax = max(maxVminimax, Vminimax(successorState, agentIndex+1, currentDepth))
          return maxVminimax
      else: #Ghosts
        if agentIndex == (gameState.getNumAgents() - 1):
          currentDepth -= 1
        minVminimax = float("+inf")
        for action in gameState.getLegalActions(agentIndex):
          successorState = gameState.generateSuccessor(agentIndex, action)
          newAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
          minVminimax = min(minVminimax, Vminimax(successorState,newAgentIndex, currentDepth))
        return minVminimax



    decisionValue = (float("-inf"), Directions.STOP)
    for action in gameState.getLegalActions(0):
      successorState = gameState.generateSuccessor(0, action)
      VminimaxTuple = (Vminimax(successorState, 1, self.depth), action)
      decisionValue = max(decisionValue, VminimaxTuple)

    # print(decisionValue[0])

    return decisionValue[1]
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    def VAlphaBeta(gameState, agentIndex, currentDepth, alphaValue, betaValue):
      #Terminating Conditions
      if gameState.isWin() or gameState.isLose():
        return gameState.getScore()
      elif currentDepth == 0:
        return self.evaluationFunction(gameState)

      if agentIndex == 0: #Pacman - MAX AGENT
          maxVminimax = float("-inf")
          for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            maxVminimax = max(maxVminimax, VAlphaBeta(successorState, agentIndex+1, currentDepth, alphaValue, betaValue))
            #Determine if beta-Condition satisfied
            if maxVminimax >= betaValue:
              return maxVminimax
            #Update Alpha Value
            alphaValue = max(alphaValue, maxVminimax)
          return maxVminimax
      else: #Ghosts - MIN AGENT
        if agentIndex == (gameState.getNumAgents() - 1):
          currentDepth -= 1
        minVminimax = float("+inf")
        for action in gameState.getLegalActions(agentIndex):
          successorState = gameState.generateSuccessor(agentIndex, action)
          newAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
          minVminimax = min(minVminimax, VAlphaBeta(successorState,newAgentIndex, currentDepth, alphaValue, betaValue))
          #Determine if alpha-Condition satisfied
          if minVminimax <= alphaValue:
            return minVminimax
          #Update Beta Value
          betaValue = min(betaValue, minVminimax)
        return minVminimax

    decisionValue = (float("-inf"), Directions.STOP)
    for action in gameState.getLegalActions(0):
      successorState = gameState.generateSuccessor(0, action)
      VminimaxTuple = (VAlphaBeta(successorState, 1, self.depth,float("-inf"), float("+inf")), action)
      decisionValue = max(decisionValue, VminimaxTuple)

    # print(decisionValue[0])

    return decisionValue[1]



    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    def Vexptimax(gameState, agentIndex, currentDepth):
      #Terminating Conditions
      if gameState.isWin() or gameState.isLose():
        return gameState.getScore()
      elif currentDepth == 0:
        return self.evaluationFunction(gameState)

      if agentIndex == 0: #Pacman
          maxVminimax = float("-inf")
          for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            maxVminimax = max(maxVminimax, Vexptimax(successorState, agentIndex+1, currentDepth))
          return maxVminimax
      else: #Ghosts
        if agentIndex == (gameState.getNumAgents() - 1):
          currentDepth -= 1
        minVminimax = 0
        #Probabilistic action -> Calculate mean value of available actions
        for action in gameState.getLegalActions(agentIndex):
          successorState = gameState.generateSuccessor(agentIndex, action)
          newAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
          minVminimax += Vexptimax(successorState,newAgentIndex, currentDepth)
        minVminimax = minVminimax / len(gameState.getLegalActions(agentIndex))
        return minVminimax



    decisionValue = (float("-inf"), Directions.STOP)
    for action in gameState.getLegalActions(0):
      successorState = gameState.generateSuccessor(0, action)
      VminimaxTuple = (Vexptimax(successorState, 1, self.depth), action)
      decisionValue = max(decisionValue, VminimaxTuple)

    return decisionValue[1]
    # END_YOUR_CODE

######################################################################################
# Problem 4a : creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
      Your extreme, unstoppable evaluation function (problem 4).

      DESCRIPTION: <write something here so we know what you did>
    """

    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    raise Exception('Not implemented yet')
    # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction

# Problem 4b : Describe your evaluation function.
