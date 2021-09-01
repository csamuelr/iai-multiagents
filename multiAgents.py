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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from time import sleep

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
		some Directions.X for some X in the set {North, South, West, East, Stop}
		"""
		# Collect legal moves and successor states

		# random.seed(random.randint(1,5))

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
	
		"*** YOUR CODE HERE ***"

		# print('successorGameState: {}'.format(successorGameState))
		# print('newPos: {}'.format(newPos))
		# print('newFood: {}'.format(newFood))
		# print('newGhostStates: {}'.format(newGhostStates[0].getPosition()))
		# print('newScaredTimes: {}'.format(newScaredTimes[0]))
		# print('Current Game State: {}'.format(currentGameState.getPacmanPosition()))
		# print('Current Lagal Actions: {}'.format(currentGameState.getLegalPacmanActions()))
		
		# print('Action: {}'.format(action))
		
		# print("HasFood: {}".format(successorGameState.hasFood(newPos[0], newPos[1])))

		reward = 1.0
		demage = 1.0

		closestG = [] 
		ghostsPositions = successorGameState.getGhostPositions()	
		ghostsDistances = [ manhattanDistance(newPos, gPos) for gPos in ghostsPositions ]

		foodDistances = []
		newFoodPositions = newFood.asList()
		if len(newFoodPositions) > 0:
			foodDistances = [ manhattanDistance(newPos, foodPos) for foodPos in newFoodPositions ]
		else:
			foodDistances.append(1) 

		capsulesDistances = []
		newCapsulesPositions = successorGameState.getCapsules()
		if len(newCapsulesPositions) > 0:
			capsulesDistances = [ manhattanDistance(newPos, capsulesPos) for capsulesPos in newCapsulesPositions ]
		else:
			capsulesDistances.append(1)


		# print('new capsules: {}'.format(capsulesDistances))

		closestF = min(foodDistances)
		closestC = min(capsulesDistances)
		closestG = min(ghostsDistances)
		
		if len(newScaredTimes):
			pass

		if newPos in newCapsulesPositions:
			reward += 50

		if newPos in ghostsPositions:
			demage -= 10
		
		if not successorGameState.hasFood(newPos[0], newPos[1]):
			demage -= 10

		if action == 'Stop':
			demage -= 50
		
		if len(foodDistances) == 0:
			return 0


		# print("Capsules: {}".format(successorGameState.getCapsules()))
		# print("Ghost Positions: {}".format(ghostsPositions))

		'''
		IDEA: 
		  >>> Calculate relative positions of pacman and ghsots
		  >>> Guide pacman deslocations by distance of the next food
		  >>> Check scared states
		  >>> check distance from capsules
		  >>> check walls
		'''

		'''
			>>> This evaluation fuction isn't working as expected
		'''

		# print('reward: {}'.format(reward))
		# print('demage: {}'.format(demage))

		return  successorGameState.getScore() + closestG / (closestF * 10) + closestC / (closestF * 100) + closestF / (closestC * 100) 


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

	# def __init__(self):
	# 	MultiAgentSearchAgent.__init__(self, evalFn = 'scoreEvaluationFunction', depth = '2')

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
		"""
		"*** YOUR CODE HERE ***"
		# util.raiseNotDefined()

		
		result = self.getValue(gameState, 0, 0)

		return result[1]

	def getValue(self, gameState, index, depth):
		
		if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
			return gameState.getScore(), None

		if index == 0:
			return self.maxValue(gameState, index, depth)

		else:
			return self.minValue(gameState, index, depth)

	def maxValue(self, gameState, index, depth):
		"""
		Returns the max utility value-action for max-agent
		"""
		legalMoves = gameState.getLegalActions(index)
		maxValue = float("-inf")
		maxAction = ""

		for action in legalMoves:
			successor = gameState.generateSuccessor(index, action)
			sucessorIndex = index + 1
			successorDepth = depth

			if sucessorIndex == gameState.getNumAgents():
				sucessorIndex = 0
				successorDepth += 1

			currentValue = self.getValue(successor, sucessorIndex, successorDepth)[0]

			if currentValue > maxValue:
				maxValue = currentValue
				maxAction = action

		return maxValue, maxAction

	def minValue(self, gameState, index, depth):
		"""
		Returns the min utility value-action for min-agent
		"""
		legalMoves = gameState.getLegalActions(index)
		minValue = float("inf")
		minAction = ""

		for action in legalMoves:
			successor = gameState.generateSuccessor(index, action)
			sucessorIndex = index + 1
			successorDepth = depth

			if sucessorIndex == gameState.getNumAgents():
				sucessorIndex = 0
				successorDepth += 1

			currentValue = self.getValue(successor, sucessorIndex, successorDepth)[0]

			if currentValue < minValue:
				minValue = currentValue
				minAction = action

		return minValue, minAction

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"
		# util.raiseNotDefined()
		
		# Format of result = [action, score]
		# Initial state: index = 0, depth = 0, alpha = -infinity, beta = +infinity
		result = self.getBestActionAndScore(gameState, 0, 0, float("-inf"), float("inf"))

		# Return the action from result
		return result[0]

	def getBestActionAndScore(self, state, index, depth, alpha, beta):
		"""
		Returns value as pair of [action, score] based on the different cases:
		1. Terminal state
		2. Max-agent
		3. Min-agent
		"""
		# Terminal states:
		if len(state.getLegalActions(index)) == 0 or depth == self.depth:
			return "", state.getScore()

		# Max-agent: Pacman has index = 0
		if index == 0:
			return self.maxValue(state, index, depth, alpha, beta)

		# Min-agent: Ghost has index > 0
		else:
			return self.minValue(state, index, depth, alpha, beta)

	def maxValue(self, state, index, depth, alpha, beta):
		"""
		Returns the max utility action-score for max-agent with alpha-beta pruning
		"""
		legalMoves = state.getLegalActions(index)
		maxValue = float("-inf")
		maxAction = ""

		for action in legalMoves:
			successor = state.generateSuccessor(index, action)
			sucessorIndex = index + 1
			successorDepth = depth

			# Update the successor agent's index and depth if it's pacman
			if sucessorIndex == state.getNumAgents():
				sucessorIndex = 0
				successorDepth += 1

			# Calculate the action-score for the current successor
			currentAction, currentValue \
				= self.getBestActionAndScore(successor, sucessorIndex, successorDepth, alpha, beta)

			# Update maxValue and maxAction for maximizer agent
			if currentValue > maxValue:
				maxValue = currentValue
				maxAction = action

			# Update alpha value for current maximizer
			alpha = max(alpha, maxValue)

			# Pruning: Returns maxValue because next possible maxValue(s) of maximizer
			# can get worse for beta value of minimizer when coming back up
			if maxValue > beta:
				return maxAction, maxValue

		return maxAction, maxValue

	def minValue(self, state, index, depth, alpha, beta):
		"""
		Returns the min utility action-score for min-agent with alpha-beta pruning
		"""
		legalMoves = state.getLegalActions(index)
		minValue = float("inf")
		minAction = ""

		for action in legalMoves:
			successor = state.generateSuccessor(index, action)
			sucessorIndex = index + 1
			successorDepth = depth

			# Update the successor agent's index and depth if it's pacman
			if sucessorIndex == state.getNumAgents():
				sucessorIndex = 0
				successorDepth += 1

			# Calculate the action-score for the current successor
			currentAction, currentValue \
				= self.getBestActionAndScore(successor, sucessorIndex, successorDepth, alpha, beta)

			# Update minValue and minAction for minimizer agent
			if currentValue < minValue:
				minValue = currentValue
				minAction = action

			# Update beta value for current minimizer
			beta = min(beta, minValue)

			# Pruning: Returns minValue because next possible minValue(s) of minimizer
			# can get worse for alpha value of maximizer when coming back up
			if minValue < alpha:
				return minAction, minValue

		return minAction, minValue

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	# def __init__(self):
	# 	MultiAgentSearchAgent.__init__(self, evalFn = 'scoreEvaluationFunction', depth = '2')

	def getAction(self, state):
		"""
		  Returns the expectimax action using self.depth and self.evaluationFunction

		  All ghosts should be modeled as choosing uniformly at random from their
		  legal moves.
		"""
		"*** YOUR CODE HERE ***"
		action, score = self.getValue(state, 0, 0)

		return action

	def getValue(self, state, index, depth):
		"""
		Returns value as pair of [action, score] based on the different cases:
		1. Terminal state
		2. Max-agent
		3. Expectation-agent
		"""
		# Terminal states:
		if state.isLose() or state.isWin() or depth == self.depth:
			return None, self.evaluationFunction(state)

		# Max-agent: Pacman has index = 0
		if index == 0:
			return self.maxValue(state, index, depth)

		# Expectation-agent: Ghost has index > 0
		else:
			return self.expectedValue(state, index, depth)

	def maxValue(self, state, index, depth):
		"""
		Returns the max utility value-action for max-agent
		"""
		legalMoves = state.getLegalActions(index)
		maxValue = float("-inf")
		maxAction = ''

		for action in legalMoves:
			successor = state.generateSuccessor(index, action)
			successorIndex = index + 1
			successorDepth = depth

			# Update the successor agent's index and depth if it's pacman
			if successorIndex == state.getNumAgents():
				successorIndex = 0
				successorDepth += 1

			currentAction, currentValue = self.getValue(successor, successorIndex, successorDepth)

			if maxValue is None or currentValue > maxValue:
				maxValue = currentValue
				maxAction = action

		return maxAction, maxValue

	def expectedValue(self, state, index, depth):
		"""
		Returns the max utility value-action for max-agent
		"""
		ghostMoves = state.getLegalActions(index)
		expectedValue = 0
		expectedAction = None

		# Find the current successor's probability using a uniform distribution
		if len(ghostMoves) != 0:
			successorProbability = float(1.0 / len(ghostMoves))

		for action in ghostMoves:
			successor = state.generateSuccessor(index, action)
			successorIndex = index + 1
			successorDepth = depth

			# Update the successor agent's index and depth if it's pacman
			if successorIndex == state.getNumAgents():
				successorIndex = 0
				successorDepth += 1

			# Calculate the action-score for the current successor
			currentAction, currentValue = self.getValue(successor, successorIndex, successorDepth)

			# Update expectedValue with the currentValue and successorProbability
			if expectedValue is None:
				expectedValue = 0.0

			score = successorProbability * currentValue
			expectedValue += score
			expectedAction = action

		if expectedValue is None:
			return None, self.evaluationFunction(state)

		return expectedAction, expectedValue
		

def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"
	util.raiseNotDefined()
	
# Abbreviation
better = betterEvaluationFunction


################################################
#				comunication				   #
################################################


class AgentsComunication(MinimaxAgent):

	def __init__(self):
		MinimaxAgent.__init__(self)
		self.agent = None
		self.con = Connection()

	def send(self):
		pass

	def request(self, agent, state):
		pass

	def message(self):
		pass

	def __repr__(self):
		return "Agent Comunication Object"

	def getAction(self, state):

		result = self.getValue(state, 0, 0)
		return result[1]

	def getValue(self, state, index, depth):
		
		if len(state.getLegalActions(index)) == 0 or depth == self.depth:
			return state.getScore(), None

		if index == 0:
			return self.maxValue(state, index, depth)

		else:
			pos = self.con.ask_position(index, state)
			if pos == state.getGhostPosition(index):
				return state.getScore(), None
			return self.minValue(state, index, depth)


class Connection:

	def __init__(self):
		pass

	def ask_position(self, agent, state):
		pos = state.getGhostPosition(agent)			
		print("Asking to Ghost {} his position...\nHis response: my position is {}.".format(agent, pos))

		return pos
