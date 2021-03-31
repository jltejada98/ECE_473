import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 1

# If you decide 1 is true,  put "return None" for
# the code blocks below.  If you decide that 1 is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 'S'
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return ['Move']
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        if state in ['A', 'B']:
            return []
        else:
            return [('A', 0.75, 0), ('B', 0.25, 10)]
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 2a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 37 lines of code, but don't worry if you deviate from this)
        totalCardValue, nextCardPeekedIndex, deckCardCounts = state
        succProbRewards = []
        if deckCardCounts == None:  # End state
            return []

        if action == 'Quit':
            newState = (totalCardValue, nextCardPeekedIndex, None)
            succProbRewards.append((newState, 1, totalCardValue)) #Get total card value, None -> No cards in deck
        elif action == 'Peek':
            if nextCardPeekedIndex != None:  # Cannot  peek twice
                return []
            else:
                for i in range(len(self.cardValues)):  # Peek one card from each value if there are available
                    if deckCardCounts[i] > 0:
                        newState = (totalCardValue, i, deckCardCounts)
                        prob = deckCardCounts[i] / sum(deckCardCounts)
                        succProbRewards.append((newState, prob, -self.peekCost))
        elif action == 'Take':
            if nextCardPeekedIndex != None:  # Take peeked card and move to appropiate next state
                totalCardValue += self.cardValues[nextCardPeekedIndex]
                if totalCardValue > self.threshold:  # Bust -> Move to End state
                    newState = (totalCardValue, None, None)
                    succProbRewards.append((newState, 1, 0))
                else:
                    newCardCounts = list(deckCardCounts)
                    newCardCounts[nextCardPeekedIndex] -= 1 #Remove peeked card
                    newState = (totalCardValue, None, tuple(newCardCounts))
                    succProbRewards.append((newState, 1, 0))
            else: #No Card previously peeked, check all card values that have at least one card
                for i in range(len(self.cardValues)):
                    if deckCardCounts[i] > 0:
                        newCardCounts = list(deckCardCounts)
                        newCardValueInHand = totalCardValue + self.cardValues[i]
                        prob = deckCardCounts[i] / sum(deckCardCounts)
                        newCardCounts[i] -= 1
                        if newCardValueInHand > self.threshold:  # Bust -> End State
                            newState = (newCardValueInHand, None, None)
                            succProbRewards.append((newState, prob, 0))
                        else:
                            if sum(newCardCounts) > 0:  #If the deck still has cards left
                                newState = (newCardValueInHand, None, tuple(newCardCounts))
                                succProbRewards.append((newState, prob, 0))
                            else: #Empty Deck, essentially quitting state
                                newState = (newCardValueInHand, None, None)
                                succProbRewards.append((newState, prob, newCardValueInHand))
        return succProbRewards
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 2b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return BlackjackMDP(cardValues=[2,3,15], multiplicity=6,threshold=20, peekCost=1)
    # END_YOUR_CODE

############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a dict of {feature name => feature value}.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f,v in self.featureExtractor(state, action).items():
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        n = self.getStepSize()
        if newState != None:
            V_opt = max((self.getQ(newState, possibleActions)) for possibleActions in self.actions(newState))
        else:
            V_opt = 0
        Q_opt = self.getQ(state, action)
        for f, v in self.featureExtractor(state, action).items():
            self.weights[f] -= n * (Q_opt - (reward + self.discount * V_opt)) * v
        # END_YOUR_CODE

# Return a single-element dict containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return {featureKey: featureValue}

############################################################
# Problem 3b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
    vI = ValueIteration()
    vI.solve(mdp)
    vIPI = vI.pi
    QLA = QLearningAlgorithm(actions=mdp.actions, discount=mdp.discount(), featureExtractor=featureExtractor)
    util.simulate(mdp=mdp, rl=QLA, numTrials=30000, verbose=False)
    QLA.explorationProb = 0
    diff = 0
    for state in mdp.states:
        if vIPI[state] != QLA.getAction(state):
            diff += 1
    print("Difference between Value Iteration and QLA:", diff/len(mdp.states))

    # END_YOUR_CODE

############################################################
# Problem 3b: features for Q-learning.

# You should return a dict of {feature key => feature value}.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the dict you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck. (1 feature)
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- For each face value, add an indicator for the action and the number of cards remaining with that face value (len(counts) features)
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    features = {}
    features[(action, total)] = 1 #Initialize action dictionary
    if counts != None:
        features[(action, tuple([1 if i>0 else 0 for i in counts]))] = 1 #Set Rewars
        for i in range(len(counts)):
            features[(action, i, counts[i])] = 1
    return features
    # END_YOUR_CODE

############################################################
# Problem 3c: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    #Use originalMDP with value iteration, solve
    vI = ValueIteration()
    vI.solve(original_mdp)
    #Simulate using Fixed RLA result on modified mdp
    RLA = util.FixedRLAlgorithm(vI.pi)
    rewards = util.simulate(modified_mdp, RLA)
    expectedReward = sum(rewards) / len(rewards)
    print("Rewards using fixed RLA on modified mdp", expectedReward)

    #Simulating QLearning directly on modified mdp, with blackjack feature extractor
    QLA = QLearningAlgorithm(actions=original_mdp.actions, discount=original_mdp.discount(), featureExtractor=featureExtractor)
    QLARewards = util.simulate(modified_mdp, QLA, numTrials=30000)
    expectedRewardQLA = sum(QLARewards) / len(QLARewards)
    print("Rewards using QLA on modified mdp",expectedRewardQLA )


    # END_YOUR_CODE
    
    
## Problem 3c. Discussion.
# The expected reward using the Fixed RLA which was trained using value iteration on the origianal MDP was of 6.6
# Whilst te expected reward obtained using the QLA on the modified mdp was measurably higher at 9.4467.
# This is mainly due to the fact that the fixed RLA algorithm simply returns the action that is set by the policy
# blidly. Meanwhile the QLA algorithm, may choose to either explore a different state, or stick to the current policy.
# Since the optimal policy found is on a different MDP, the Fixed RLA may perform suboptimally, whilst the QLA may
# explore a different state which may lead to a more optimal policy, when updating the weights of the estimator.
#
