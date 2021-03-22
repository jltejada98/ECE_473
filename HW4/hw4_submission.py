import shell
import util
import wordsegUtil

############################################################
# Problem 1: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    #Let state indicate up to which index have we traversed in the string
    #Thus, we must subdivide the string into all possible substrings,
    # determine its score and append if the score is the lowest.=

    def start(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0 #Start at the beggining of string
        # END_YOUR_CODE

    def goalp(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return state == len(self.query) #Reached end of string
        # END_YOUR_CODE

    def expand(self, state):
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        stateList = []
        for i in range(1, len(self.query)-state+1): #Iterate all possible substring positions from 1 up to the current state
            action = self.query[state:(i+state)]
            newState = i+state
            cost = self.unigramCost(action)
            stateList.append((action, newState, cost))
        return stateList
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 2: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def start(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN) #Starting state at beggining of sentence
        # END_YOUR_CODE

    def goalp(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords)
        # END_YOUR_CODE

    def expand(self, state):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        stateList = []
        if len(self.possibleFills(self.queryWords[state[0]])) == 0: #No possible actions to be taken, see next word
            action = self.queryWords[state[0]]
            newState = (state[0] + 1, action)
            cost = self.bigramCost(state[1], action)
            stateList.append((action, newState, cost))
            return stateList

        for reconstruct in self.possibleFills(self.queryWords[state[0]]):
            action = reconstruct
            newState = (state[0] + 1, action)
            cost = self.bigramCost(state[1], action)
            stateList.append((action, newState, cost))
        return stateList
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords) == 0:
        return ''

    uniformCostSearch = util.UniformCostSearch(verbose=1)
    uniformCostSearch.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    return ' '.join(uniformCostSearch.actions)
    # END_YOUR_CODE


if __name__ == '__main__':
    shell.main()
