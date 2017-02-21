########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang, Avishek Dutta
# Description:  Set 5 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding.s If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        ### 
        ### TODO: Insert Your Code Here (2E)
        ###

        import operator

        for j in range(self.L):
            probs[1][j] = self.A_start[j]*self.O[j][x[0]]
            seqs[1][j] = seqs[0][j] + str(j)

        for i in range(2,M+1):
            for j in range(self.L):
                probabilities = [probs[i-1][k]*self.A[k][j] for k in range(self.L)]
                best_state,best_prob = max(enumerate(probabilities), key=operator.itemgetter(1))
                probs[i][j] = best_prob*self.O[j][x[i-1]]
                seqs[i][j] = seqs[i-1][best_state] + str(j)

        max_state,max_prob = max(enumerate(probs[M]), key=operator.itemgetter(1))
        max_seq = seqs[M][max_state]

        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ### 
        ### TODO: Insert Your Code Here (2Fi)
        ###

        for j in range(self.L):
            alphas[1][j] = self.O[j][x[0]]*self.A_start[j]

        for i in range(2,M+1):
            for j in range(self.L):
                probabilities = [self.A[k][j]*alphas[i-1][k] for k in range(self.L)]
                alphas[i][j] = self.O[j][x[i-1]]*sum(probabilities)

        if normalize:
            for i in range(1,M+1):
                norm = sum(alphas[i])
                for j in range(len(alphas[i])):
                    alphas[i][j] /= norm

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ### 
        ### TODO: Insert Your Code Here (2Fii)
        ###

        betas[M] = [1.]*self.L

        for i in reversed(range(1,M)):
            for j in range(self.L):
                betas[i][j] = sum([betas[i+1][k]*self.A[j][k]*self.O[k][x[i]] for k in range(self.L)])

        if normalize:
            for i in range(1,M+1):
                norm = sum(betas[i])
                for j in range(len(betas[i])):
                    betas[i][j] /= norm

        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        stateCount = [0] * self.L
        transitionCount = [[0.] * self.L for _ in range(self.L)]
        emissionCount = [[0.] * self.D for _ in range(self.L)]

        # Calculate each element of A using the M-step formulas.

        for seq in Y:
            for curr in range(len(seq)-1):
                stateCount[seq[curr]] += 1
                transitionCount[seq[curr]][seq[curr+1]] += 1

        for curr in range(self.L):
            for nex in range(self.L):
                self.A[curr][nex] = transitionCount[curr][nex] / stateCount[curr]

        # Calculate each element of O using the M-step formulas.

        # last column for stateCount
        for seq in Y:
            stateCount[seq[-1]] += 1

        for i in range(len(X)):
            out = X[i]
            seq = Y[i]
            for t in range(len(out)-1):
                emissionCount[seq[t]][out[t]] += 1

        for curr in range(self.L):
            for obs in range(self.D):
                self.O[curr][obs] = emissionCount[curr][obs] / stateCount[curr]
        
        pass


    def unsupervised_learning(self, X):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
        '''

        ### 
        ### TODO: Insert Your Code Here (2H)
        ###

        numIterations = 1
        for iteration in range(numIterations):
            print iteration

            stateProbsList = []
            transitionProbsList = []

            # E-step

            for x in X:
                M = len(x)
                alphas = self.forward(x,normalize=True)
                betas = self.backward(x,normalize=True)

                stateProbs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
                for t in range(1,M+1):
                    denom = sum([alphas[t][state]*betas[t][state] for state in range(self.L)])
                    stateProbs[t] = [alphas[t][state]*betas[t][state]/denom for state in range(self.L)]
                stateProbsList.append(stateProbs)

                transitionProbs = [[[0. for _ in range(self.L)] for _ in range(self.L)] for _ in range(M)]
                for t in range(1,M):
                    toSum = [[alphas[t][curr]*self.O[nex][x[t]]*self.A[curr][nex]*betas[t+1][nex] for nex in range(self.L)] for curr in range(self.L)]
                    denom = sum(map(sum,toSum))
                    transitionProbs[t] = [[alphas[t][curr]*self.O[nex][x[t]]*self.A[curr][nex]*betas[t+1][nex]/denom for nex in range(self.L)] for curr in range(self.L)]
                transitionProbsList.append(transitionProbs)

            # M-step

            for curr in range(self.L):
                for nex in range(self.L):
                    numer = 0.0
                    denom = 0.0
                    for i in range(len(X)):
                        stateProbs = stateProbsList[i]
                        transitionProbs = transitionProbsList[i]
                        for t in range(1,len(transitionProbs)):
                            numer += transitionProbs[t][curr][nex]
                            denom += stateProbs[t][curr]
                    self.A[curr][nex] = numer/denom

            for curr in range(self.L):
                for obs in range(self.D):
                    numer = 0.0
                    denom = 0.0
                    for i in range(len(X)):
                        stateProbs = stateProbsList[i]
                        for t in range(1,len(stateProbs)):
                            denom += stateProbs[t][curr]
                            if X[i][t-1] == obs:
                                numer += stateProbs[t][curr]
                    self.O[curr][obs] = numer/denom

        pass


    def generate_emission(self, start, M):
        import numpy
        emission = []

        for t in range(M):
            if t > 0:
                currState = numpy.random.choice(range(self.L), p=self.A[currState])
            else:
                currState = start
            emission.append(numpy.random.choice(range(self.D), p=self.O[currState]))

        return emission

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    random.seed(5)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X)

    return HMM
