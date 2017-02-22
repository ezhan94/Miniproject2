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
import numpy
import operator

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


    # # Eric's solution
    # def unsupervised_learning(self, X):
    #     '''
    #     Trains the HMM using the Baum-Welch algorithm on an unlabeled
    #     datset X. Note that this method does not return anything, but
    #     instead updates the attributes of the HMM object.

    #     Arguments:
    #         X:          A dataset consisting of input sequences in the form
    #                     of lists of length M, consisting of integers ranging
    #                     from 0 to D - 1. In other words, a list of lists.
    #     '''

    #     ### 
    #     ### TODO: Insert Your Code Here (2H)
    #     ###

    #     numIterations = 10
    #     for iteration in range(numIterations):
    #         # print iteration

    #         A_Probs = [[0. for _ in range(self.L)] for _ in range(self.L)]
    #         O_Probs = [[0. for _ in range(self.D)] for _ in range(self.L)]

    #         for x in X:
    #             M = len(x)
    #             stateProbs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
    #             transitionProbs = [[[0. for _ in range(self.L)] for _ in range(self.L)] for _ in range(M)]
    #             alphas = self.forward(x,normalize=True)
    #             betas = self.backward(x,normalize=True)

    #             for t in range(1,M+1):
    #                 denom = sum([alphas[t][state]*betas[t][state] for state in range(self.L)])
    #                 stateProbs[t] = [alphas[t][state]*betas[t][state]/denom for state in range(self.L)]

    #             for t in range(1,M):
    #                 toSum = [[alphas[t][cur]*self.O[nxt][x[t]]*self.A[cur][nxt]*betas[t+1][nxt] for nxt in range(self.L)] for cur in range(self.L)]
    #                 denom = sum(map(sum,toSum))
    #                 transitionProbs[t] = [[alphas[t][cur]*self.O[nxt][x[t]]*self.A[cur][nxt]*betas[t+1][nxt]/denom for nxt in range(self.L)] for cur in range(self.L)]

    #             for t in range(1,M+1):
    #                 for cur in range(self.L):
    #                     for nxt in range(self.L):
    #                         A_Probs[cur][nxt] += transitionProbs[t-1][cur][nxt]
    #                 for cur in range(self.L):
    #                     O_Probs[cur][x[t-1]] += stateProbs[t][cur]

    #         for i in range(self.L):
    #             A_norm = sum(A_Probs[i])
    #             O_norm = sum(O_Probs[i])
    #             self.A[i] = [p/A_norm for p in A_Probs[i]]
    #             self.O[i] = [p/O_norm for p in O_Probs[i]]


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

        # Note that a comment starting with 'E' refers to the fact that
        # the code under the comment is part of the E-step.

        # Similarly, a comment starting with 'M' refers to the fact that
        # the code under the comment is part of the M-step.

        iters = 10
        for iteration in range(iters):
            print("Iteration: " + str(iteration))

            # Numerator and denominator for the update terms of A and O.
            A_num = [[0. for i in range(self.L)] for j in range(self.L)]
            O_num = [[0. for i in range(self.D)] for j in range(self.L)]
            A_den = [0. for i in range(self.L)]
            O_den = [0. for i in range(self.L)]

            # For each input sequence:
            for x in X:
                M = len(x)
                # Compute the alpha and beta probability vectors.
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                # E: Update the expected observation probabilities for a
                # given (x, y).
                # The i^th index is P(y^t = i, x).
                for t in range(1, M + 1):
                    P_curr = [0. for _ in range(self.L)]
                    
                    for curr in range(self.L):
                        P_curr[curr] = alphas[t][curr] * betas[t][curr]

                    # Normalize the probabilities.
                    norm = sum(P_curr)
                    for curr in range(len(P_curr)):
                        P_curr[curr] /= norm

                    for curr in range(self.L):
                        if t != M:
                            A_den[curr] += P_curr[curr]
                        O_den[curr] += P_curr[curr]
                        O_num[curr][x[t - 1]] += P_curr[curr]

                # E: Update the expectedP(y^j = a, y^j+1 = b, x) for given (x, y)
                for t in range(1, M):
                    P_curr_nxt = [[0. for _ in range(self.L)] for _ in range(self.L)]

                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] = alphas[t][curr] \
                                                    * self.A[curr][nxt] \
                                                    * self.O[nxt][x[t]] \
                                                    * betas[t + 1][nxt]

                    # Normalize:
                    norm = 0
                    for lst in P_curr_nxt:
                        norm += sum(lst)
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] /= norm

                    # Update A_num
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_num[curr][nxt] += P_curr_nxt[curr][nxt]

            for curr in range(self.L):
                for nxt in range(self.L):
                    self.A[curr][nxt] = A_num[curr][nxt] / A_den[curr]

            for curr in range(self.L):
                for xt in range(self.D):
                    self.O[curr][xt] = O_num[curr][xt] / O_den[curr]


    def generate_emission(self, lastObs, M):
        emission = [lastObs]
        currState, _ = max(enumerate([self.O[state][lastObs] for state in range(self.L)]), key=operator.itemgetter(1))

        for t in range(1,M):
            currState = numpy.random.choice(range(self.L), p=self.A[currState])
            emission.append(numpy.random.choice(range(self.D), p=self.O[currState]))

        return emission


    # # This is HW5 solutions' answer, for reference
    # def generate_emission(self, M):
    #     '''
    #     Generates an emission of length M, assuming that the starting state
    #     is chosen uniformly at random. 

    #     Arguments:
    #         M:          Length of the emission to generate.

    #     Returns:
    #         emission:   The randomly generated emission as a string.
    #     '''

    #     emission = ''
    #     state = random.choice(range(self.L))

    #     for t in range(M):
    #         # Sample next observation.
    #         rand_var = random.uniform(0, 1)
    #         next_obs = 0

    #         while rand_var > 0:
    #             rand_var -= self.O[state][next_obs]
    #             next_obs += 1

    #         next_obs -= 1
    #         emission += str(next_obs)

    #         # Sample next state.
    #         rand_var = random.uniform(0, 1)
    #         next_state = 0

    #         while rand_var > 0:
    #             rand_var -= self.A[state][next_state]
    #             next_state += 1

    #         next_state -= 1
    #         state = next_state

    #     return emission
    

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

    random.seed(7)

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
