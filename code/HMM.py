import random
import numpy
import operator

class HiddenMarkovModel:

    def __init__(self, A, O):
        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]
        self.iterations = 10


    def forward(self, x, normalize=False):
        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

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
        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

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
        for iteration in range(self.iterations):
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


    # def generate_emission(self, lastObs, M):
    #     emission = [lastObs]
    #     currState, _ = max(enumerate([self.O[state][lastObs] for state in range(self.L)]), key=operator.itemgetter(1))

    #     for t in range(1,M):
    #         currState = numpy.random.choice(range(self.L), p=self.A[currState])
    #         emission.append(numpy.random.choice(range(self.D), p=self.O[currState]))

    #     return emission


    def generate_new_emission(self, lastObs, obsList, syllDict):
        emission = [lastObs]
        currState, _ = max(enumerate([self.O[state][lastObs] for state in range(self.L)]), key=operator.itemgetter(1))
        syllCount = 0
        currWord = obsList[lastObs]
        numSyll, currEmph = syllDict[currWord]
        syllCount += numSyll
        emph = True
        if currEmph and syllCount % 2 is 0:
            emph = False
        elif not currEmph and syllCount % 2 is not 0:
            emph = False
        while syllCount < 10:
            currState = numpy.random.choice(range(self.L), p=self.A[currState])
            probs = self.generate_emission_probs(10-syllCount, emph, currState, obsList, syllDict)
            lastObs = numpy.random.choice(range(self.D), p=probs)
            emission.append(lastObs)
            currWord = obsList[lastObs]
            numSyll, currEmph = syllDict[currWord]
            if numSyll == 1:
                currEmph = not emph
            syllCount += numSyll
            emph = True
            if currEmph and syllCount % 2 is 0:
                emph = False
            elif not currEmph and syllCount % 2 is not 0:
                emph = False

        return emission


    def generate_emission_probs(self, num_syllables, prev_emph, curr_state, obsList, syllDict):
        probs = [i for i in self.O[curr_state]]
        for state in range(len(probs)):
            word = obsList[state]
            syll_count, emph_level = syllDict[word]
            emph = False
            if emph_level > 0:
                emph = True
            if syll_count > num_syllables:
                probs[state] = 0
            elif (emph == prev_emph) & (num_syllables is not 1):
                probs[state] = 0
        probs = [probs[i]/sum(probs) for i in range(len(probs))]
        return probs
    

def unsupervised_HMM(X, n_states):
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
