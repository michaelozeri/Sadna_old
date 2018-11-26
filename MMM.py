import numpy as np
import scipy.special as sp

# we dont want to update signatures array (itay asked) at this point so i made
# a global to set if to update the signatures data or not at this time
UPDATE_SIGNATURES_DATA = False


def create_b_array(input_x):
    length = len(input_x)
    b = [0 for i in range(0, np.amax(input_x))]
    for i in range(0, length - 1):
        b[input_x[i] - 1] += 1
    return b


class MMM:
    def __init__(self, signatures_data, initial_pi, input_x):

        # constants - don't change
        self.B = create_b_array(input_x)
        self.n = len(self.signatures_data)
        self.m = len(self.signatures_data[0])
        self.T = len(input_x)

        # are calculated each iteration
        self.E = [[] for i in range(self.n)]
        self.A = [0 for i in range(self.n)]

        # defining the mmm
        self.signatures_data = signatures_data
        self.initial_pi = initial_pi

    # on input data (sequence or sequences) do EM iterations until the model improvement is less
    # than  threshold , or until max_iterations iterations.
    def fit(self, data, threshold, max_iterations):
        old_convergence = self.check_convergence()
        self.e_step()
        self.m_step(UPDATE_SIGNATURES_DATA)
        new_convergence = self.check_convergence()
        while (new_convergence - old_convergence) > threshold:
            old_convergence = new_convergence
            self.e_step()
            self.m_step(UPDATE_SIGNATURES_DATA)
            new_convergence = self.check_convergence()
        return

    def e_step(self):
        for i in range(self.n):
            for j in range(self.m):
                b_j_ = self.B[j]
                pi_i_ = self.initial_pi[i]
                e_ij = self.signatures_data[i][j]
                temp = 0
                for k in range(0, self.n - 1):
                    temp += (self.initial_pi[k] * self.signatures_data[k][j])

                self.E[i][j] = b_j_ * ((pi_i_ * e_ij) / temp)

        for i in range(self.n):
            self.A[i] = np.sum(self.E, axis=0)[i]

    # checks convergence from formula
    def check_convergence(self):
        convergence = 0
        for i in range(self.T):
            # convergence += sp.logsumexp(np.log(initial_pi+))
            convergence+=0
            # TODO: understand from mail what to do here by answer. need to use function on scipy
        return convergence

    # on input on input data (sequence or sequences), return log probability to see it
    def likelihood(self, data):
        # TODO: finish this function
        return 0

    def m_step(self, update_e_ij):
        for i in range(self.n):
            if (update_e_ij):
                for j in range(self.m):
                    self.signatures_data[i][j] = self.E[i][j] / (np.sum(self.E, axis=1)[j])
            self.initial_pi[i] = self.A[i] / np.sum(self.A)
