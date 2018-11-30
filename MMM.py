from numpy import log, sum, amax, exp, shape
from scipy.special import logsumexp

# we dont want to update signatures array (itay asked) at this point so i made
# a global to set if to update the signatures data or not at this time
UPDATE_SIGNATURES_DATA = False


def create_b_array(input_x, m):
    length = len(input_x)
    b = [0 for i in range(m)]
    for i in range(length):
        b[input_x[i] - 1] += 1
    return b


def convert_to_log_scale(initial_pi):
    # find dimension of array to convert
    s = shape(initial_pi)
    if len(s) == 2:
        return [[log(xij) for xij in xi] for xi in initial_pi]
    else:
        return [log(xi) for xi in initial_pi]


def log_to_regular(param):
    return exp(param)


class MMM:
    def __init__(self, signatures_data, initial_pi, input_x):

        # defining the mmm
        self.log_signatures_data = convert_to_log_scale(signatures_data)
        self.log_initial_pi = convert_to_log_scale(initial_pi)

        # constants - don't change
        self.n = len(self.log_signatures_data)
        self.m = len(self.log_signatures_data[0])
        self.T = len(input_x)
        self.B = create_b_array(input_x, self.m)

        # are calculated each iteration
        self.E = [[0 for j in range(self.m)] for i in range(self.n)]
        self.A = [0 for i in range(self.n)]

    # on input data (sequence or sequences) do EM iterations until the model improvement is less
    # than  threshold , or until max_iterations iterations.
    def fit(self, input_x_data, threshold, max_iterations):
        old_convergence = self.likelihood(input_x_data)
        self.e_step()
        self.m_step(UPDATE_SIGNATURES_DATA)
        new_convergence = self.likelihood(input_x_data)
        while (new_convergence - old_convergence) > threshold:
            old_convergence = new_convergence
            self.e_step()
            self.m_step(UPDATE_SIGNATURES_DATA)
            new_convergence = self.likelihood(input_x_data)
        return

    def e_step(self):
        for i in range(self.n):
            for j in range(self.m):
                b_j_ = self.B[j]
                pi_i_ = log_to_regular(self.log_initial_pi[i])
                e_ij = log_to_regular(self.log_signatures_data[i][j])
                temp = 0
                for k in range(0, self.n):
                    temp += (log_to_regular(self.log_initial_pi[k]) * log_to_regular(self.log_signatures_data[k][j]))

                self.E[i][j] = b_j_ * ((pi_i_ * e_ij) / temp)

        for i in range(self.n):
            self.A[i] = sum(self.E, axis=0)[i]

    # checks convergence from formula
    # on input on input data (sequence or sequences), return log probability to see it
    def likelihood(self, input_x_data):
        convergence = 0
        for t in range(self.T):
            temp_log_sum_array = [0 for i in range(self.n)]
            for i in range(self.n):
                temp_log_sum_array[i] = self.log_initial_pi[i] + self.log_signatures_data[i][input_x_data[t]]
            convergence += logsumexp(temp_log_sum_array)
        return convergence

    def m_step(self, update_e_ij):
        for i in range(self.n):
            if update_e_ij:
                for j in range(self.m):
                    # numerically stable for pi
                    self.log_signatures_data[i][j] = log(self.E[i][j]) - log(sum(self.E, axis=1)[j])
            # numerically stable for pi
            self.log_initial_pi[i] = log(self.A[i]) - log(self.T)
