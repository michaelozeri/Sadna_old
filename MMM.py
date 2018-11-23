class MMM:
    def __init__(self, signatures_data, initial_pi, input_x):
        self.signatures_data = signatures_data
        self.initial_pi = initial_pi
        self.input_x = input_x

    def getN(self):
        return len(self.signatures_data)

    def getM(self):
        return len(self.signatures_data[0])

    # on input data (sequence or sequences) do EM iterations until the model improvement is less
    # than  threshold , or until max_iterations iterations.
    def fit(data , threshold , max_iterations ):
        return 0

    # on input on input data (sequence or sequences), return log probability to see it
    def likelihood(data ):
        return 0