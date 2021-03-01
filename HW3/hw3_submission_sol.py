import collections
import math
import numpy as np

class Gaussian_Naive_Bayes():
    def fit(self, X_train, y_train):
        """
        fit with training data
        Inputs:
            - X_train: A numpy array of shape (N, D) containing training data; there are N
                training samples each of dimension D.
            - y_train: A numpy array of shape (N,) containing training labels; y[i] = c
                means that X[i] has label 0 <= c < C for C classes.
                
        With the input dataset, function gen_by_class will generate class-wise mean and variance to implement bayes inference.

        Returns:
        None
        
        """
        self.x = X_train
        self.y = y_train  
        
        self.gen_by_class()


    def gen_by_class(self):
        """
        With the given input dataset (self.x, self.y), generate 3 dictionaries to calculate class-wise mean and variance of the data.
        - self.x_by_class : A dictionary of numpy arraies with the keys as each class label and values as data with such label.
        - self.mean_by_class : A dictionary of numpy arraies with the keys as each class label and values as mean of the data with such label.
        - self.std_by_class : A dictionary of numpy arraies with the keys as each class label and values as standard deviation of the data with such label.
        - self.y_prior : A numpy array of shape (C,) containing prior probability of each class
        """
        self.x_by_class = dict()
        self.mean_by_class = dict()
        self.std_by_class = dict()
        self.y_prior = None
        
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Generate dictionaries.
        # hint : to see all unique y labels, you might use np.unique function, e.g., np.unique(self.y)

        unique, counts = np.unique(self.y, return_counts=True)
        self.y_prior = np.zeros((1, len(unique)))
        numSamples = np.sum(counts)
        classIndex = 0
        while (classIndex < len(unique)):
            self.y_prior[0][classIndex] = counts[classIndex] / numSamples
            classIndex += 1

        sampleIndex = 0
        while(sampleIndex < len(self.x)):
            sampleClass = self.y[sampleIndex]
            if sampleClass in self.x_by_class:
                self.x_by_class[sampleClass] = np.vstack([self.x_by_class[sampleClass], self.x[sampleIndex]])
            else:
                self.x_by_class[sampleClass] = np.array(self.x[sampleIndex])
            sampleIndex += 1

        for key in self.x_by_class:
            self.mean_by_class[key] = self.mean(self.x_by_class[key])
            self.std_by_class[key] = self.std(self.x_by_class[key])


        # END_YOUR_CODE
        ############################################################
        ############################################################        

    def mean(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate mean of input x
        
        mean = np.mean(x, axis=0)
    
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return mean
    
    def std(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate standard deviation of input x, do not use np.std

        mean = self.mean(x)
        sample_minus_mean = x-mean
        square_difference = sample_minus_mean**2
        summation = np.sum(square_difference, axis=0) + 1E-7
        std = np.sqrt((1.0/(len(x)-1))*(summation))
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return std
    
    def calc_gaussian_dist(self, x, mean, std):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate gaussian probability of input x given mean and std
        coefficient = 1.0 / (std*math.sqrt(2.0*math.pi))
        exponent = np.exp(-0.5*((x-mean)/std)**2)

        return coefficient * exponent
        # END_YOUR_CODE
        ############################################################
        ############################################################

    def predict(self, x):
        """
        Use the acquired mean and std for each class to predict class for input x.
        Inputs:

        Returns:
        - prediction: Predicted labels for the data in x. prediction is (N, C) dimensional array, for N samples and C classes.
        """
            
        n = len(x)
        num_class = len(np.unique(self.y))
        prediction = np.zeros((n, num_class))
        
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x


        for sample in range(0,n):
            for c in range(0, num_class):
                prediction[sample][c] =  self.y_prior[0][c] + np.sum(np.log(self.calc_gaussian_dist(x[sample], self.mean_by_class[c], self.std_by_class[c])+ + 1E-20))


        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return prediction


class Neural_Network():
    def __init__(self, hidden_size = 64, output_size = 1):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.hidden_size = hidden_size
        self.output_size = output_size

    def fit(self, x, y, batch_size = 64, iteration = 2000, learning_rate = 1e-3):
        """
        Train this 2 layered neural network classifier using mini-batch stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - iteration: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.

        Use the given learning_rate, iteration, or batch_size for this homework problem.

        Returns:
        None
        """
        dim = x.shape[1]
        num_train = x.shape[0]

        #initialize W
        if self.W1 == None:
            self.W1 = 0.001 * np.random.randn(dim, self.hidden_size)
            self.b1 = 0

            self.W2 = 0.001 * np.random.randn(self.hidden_size, self.output_size)
            self.b2 = 0


        for it in range(iteration):
            batch_ind = np.random.choice(num_train, batch_size)

            x_batch = x[batch_ind]
            y_batch = y[batch_ind]

            loss, gradient = self.loss(x_batch, y_batch)

            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Update parameters with mini-batch stochastic gradient descent method

            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################

            y_pred = self.predict(x_batch)
            acc = np.mean(y_pred == y_batch)

            if it % 50 == 0:
                print('iteration %d / %d: accuracy : %f: loss : %f' % (it, iteration, acc, loss))

    def loss(self, x_batch, y_batch, reg = 1e-3):
            """
            Implement feed-forward computation to calculate the loss function.
            And then compute corresponding back-propagation to get the derivatives.

            Inputs:
            - X_batch: A numpy array of shape (N, D) containing a minibatch of N
              data points; each point has dimension D.
            - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
            - reg: hyperparameter which is weight of the regularizer.

            Returns: A tuple containing:
            - loss as a single float
            - gradient dictionary with four keys : 'dW1', 'db1', 'dW2', and 'db2'
            """
            gradient = {'dW1' : None, 'db1' : None, 'dW2' : None, 'db2' : None}


            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate y_hat which is probability of the instance is y = 0.

            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################


            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate loss and gradient

            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################
            return loss, gradient

    def activation(self, z):
        """
        Compute the ReLU output of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : output of ReLU(z)
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Implement ReLU

        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################

        return s

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : sigmoid of input
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE

        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################

        return s

    def predict(self, x):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate predicted y

        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return y_hat

    