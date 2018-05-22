import numpy as np
import mltools as ml
import matplotlib.pyplot as plt


# Fix the required "not implemented" functions for the homework ("TODO")

################################################################################
## LOGISTIC REGRESSION BINARY CLASSIFIER #######################################
################################################################################


class logisticClassify2(ml.classifier):
    """A binary (2-class) logistic regression classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier 
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for logisticClassify2 object.  

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array 
        """
        self.classes = [0, 1]              # (default to 0/1; replace during training)
        self.theta = np.array([])         # placeholder value before training

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"


## METHODS ################################################################

    def sigmoid(self, x):
        "Numerically stable sigmoid function."
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = np.exp(x)
            return z / (1 + z)

    def plotBoundary(self, X, Y):
        """ Plot the (linear) decision boundary of the classifier, along with data """
        if len(self.theta) != 3:
            raise ValueError('Data & model must be 2D')
        ax = X.min(0), X.max(0)
        ax = (ax[0][0], ax[1][0], ax[0][1], ax[1][1])
        x1b = np.array([ax[0],ax[1]])  # at X1 = points in x1b
        x2b = -(self.theta[0] + self.theta[1]*x1b) / self.theta[2]      # find x2 values as a function of x1's values
        # Now plot the data and the resulting boundary:
        A = Y == 1  # and plot it:
        plt.plot(X[A, 0], X[A, 1], 'b.', X[~A, 0], X[~A, 1], 'r.', x1b, x2b, 'k-')
        plt.axis(ax)
        plt.show()

    def predictSoft(self, X):
        """ Return the probability of each class under logistic regression """
        raise NotImplementedError
        ## You do not need to implement this function.
        ## If you *want* to, it should return an Mx2 numpy array "P", with 
        ## P[:,1] = probability of class 1 = sigma( theta*X )
        ## P[:,0] = 1 - P[:,1] = probability of class 0 
        return P

    def predict(self, X):
        """ Return the predictied class of each data point in X"""
        length = len(X)
        r = np.zeros(length)
        Yhat = np.zeros(length)

        for i in range(length):
            r[i] = self.theta[0] + self.theta[1] * X[i, 0] + self.theta[2] * X[i, 1]
            if r[i] < 0:
                Yhat[i] = self.classes[0]
            else:
                Yhat[i] = self.classes[1]

        return Yhat


    def train(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000, plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        M,N = X.shape                     # initialize the model if necessary:
        self.classes = np.unique(Y)       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X))   # XX is X, but with an extra column of ones
        YY = ml.toIndex(Y,self.classes)  # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        # init loop variables:
        epoch=0; done=False; Jnll=[]; J01=[]; 
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                ri    = XX[i, :].dot(self.theta)
                sigma = self.sigmoid(ri)
                gradi = (sigma - YY[i]) * XX[i, :]
                self.theta -= stepsize * gradi;  # take a gradient step

            J01.append( self.err(X,Y) )  # evaluate the current error rate

            J = np.zeros(M)
            for j in range(M):
                rj = XX[j, :].dot(self.theta)
                if YY[j] == 1:
                    J[j] = -YY[j] * np.log(self.sigmoid(rj))
                else:
                    J[j] = -(1-YY[j]) * np.log(1-self.sigmoid(rj))

            Jnll.append(np.mean(J))

            plt.figure(1); plt.plot(Jnll,'b-',J01,'r-'); plt.draw();    # plot losses
            if N==2: plt.figure(2); self.plotBoundary(X,Y);
            plt.draw(); # & predictor if 2D
            plt.pause(.01);                    # let OS draw the plot

            ## For debugging: you may want to print current parameters & losses
            # print self.theta, ' => ', Jsur[-1], ' / ', J01[-1]  
            # raw_input()   # pause for keystroke

            if epoch > stopEpochs:
                done = True
            if epoch > 2 and abs(Jnll[-1] - Jnll[-2]) < stopTol:  # or if Jnll not changing between epochs ( < stopTol )
                done = True

    def trainL2(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000, plot=None, alpha=2):

        M, N = X.shape
        self.classes = np.unique(Y)
        XX = np.hstack((np.ones((M, 1)), X))
        YY = ml.toIndex(Y, self.classes)
        if len(self.theta) != N + 1:
            self.theta = np.random.rand(N + 1)
        # init loop variables:
        epoch = 0
        done = False
        Jnll = []
        J01 = []
        while not done:
            stepsize, epoch = initStep * 2.0 / (2.0 + epoch), epoch + 1
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                ri = XX[i, :].dot(self.theta)
                sigma = self.sigmoid(ri)
                gradi = (sigma - YY[i]) * XX[i, :] + alpha * 2 * self.theta
                self.theta -= stepsize * gradi

            J01.append(self.err(X, Y))

            J = np.zeros(M)
            for j in range(M):
                rj = XX[j, :].dot(self.theta)
                if YY[j] == 1:
                    J[j] = -YY[j] * np.log(self.sigmoid(rj))
                else:
                    J[j] = -(1 - YY[j]) * np.log(1 - self.sigmoid(rj))

            Jsur = np.mean(J) + self.theta.dot(self.theta) * alpha
            Jnll.append(Jsur)

            plt.figure(1)
            plt.plot(Jnll, 'b-', J01, 'r-')
            plt.draw()
            if N == 2: plt.figure(2); self.plotBoundary(X, Y);
            plt.draw();  # & predictor if 2D
            plt.pause(.01)

            ## For debugging: you may want to print current parameters & losses
            # print self.theta, ' => ', Jsur[-1], ' / ', J01[-1]
            # raw_input()   # pause for keystroke

            if epoch > stopEpochs:
                done = True
            if epoch > 2 and abs(Jnll[-1] - Jnll[-2]) < stopTol:  # or if Jnll not changing between epochs ( < stopTol )
                done = True

################################################################################
################################################################################
################################################################################

