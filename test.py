import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
x = np.random.randn(2000)
noise = np.random.randn(2000)
y = 3*x+2+noise
# vectorization approach: exploit elementwise operations of NumPy 
def make_prediction(X, W, b):
    return W*X + b

def visualize_data(X, Y):
    """Plot the data points
    Arguments:
        X: ndarray of shape (n_samples, )
            The values we observed
        Y: ndarray of shape (n_samples, )
            The true values of what we want to predict
    """
    plt.plot(X, Y, "o", alpha=0.2)
    plt.title("Data points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def visualize_prediction(X, Y, W, b):
    """Plot the data points along with the predicted line
    Arguments:
        X: ndarray of shape (n_samples, )
            The values we observed
        Y: ndarray of shape (n_samples, )
            The true values of what we want to predict
        W: np.float32
            The weight parameter to be optimized
        b: np.float32
            The bias parameter to be optimized
    """
    Y_preds = make_prediction(X, W, b)
    plt.plot(X, Y, "o", alpha=0.9)
    plt.plot(X, Y_preds)
    plt.show()

#visualize_data(x,y)
visualize_prediction(x,y,W=3, b=1)


