import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_pred = sigmoid(self.z2)
        return self.y_pred

    def backward(self, X, y, output):
        m = y.shape[0]
        dz2 = output - y
        dw2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * sigmoid_derivative(self.a1)
        dw1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
        return dw1, db1, dw2, db2

    def update_parameters(self, dw1, db1, dw2, db2, learning_rate):
        self.W1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            dw1, db1, dw2, db2 = self.backward(X, y, output)
            self.update_parameters(dw1, db1, dw2, db2, learning_rate)
            if epoch % 100 == 0:
                loss = mse_loss(y, output)
                print(f'Epoch {epoch}, Loss: {loss}')

# Example usage
if __name__ == "__main__":
    # Generate some synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = np.random.randint(2, size=(100, 1))  # Binary classification

    # Create and train the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=1000, learning_rate=0.1)

    # Test the neural network
    test_X = np.random.randn(10, 2)  # 10 test samples
    predictions = nn.forward(test_X)
    print("\nPredictions:", predictions)
