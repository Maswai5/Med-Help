import numpy as np

# Enhanced sigmoid activation function with numerical stability
def sigmoid(x):
    # Prevent overflow by clipping large values
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Enhanced Neural Network class with improvements
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Validate input parameters
        if not all(isinstance(size, int) and size > 0 for size in [input_size, hidden_size, output_size]):
            raise ValueError("All layer sizes must be positive integers")
        
        # Initialize weights with Xavier/Glorot initialization for better convergence
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b2 = np.zeros((1, output_size))
        
        # Store architecture for validation
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def _validate_input(self, X, expected_samples=None):
        """Validate input data shape and values"""
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if X.ndim != 2:
            raise ValueError("Input must be 2D array")
        
        if X.shape[1] != self.input_size:
            raise ValueError(f"Input must have {self.input_size} features, got {X.shape[1]}")
        
        if expected_samples and X.shape[0] != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples, got {X.shape[0]}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input contains NaN or infinite values")

    def forward(self, X):
        """Forward propagation with input validation"""
        self._validate_input(X)
        
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_pred = sigmoid(self.z2)
        return self.y_pred

    def backward(self, X, y, output):
        """Backward propagation with validation"""
        self._validate_input(X, y.shape[0])
        
        if not isinstance(y, np.ndarray) or y.shape[1] != self.output_size:
            raise ValueError(f"Target must be numpy array with {self.output_size} outputs")
        
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
        """Update parameters with gradient clipping for stability"""
        # Clip gradients to prevent explosion
        dw1 = np.clip(dw1, -1.0, 1.0)
        db1 = np.clip(db1, -1.0, 1.0)
        dw2 = np.clip(dw2, -1.0, 1.0)
        db2 = np.clip(db2, -1.0, 1.0)
        
        self.W1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, learning_rate, verbose=True):
        """Enhanced training with validation and progress monitoring"""
        self._validate_input(X, y.shape[0])
        
        if not isinstance(y, np.ndarray) or y.shape[1] != self.output_size:
            raise ValueError(f"Target must be numpy array with {self.output_size} outputs")
        
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            dw1, db1, dw2, db2 = self.backward(X, y, output)
            self.update_parameters(dw1, db1, dw2, db2, learning_rate)
            
            loss = mse_loss(y, output)
            losses.append(loss)
            
            if verbose and epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.6f}')
        
        return losses

    def predict(self, X, threshold=0.5):
        """Make predictions with optional thresholding for classification"""
        predictions = self.forward(X)
        if self.output_size == 1:  # Binary classification
            return (predictions > threshold).astype(int)
        return predictions

# Example usage with improved dataset
if __name__ == "__main__":
    print("Enhanced Neural Network Demo")
    print("=" * 40)
    
    # Generate better synthetic data (linearly separable)
    np.random.seed(42)
    n_samples = 200
    
    # Class 0: centered around (-1, -1)
    class0 = np.random.randn(n_samples // 2, 2) - 1
    # Class 1: centered around (1, 1)
    class1 = np.random.randn(n_samples // 2, 2) + 1
    
    X = np.vstack([class0, class1])
    y = np.vstack([np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))])
    
    # Create and train the enhanced neural network
    nn = NeuralNetwork(input_size=2, hidden_size=8, output_size=1)
    
    print("Training enhanced neural network...")
    losses = nn.train(X, y, epochs=1000, learning_rate=0.1)
    
    # Test the neural network
    test_X = np.random.randn(10, 2)
    predictions = nn.predict(test_X)
    probabilities = nn.forward(test_X)
    
    print("\nTest Predictions:")
    for i in range(len(test_X)):
        print(f"Input: {test_X[i]}, Probability: {probabilities[i][0]:.4f}, Prediction: {predictions[i][0]}")
    
    print(f"\nFinal training loss: {losses[-1]:.6f}")
    print("Enhanced neural network ready for medical AI applications!")
