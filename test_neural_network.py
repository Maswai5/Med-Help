import numpy as np

# Test the neural network with a simple XOR problem
def test_xor_problem():
    print("Testing Neural Network with XOR Problem")
    print("=" * 40)
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Import the NeuralNetwork class
    from MedAI import NeuralNetwork
    
    # Create and train the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    print("Training neural network...")
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    
    print("\nTesting predictions:")
    for i in range(len(X)):
        prediction = nn.forward(X[i:i+1])
        print(f"Input: {X[i]}, Expected: {y[i][0]}, Predicted: {prediction[0][0]:.4f}")
    
    print("\nFinal weights and biases:")
    print("W1:", nn.W1)
    print("b1:", nn.b1)
    print("W2:", nn.W2)
    print("b2:", nn.b2)

if __name__ == "__main__":
    test_xor_problem()
