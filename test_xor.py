import numpy as np
from MedAI import NeuralNetwork

def test_xor():
    """Test the neural network on XOR problem"""
    print("Testing Neural Network on XOR Problem")
    print("=" * 40)
    
    # XOR dataset - the classic non-linear problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    print("XOR Dataset:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Expected: {y[i][0]}")
    
    # Create and train the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    print("\nTraining neural network...")
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    
    print("\nTesting predictions:")
    for i in range(len(X)):
        prediction = nn.forward(X[i:i+1])
        predicted_class = 1 if prediction[0][0] > 0.5 else 0
        print(f"Input: {X[i]}, Expected: {y[i][0]}, Predicted: {prediction[0][0]:.4f} ({predicted_class})")
    
    print("\nFinal weights and biases:")
    print("W1 (input to hidden):")
    print(nn.W1)
    print("b1 (hidden biases):")
    print(nn.b1)
    print("W2 (hidden to output):")
    print(nn.W2)
    print("b2 (output bias):")
    print(nn.b2)

if __name__ == "__main__":
    test_xor()
