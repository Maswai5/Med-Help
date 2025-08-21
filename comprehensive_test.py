import numpy as np
import time
from MedAI import NeuralNetwork, sigmoid, sigmoid_derivative, mse_loss

def test_edge_cases():
    """Test neural network with edge case inputs"""
    print("Testing Edge Cases")
    print("=" * 40)
    
    # Test with very small values
    X_small = np.array([[1e-10, 1e-10], [1e-10, 1e-10]])
    y_small = np.array([[0], [1]])
    
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    output = nn.forward(X_small)
    print(f"Small inputs test - Output shape: {output.shape}")
    
    # Test with very large values
    X_large = np.array([[1e10, 1e10], [1e10, 1e10]])
    output = nn.forward(X_large)
    print(f"Large inputs test - Output shape: {output.shape}")
    
    print("Edge case testing completed\n")

def test_network_architectures():
    """Test different network architectures"""
    print("Testing Different Network Architectures")
    print("=" * 50)
    
    architectures = [
        (2, 2, 1),   # Small hidden layer
        (2, 8, 1),   # Larger hidden layer
        (2, 16, 1),  # Even larger hidden layer
        (2, 4, 2),   # Multiple outputs
    ]
    
    X = np.random.randn(10, 2)
    
    for input_size, hidden_size, output_size in architectures:
        nn = NeuralNetwork(input_size, hidden_size, output_size)
        output = nn.forward(X)
        print(f"Architecture {input_size}-{hidden_size}-{output_size}: Output shape {output.shape}")
    
    print("Architecture testing completed\n")

def test_learning_rates():
    """Test different learning rates"""
    print("Testing Different Learning Rates")
    print("=" * 40)
    
    X = np.random.randn(20, 2)
    y = np.random.randint(2, size=(20, 1))
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    
    for lr in learning_rates:
        nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
        
        # Train for a few epochs to see convergence
        initial_loss = mse_loss(y, nn.forward(X))
        nn.train(X, y, epochs=100, learning_rate=lr)
        final_loss = mse_loss(y, nn.forward(X))
        
        print(f"LR {lr}: Initial loss {initial_loss:.4f}, Final loss {final_loss:.4f}")
    
    print("Learning rate testing completed\n")

def test_convergence():
    """Test convergence on different datasets"""
    print("Testing Convergence on Various Datasets")
    print("=" * 50)
    
    # Linear separable data
    X_linear = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    y_linear = np.array([[0], [0], [1], [1]])
    
    # XOR problem (non-linear)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    datasets = [
        ("Linear", X_linear, y_linear),
        ("XOR", X_xor, y_xor),
        ("Random", np.random.randn(10, 2), np.random.randint(2, size=(10, 1)))
    ]
    
    for name, X, y in datasets:
        nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
        
        initial_loss = mse_loss(y, nn.forward(X))
        nn.train(X, y, epochs=1000, learning_rate=0.1)
        final_loss = mse_loss(y, nn.forward(X))
        
        print(f"{name} dataset: Initial {initial_loss:.4f}, Final {final_loss:.4f}, Improvement: {((initial_loss - final_loss)/initial_loss*100):.1f}%")
    
    print("Convergence testing completed\n")

def test_performance():
    """Test performance and memory usage"""
    print("Testing Performance and Memory Usage")
    print("=" * 45)
    
    # Larger dataset for performance testing
    X_large = np.random.randn(1000, 5)
    y_large = np.random.randint(2, size=(1000, 1))
    
    nn = NeuralNetwork(input_size=5, hidden_size=10, output_size=1)
    
    # Time forward pass
    start_time = time.time()
    output = nn.forward(X_large)
    forward_time = time.time() - start_time
    
    # Time training for 10 epochs
    start_time = time.time()
    nn.train(X_large, y_large, epochs=10, learning_rate=0.1)
    train_time = time.time() - start_time
    
    print(f"Forward pass time (1000 samples): {forward_time:.4f}s")
    print(f"Training time (10 epochs): {train_time:.4f}s")
    print(f"Memory usage - Weights: {nn.W1.nbytes + nn.W2.nbytes} bytes")
    print(f"Memory usage - Biases: {nn.b1.nbytes + nn.b2.nbytes} bytes")
    
    print("Performance testing completed\n")

def test_error_handling():
    """Test error handling for invalid inputs"""
    print("Testing Error Handling")
    print("=" * 30)
    
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    # Test with invalid shapes
    try:
        X_invalid = np.array([1, 2, 3])  # Wrong shape
        output = nn.forward(X_invalid)
        print("Invalid shape test: PASSED")
    except Exception as e:
        print(f"Invalid shape test: FAILED - {type(e).__name__}")
    
    # Test with NaN values
    try:
        X_nan = np.array([[1, 2], [np.nan, 4]])
        output = nn.forward(X_nan)
        print("NaN values test: PASSED")
    except Exception as e:
        print(f"NaN values test: FAILED - {type(e).__name__}")
    
    print("Error handling testing completed\n")

if __name__ == "__main__":
    print("Comprehensive Neural Network Testing")
    print("=" * 40)
    print()
    
    test_edge_cases()
    test_network_architectures()
    test_learning_rates()
    test_convergence()
    test_performance()
    test_error_handling()
    
    print("All comprehensive tests completed!")
