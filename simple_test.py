print("Testing Python environment...")
print("=" * 30)

# Test basic Python functionality
print("1. Basic Python test:")
x = 5
y = 10
print(f"x + y = {x + y}")

# Test NumPy import
print("\n2. NumPy import test:")
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    print("NumPy imported successfully!")
    
    # Test basic NumPy operations
    print("\n3. NumPy operations test:")
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Array: {arr}")
    print(f"Mean: {np.mean(arr)}")
    
except ImportError as e:
    print(f"NumPy import failed: {e}")

print("\n4. Environment test complete!")
