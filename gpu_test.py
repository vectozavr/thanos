import torch
import time

def measure_batched_operations(matrix_size=1024, batch_size=32, seed=42, epsilon=1e-5):
    results = []

    # Set the seed for reproducibility
    torch.manual_seed(seed)

    # Create random batched matrices for multiplication
    A = torch.randn(batch_size, matrix_size, matrix_size)
    B = torch.randn(batch_size, matrix_size, matrix_size)

    # Create random batched matrices for solving linear equations
    X = torch.randn(batch_size, matrix_size, matrix_size)
    Y = torch.randn(batch_size, matrix_size)

    # Add a small diagonal value to each batch matrix to avoid ill-conditioned systems
    for i in range(batch_size):
        X[i] += epsilon * torch.eye(matrix_size)

    # Test on CPU
    device = torch.device("cpu")

    A_device = A.to(device)
    B_device = B.to(device)
    X_device = X.to(device)
    Y_device = Y.to(device)

    # Measure batched matrix multiplication time on CPU
    start_time = time.time()
    for _ in range(32):
        C = torch.bmm(A_device, B_device)
    end_time = time.time()
    elapsed_time_mult = end_time - start_time
    results.append((device, 'batched_matrix_multiplication', elapsed_time_mult))
    print(f"CPU - Batched Matrix Multiplication: {elapsed_time_mult:.6f} seconds")

    # Measure batched solve linear equations time on CPU
    start_time = time.time()
    for _ in range(32):
        Z = torch.linalg.solve(X_device, Y_device)
    end_time = time.time()
    elapsed_time_solve = end_time - start_time
    results.append((device, 'batched_solve_linear_equations', elapsed_time_solve))
    print(f"CPU - Batched Solve Linear Equations: {elapsed_time_solve:.6f} seconds")
    print("######################################\n")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs:")

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        print()

        for i in range(num_gpus):
            device = torch.device(f"cuda:{i}")
            torch.cuda.set_device(device)

            # Transfer matrices to the current GPU
            A_device = A.to(device)
            B_device = B.to(device)
            X_device = X.to(device)
            Y_device = Y.to(device)

            # Warm up GPU with batched operations
            for _ in range(10):
                torch.bmm(A_device, B_device)
                torch.linalg.solve(X_device, Y_device)

            # Measure batched matrix multiplication time on GPU
            start_time = time.time()
            for _ in range(32):
                C = torch.bmm(A_device, B_device)
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time_mult = end_time - start_time
            results.append((device, 'batched_matrix_multiplication', elapsed_time_mult))
            print(f"GPU {i} - Batched Matrix Multiplication: {elapsed_time_mult:.6f} seconds")

            start_time = time.time()
            for _ in range(32):
                Z = torch.linalg.solve(X_device, Y_device)
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time_solve = end_time - start_time
            results.append((device, 'batched_solve_linear_equations', elapsed_time_solve))
            print(f"GPU {i} - Batched Solve Linear Equations: {elapsed_time_solve:.6f} seconds")

            print("######################################")

    return results


if __name__ == "__main__":
    matrix_size = 1024  # You can change this size to fit your needs
    batch_size = 512  # You can change this batch size to fit your needs
    seed = 42  # Seed for reproducibility
    epsilon = 1e-3  # Small value to add to the diagonal
    results = measure_batched_operations(matrix_size, batch_size, seed, epsilon)
