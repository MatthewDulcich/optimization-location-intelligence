import numpy as np
from collections import namedtuple

# Define a named tuple for the result
GDResult = namedtuple('GDResult', ['x', 'fun', 'success', 'message', 'niter'])

def gd_minimize(
        fun,  # Objective function
        x0,
        args=(),  # Additional arguments for the objective function
        constraints=None,
        bounds=None,
        options=None
    ):
    """
    Gradient descent implementation with memory management.

    Parameters:
    - fun: Objective function to minimize.
    - x0: Initial guess for the variables.
    - args: Tuple of additional arguments to pass to the objective function.
    - constraints: Constraints for the optimization (not implemented in this version).
    - bounds: Bounds for the variables (not implemented in this version).
    - options: Dictionary of options (e.g., max iterations).

    Returns:
    - GDResult: Named tuple containing optimized variables, final cost value, success flag, message, and iteration count.
    """
    x = np.array(x0, dtype=float)  # Ensure x0 is a numpy array of floats
    learning_rate = 0.01  # Standard learning rate
    max_iter = options.get('maxiter', 500) if options else 500  # Default to 500 iterations
    tolerance = 1e-6  # Standard tolerance for convergence
    momentum = 0.8  # Typical momentum factor
    velocity = np.zeros_like(x)  # Initialize velocity for momentum
    iter_count = 0  # Start iteration count
    prev_x = x.copy()  # Keep track of the previous x
    prev_cost = fun(x, *args)  # Evaluate the initial cost

    success = False
    message = "Maximum iterations reached without convergence."

    # Gradient descent loop
    while iter_count < max_iter:
        # Calculate gradient using finite differences
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += tolerance
            cost_plus = fun(x_plus, *args)
            grad[i] = (cost_plus - prev_cost) / tolerance
            del x_plus  # Clean up temporary array

        # Update with momentum
        velocity = momentum * velocity - learning_rate * grad
        x += velocity

        # Log progress
        if iter_count % 100 == 0:
            print(f"Iteration {iter_count}/{max_iter}, Cost: {prev_cost:.6f}")

        # Check for convergence
        if np.linalg.norm(x - prev_x) < tolerance:
            success = True
            message = "Optimization converged successfully."
            break

        prev_x = x.copy()
        prev_cost = fun(x, *args)
        iter_count += 1

        # Force garbage collection periodically
        if iter_count % 1000 == 0:
            import gc
            gc.collect()

    return GDResult(x=x, fun=prev_cost, success=success, message=message, niter=iter_count)