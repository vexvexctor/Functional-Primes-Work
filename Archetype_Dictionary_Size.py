# Archetype Dictionary Size Creation Function ()
import torch
import numpy as np
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_approx_distance_to_convex_hull(P, q, epsilon):


    delta = np.max(cdist(P, P))  # Compute diameter ∆ of P
    t = P[np.argmin(np.linalg.norm(P - q, axis=1))]  # Start with the closest point to q

    for _ in range(int(1 / epsilon**2)):
        v = q - t
        projections = np.dot(P, v)
        p = P[np.argmax(projections)]
        # Set t_next to be the midpoint of the segment [t, p]
        t_next = (t + p) / 2

        # Update t if the new point t_next is different from the current t
        if np.array_equal(t, t_next):
            break
        else:
            t = t_next

        # Check if the approximate distance is within the threshold
        if np.linalg.norm(q - t) <= epsilon * delta:
            break

    return np.linalg.norm(q - t), t



def approximate_convex_hull(points, epsilon):
    """
    Approximate the convex hull of a set of points in d-dimensional space.

    Args:
    - points (np.array): Array of points where each point is in d-dimensional space.
    - epsilon (float): Tolerance factor for stopping the approximation.

    Returns:
    - np.array: An array representing points in the approximated convex hull.
    """
    if len(points) == 0:
        return np.array([])  # Early exit for empty input

    # Get an approximate diameter ∆0, such that ∆ ≤ ∆0 ≤ 2∆
    delta_0 = np.max(cdist(points, points))
    print(f"Delta (Diameter of Set Equals {delta_0})")
    adjusted_epsilon = 8 * epsilon ** (1/3) + epsilon
    print(f"Adjusted_epsilion (stop condition) equals {adjusted_epsilon}")

    print(f"Stopping Distance = {adjusted_epsilon * delta_0 / 2}")

    U = [min(points, key=lambda p: np.linalg.norm(p))] # We start with the "smallest" point (magnitude closest to zero), if these become negative this is also p free (we calcualte the furthest point)

    while True:
        max_distance = 0
        t_i = None
        for point in points:
            if not any(np.array_equal(point, x) for x in U):  # Check if point is not in U
                distance_to_U, _ = compute_approx_distance_to_convex_hull(np.array(U), point, epsilon)
                if distance_to_U > max_distance:
                    max_distance = distance_to_U
                    t_i = point

        if max_distance <= adjusted_epsilon * delta_0 / 2 or t_i is None:
        #if max_distance <= 0.05:
            print("Stopping condition met or no new point found further than max_distance.")
            break

        U.append(t_i)
        print(f"Added point {t_i} to U, Max distance this iteration: {max_distance}")

    return np.array(U)



# Way to visualize convex hull (2d) so we can make sure it is not bugging

import matplotlib.pyplot as plt
import numpy as np

def plot_convex_hull(points, convex_hull_coords):
    """
    Plots the given points and the convex hull for the first two dimensions.

    Args:
    - points: A NumPy array of shape (n, d) representing all points, where d >= 2.
    - convex_hull_coords: A NumPy array of shape (m, d) representing the coordinates
                          of the points forming the convex hull, where d >= 2.
    """
    if points.shape[1] < 2 or convex_hull_coords.shape[1] < 2:
        raise ValueError("The input arrays must have at least 2 dimensions.")

    # Extract the first two dimensions for plotting
    points_2d = points[:, :2]
    convex_hull_coords_2d = convex_hull_coords[:, :2]

    # Plot all points in the first two dimensions
    plt.scatter(points_2d[:, 0], points_2d[:, 1], color='blue', label='Points')

    # Plot convex hull points in the first two dimensions
    plt.scatter(convex_hull_coords_2d[:, 0], convex_hull_coords_2d[:, 1], color='red', label='Convex Hull')

    # Optional: Connect the convex hull points to visualize the hull in 2D
    # hull_coords_with_closure_2d = np.vstack([convex_hull_coords_2d, convex_hull_coords_2d[0]])  # Close the loop
    # plt.plot(hull_coords_with_closure_2d[:, 0], hull_coords_with_closure_2d[:, 1], color='red')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Convex Hull Visualization in 2D')
    plt.legend()
    plt.show()
