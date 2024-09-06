import os
import numpy as np
from matplotlib import pyplot as plt, animation

from pyvolutionary.models import OptimizationResult, Population


def plot(
    fitness_function: callable,
    pos_min: float,
    pos_max: float,
    evolution: list[Population],
):
    # Plotting preparation
    x = np.linspace(pos_min, pos_max, 80)
    y = np.linspace(pos_min, pos_max, 80)
    X, Y = np.meshgrid(x, y)
    Z = fitness_function([X, Y])

    # Plot the surface.
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Plot the wireframe
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Animation image placeholder
    images = []

    # Add plot for each generation
    for generation in evolution:
        x = np.zeros(len(generation.agents))
        y = np.zeros(len(generation.agents))
        z = np.zeros(len(generation.agents))
        for i, agent in enumerate(generation.agents):
            x[i], y[i] = agent.position
            z[i] = agent.cost
        image = ax.scatter3D(x, y, z, c='b')
        images.append([image])

    return fig, images


def animate(
    fitness_function: callable,
    optimization_result: OptimizationResult,
    pos_min: float,
    pos_max: float,
    filename: str
):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate and save the animation image
    figure, images = plot(fitness_function, pos_min, pos_max, optimization_result.evolution)
    animated_image = animation.ArtistAnimation(figure, images)
    animated_image.save(os.path.join(current_dir, "..", filename), writer='pillow')
