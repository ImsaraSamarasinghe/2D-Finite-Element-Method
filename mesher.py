import numpy as np
import matplotlib.pyplot as plt

class RectangleMesh:
    """
    Contains functions for creating a rectangular finite element mesh.
    
    Attributes:
        __init__
        create_rectangular_mesh
        plot_mesh
    """
    def __init__(self,length=10,width=5,nx=10,ny=5):
        """
        Class Initializer with default values for dimensions
        and Spacing.

        Parameters:
            length (float): Length of the rectangle.
            width (float): Width of the rectangle.
            nx (int): Number of elements along the length.
            ny (int): Number of elements along the width.
        """
        self.length = length
        self.width = width
        self.nx = nx
        self.ny = ny
        self.nodes = None
        self.elements = None

    def create_rectangular_mesh(self):
        """
        Creates a rectangular finite element mesh.

        Parameters:
            self
        Returns:
            nodes (np.ndarray): Array of node coordinates.
            elements (np.ndarray): Array of elements defined by node indices.
        """
        # Calculate the number of nodes in each direction
        nx_nodes = self.nx + 1
        ny_nodes =self. ny + 1

        # Generate the node coordinates
        x_coords = np.linspace(0, self.length, nx_nodes)
        y_coords = np.linspace(0,self.width, ny_nodes)
        self.nodes = np.array([[x, y] for y in y_coords for x in x_coords])

        # Generate the elements
        self.elements = []
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = j * nx_nodes + i
                n2 = n1 + 1
                n3 = n1 + nx_nodes + 1
                n4 = n1 + nx_nodes
                self.elements.append([n1, n2, n3, n4])

        return np.array(self.nodes), np.array(self.elements)

    def plot_mesh(self):
        """
        Visualizes a rectangular finite element mesh.

        Parameters:
            self
        
        Returns:
            "mesh.png"
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        
        # Plot elements
        for element in self.elements:
            # Get the coordinates of the nodes for this element
            x_coords = self.nodes[element, 0]
            y_coords = self.nodes[element, 1]
            # Append the first point to close the loop
            x_coords = np.append(x_coords, x_coords[0])
            y_coords = np.append(y_coords, y_coords[0])
            # Plot the element as a polygon
            ax.plot(x_coords, y_coords, 'b-', lw=1)

        # Plot nodes
        ax.plot(self.nodes[:, 0], self.nodes[:, 1], 'ro', markersize=2)  # Plot nodes as red points

        # Add labels and grid
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        plt.title('FE Mesh Visualization')
        plt.savefig('mesh.png')
