#!/usr/bin/env python3
import sys
import numpy as np

try:
    import gmsh
except ImportError:
    print("Installing gmsh...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gmsh", "-q"])
    import gmsh

try:
    import plotly.graph_objects as go
except ImportError:
    print("Installing plotly...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "-q"])
    import plotly.graph_objects as go

def visualize_gmsh_mesh(msh_file):
    """Load and visualize a Gmsh mesh file."""
    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    # Load the mesh
    gmsh.open(msh_file)

    # Get all nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)

    # Get surface elements for visualization
    triangles = []
    lines = []

    # Get 2D elements (surface triangles)
    elem_types_2d, elem_tags_2d, elem_connectivity_2d = gmsh.model.mesh.getElements(2)
    if len(elem_types_2d) > 0:
        for elem_type, connectivity in zip(elem_types_2d, elem_connectivity_2d):
            # Reshape connectivity based on element type (3 nodes for triangles, 4 for quads)
            if elem_type == 2:  # Triangle
                n_nodes = 3
            elif elem_type == 3:  # Quad
                n_nodes = 4
            else:
                continue

            conn = np.array(connectivity) - 1  # Convert to 0-indexed
            for i in range(0, len(conn), n_nodes):
                if i + n_nodes <= len(conn):
                    triangles.append(conn[i:i+n_nodes])

    gmsh.finalize()

    # Create visualization
    fig = go.Figure()

    # Visualize triangular elements as a mesh
    if len(triangles) > 0:
        triangles = np.array(triangles)

        # Extract x, y, z coordinates for the mesh
        x = coords[triangles, 0].flatten()
        y = coords[triangles, 1].flatten()
        z = coords[triangles, 2].flatten()

        # Create connectivity for Mesh3d
        i_indices = triangles[:, 0]
        j_indices = triangles[:, 1]
        k_indices = triangles[:, 2]

        fig.add_trace(go.Mesh3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            i=i_indices,
            j=j_indices,
            k=k_indices,
            opacity=0.8,
            color='cyan',
            name='Surface'
        ))

    # Update layout
    fig.update_layout(
        title=f"3D Mesh Visualization: {msh_file.split('/')[-1]}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        width=1200,
        height=900,
        hovermode='closest'
    )

    return fig

if __name__ == "__main__":
    msh_file = "/home/david/workspace/nn-cloaking/output/rayleigh3d_conical_cells10/_cloak_mesh_3d_full.msh"
    print(f"Loading mesh from {msh_file}...")

    fig = visualize_gmsh_mesh(msh_file)
    output_file = "/tmp/mesh_visualization.html"
    fig.write_html(output_file)
    print(f"✓ Visualization saved to {output_file}")
    print(f"Open in browser to interact with the 3D mesh (rotate, zoom, pan)")
