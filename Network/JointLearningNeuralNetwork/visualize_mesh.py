#!/usr/bin/env python3
"""
Visualize OBJ mesh files and render to images.

Methods:
1. Matplotlib 3D plot (always works, no GPU needed)
2. Trimesh scene rendering (if pyrender available)
3. Open3D visualization (if open3d available)

Usage:
    python visualize_mesh.py hand_mesh.obj --output render.png
    python visualize_mesh.py hand_mesh.obj --joints hand_mesh_joints.obj --output render.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


def load_obj(filepath):
    """Load OBJ file and return vertices and faces."""
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Handle different face formats: "f 1 2 3" or "f 1/1/1 2/2/2 3/3/3"
                face = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                    face.append(idx)
                faces.append(face)
    
    return np.array(vertices), np.array(faces)


def render_with_matplotlib(vertices, faces, joints=None, output_path=None, 
                          view_angle=(30, 45), figsize=(12, 10)):
    """
    Render mesh using matplotlib 3D plot.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        joints: (21, 3) optional joint positions
        output_path: path to save image
        view_angle: (elevation, azimuth) camera angles
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh polygons
    mesh_polygons = []
    for face in faces:
        polygon = vertices[face]
        mesh_polygons.append(polygon)
    
    # Add mesh
    mesh = Poly3DCollection(mesh_polygons, alpha=0.7)
    mesh.set_facecolor('peachpuff')
    mesh.set_edgecolor('gray')
    mesh.set_linewidth(0.1)
    ax.add_collection3d(mesh)
    
    # Add joints if provided
    if joints is not None:
        # Plot joints
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                  c='red', s=50, marker='o', label='Joints')
        
        # Draw skeleton connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        for i, j in connections:
            ax.plot3D([joints[i, 0], joints[j, 0]],
                     [joints[i, 1], joints[j, 1]],
                     [joints[i, 2], joints[j, 2]],
                     'b-', linewidth=2)
    
    # Set axis properties
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    # Set equal aspect ratio
    max_range = np.max([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    ]) / 2.0
    
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) / 2
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) / 2
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    ax.set_title('Hand Mesh Visualization')
    
    if joints is not None:
        ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved render to {output_path}")
    else:
        plt.show()
    
    plt.close()


def render_multiple_views(vertices, faces, joints=None, output_path=None):
    """Render mesh from multiple viewpoints."""
    fig = plt.figure(figsize=(16, 12))
    
    views = [
        (30, 45, 'Front-Right'),
        (30, 135, 'Front-Left'),
        (30, -45, 'Back-Right'),
        (90, 0, 'Top'),
        (0, 0, 'Side'),
        (0, 90, 'Front')
    ]
    
    for idx, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        # Create mesh polygons
        mesh_polygons = [vertices[face] for face in faces]
        
        mesh = Poly3DCollection(mesh_polygons, alpha=0.7)
        mesh.set_facecolor('peachpuff')
        mesh.set_edgecolor('gray')
        mesh.set_linewidth(0.1)
        ax.add_collection3d(mesh)
        
        # Add joints
        if joints is not None:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                      c='red', s=20, marker='o')
        
        # Set equal aspect
        max_range = np.max([
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min()
        ]) / 2.0
        
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) / 2
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) / 2
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-view render to {output_path}")
    else:
        plt.show()
    
    plt.close()


def try_trimesh_render(mesh_path, output_path):
    """Try to render using trimesh (higher quality)."""
    try:
        import trimesh
        from PIL import Image
        
        mesh = trimesh.load(mesh_path)
        
        # Create a scene
        scene = trimesh.Scene(mesh)
        
        # Try to render
        try:
            # This requires pyrender and OpenGL
            png = scene.save_image(resolution=[1024, 768])
            with open(output_path, 'wb') as f:
                f.write(png)
            print(f"Saved trimesh render to {output_path}")
            return True
        except Exception as e:
            print(f"Trimesh rendering not available: {e}")
            return False
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Visualize OBJ mesh files')
    parser.add_argument('mesh', type=str, help='Path to OBJ mesh file')
    parser.add_argument('--joints', type=str, default=None,
                        help='Path to joints OBJ file (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (default: show interactively)')
    parser.add_argument('--multi-view', action='store_true',
                        help='Render from multiple viewpoints')
    parser.add_argument('--elevation', type=float, default=30,
                        help='Camera elevation angle')
    parser.add_argument('--azimuth', type=float, default=45,
                        help='Camera azimuth angle')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mesh):
        print(f"Error: Mesh file not found: {args.mesh}")
        return
    
    print(f"Loading mesh: {args.mesh}")
    vertices, faces = load_obj(args.mesh)
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    
    joints = None
    if args.joints and os.path.exists(args.joints):
        print(f"Loading joints: {args.joints}")
        joints, _ = load_obj(args.joints)
        print(f"  Joints: {len(joints)}")
    
    # Set default output path
    if args.output is None:
        args.output = args.mesh.replace('.obj', '_render.png')
    
    # Try trimesh first (higher quality)
    # if not try_trimesh_render(args.mesh, args.output):
    
    # Fall back to matplotlib
    print("Rendering with matplotlib...")
    if args.multi_view:
        render_multiple_views(vertices, faces, joints, args.output)
    else:
        render_with_matplotlib(vertices, faces, joints, args.output,
                              view_angle=(args.elevation, args.azimuth))


if __name__ == "__main__":
    main()

