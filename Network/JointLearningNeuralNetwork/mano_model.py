"""
Python implementation of MANO hand model.
Loads the mano_r.json and generates 3D hand meshes from joint positions.
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

class MANOModel:
    def __init__(self, model_path):
        """Load MANO model from JSON file."""
        print(f"Loading MANO model from {model_path}...")
        with open(model_path, 'r') as f:
            data = json.load(f)
        
        # PCA basis for pose (45 x 12 or 45 x 45)
        self.pose_pca_basis = np.array(data['pose_pca_basis']).T  # Transpose for proper multiplication
        self.pose_pca_mean = np.array(data['pose_pca_mean'])
        
        # Joint regressor (16 x 778) - maps vertices to 16 MANO joints
        self.J_regressor = np.array(data['J_regressor'])
        
        # Full joint regressor (21 x 778) - maps vertices to 21 joints
        self.J_regressor_full = np.array(data['J_regressor_full'])
        
        # Skinning weights (778 x 16) - LBS weights
        self.skinning_weights = np.array(data['skinning_weights'])
        
        # Mesh deformation bases
        self.mesh_pose_basis = np.array(data['mesh_pose_basis'])  # 778 x 3 x 135
        self.mesh_shape_basis = np.array(data['mesh_shape_basis'])  # 778 x 3 x 10
        
        # Template mesh (778 x 3)
        self.mesh_template = np.array(data['mesh_template'])
        
        # Faces (1538 x 3)
        self.faces = np.array(data['faces'])
        
        # Kinematic tree parents
        self.parents = data['parents']
        
        self.n_vertices = 778
        self.n_joints = 16
        self.n_pose_pca = 12  # Use 12 PCA components
        self.n_shape = 10
        
        # Initialize shape and pose
        self.shape = np.zeros(self.n_shape)
        self.v_shaped = None
        self.J = None
        
        print(f"MANO model loaded: {self.n_vertices} vertices, {len(self.faces)} faces")
    
    def init_rest_model(self, shape=None):
        """Initialize the rest pose model with shape parameters."""
        if shape is None:
            shape = np.zeros(self.n_shape)
        self.shape = shape
        
        # Apply shape deformation to template
        shape_offset = np.einsum('ijk,k->ij', self.mesh_shape_basis, shape)
        self.v_shaped = self.mesh_template + shape_offset
        
        # Compute rest pose joint positions
        self.J = self.J_regressor @ self.v_shaped
        
        return self.v_shaped
    
    def rodrigues(self, r):
        """Convert axis-angle to rotation matrix using Rodrigues formula."""
        theta = np.linalg.norm(r)
        if theta < 1e-8:
            return np.eye(3)
        
        r = r / theta
        K = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    def get_posed_model(self, pose_pca, global_R, global_T):
        """
        Get the posed MANO mesh.
        
        Args:
            pose_pca: (12,) PCA pose parameters
            global_R: (3,) global rotation (axis-angle)
            global_T: (3,) global translation
            
        Returns:
            vertices: (778, 3) posed mesh vertices
        """
        if self.v_shaped is None:
            self.init_rest_model()
        
        # Convert PCA to full pose
        pose_pca = pose_pca[:self.n_pose_pca]
        full_pose = self.pose_pca_basis[:, :self.n_pose_pca] @ pose_pca + self.pose_pca_mean
        
        # Reshape to (15, 3) for 15 finger joints
        joint_rotations = full_pose.reshape(-1, 3)
        
        # Build rotation matrices
        R = [self.rodrigues(global_R)]  # Root rotation
        for i in range(15):
            R.append(self.rodrigues(joint_rotations[i]))
        
        # Apply pose blend shapes
        pose_bias = np.zeros(15 * 9)
        for i in range(15):
            rot = R[i + 1]
            for r in range(3):
                for c in range(3):
                    if r == c:
                        pose_bias[9 * i + 3 * r + c] = rot[r, c] - 1
                    else:
                        pose_bias[9 * i + 3 * r + c] = rot[r, c]
        
        # Add pose deformation
        v_posed = self.v_shaped.copy()
        pose_offset = np.einsum('ijk,k->ij', self.mesh_pose_basis, pose_bias)
        v_posed = v_posed + pose_offset
        
        # Forward kinematics
        G = [None] * self.n_joints
        
        # Root transform
        G[0] = np.eye(4)
        G[0][:3, :3] = R[0]
        G[0][:3, 3] = self.J[0] + global_T
        
        # Propagate through kinematic chain
        for i in range(1, self.n_joints):
            parent = self.parents[i]
            local_transform = np.eye(4)
            local_transform[:3, :3] = R[i]
            local_transform[:3, 3] = self.J[i] - self.J[parent]
            G[i] = G[parent] @ local_transform
        
        # Remove rest pose
        for i in range(self.n_joints):
            G[i][:3, 3] = G[i][:3, 3] - G[i][:3, :3] @ self.J[i]
        
        # Linear blend skinning
        T = np.zeros((self.n_vertices, 4, 4))
        for i in range(self.n_joints):
            T += self.skinning_weights[:, i:i+1, None] * G[i][None, :, :]
        
        # Apply transforms
        v_homo = np.hstack([v_posed, np.ones((self.n_vertices, 1))])
        vertices = np.einsum('ijk,ik->ij', T, v_homo)[:, :3]
        
        return vertices
    
    def get_posed_joints(self, pose_pca, global_R, global_T):
        """Get the 21 posed joint positions."""
        vertices = self.get_posed_model(pose_pca, global_R, global_T)
        joints = self.J_regressor_full @ vertices
        return joints
    
    def fit_to_joints(self, target_joints, max_iter=100):
        """
        Fit MANO parameters to target 21 joint positions.
        
        Args:
            target_joints: (21, 3) target joint positions
            max_iter: maximum optimization iterations
            
        Returns:
            pose_pca: (12,) optimized pose PCA parameters
            global_R: (3,) optimized global rotation
            global_T: (3,) optimized global translation
        """
        if self.v_shaped is None:
            self.init_rest_model()
        
        # Initialize parameters
        x0 = np.zeros(self.n_pose_pca + 6)  # 12 pose + 3 rotation + 3 translation
        
        # Initialize translation to center of target joints
        x0[-3:] = target_joints.mean(axis=0) - self.J[0]
        
        def objective(x):
            pose_pca = x[:self.n_pose_pca]
            global_R = x[self.n_pose_pca:self.n_pose_pca+3]
            global_T = x[self.n_pose_pca+3:]
            
            pred_joints = self.get_posed_joints(pose_pca, global_R, global_T)
            error = np.sum((pred_joints - target_joints) ** 2)
            
            # Add regularization on pose
            reg = 0.01 * np.sum(pose_pca ** 2)
            
            return error + reg
        
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'disp': False})
        
        pose_pca = result.x[:self.n_pose_pca]
        global_R = result.x[self.n_pose_pca:self.n_pose_pca+3]
        global_T = result.x[self.n_pose_pca+3:]
        
        return pose_pca, global_R, global_T, result.fun
    
    def export_obj(self, filepath, vertices=None):
        """Export mesh as OBJ file."""
        if vertices is None:
            vertices = self.v_shaped if self.v_shaped is not None else self.mesh_template
        
        with open(filepath, 'w') as f:
            f.write("# MANO Hand Mesh\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(self.faces)}\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-indexed)
            for face in self.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"Exported mesh to {filepath}")


def test_mano():
    """Test the MANO model with sample joints."""
    import os
    
    # Path to MANO model
    mano_path = "../../InteractionReconstruction/InteractionReconstruction/data/mano/mano_r.json"
    
    if not os.path.exists(mano_path):
        print(f"MANO model not found at {mano_path}")
        return
    
    # Load model
    mano = MANOModel(mano_path)
    
    # Initialize with default shape
    mano.init_rest_model()
    
    # Export rest pose mesh
    mano.export_obj("hand_rest_pose.obj")
    print("Exported rest pose mesh to hand_rest_pose.obj")
    
    # Test with random pose
    pose_pca = np.random.randn(12) * 0.5
    global_R = np.array([0.1, 0.2, 0.3])
    global_T = np.array([0, 0, 400])
    
    vertices = mano.get_posed_model(pose_pca, global_R, global_T)
    mano.export_obj("hand_posed.obj", vertices)
    print("Exported posed mesh to hand_posed.obj")
    
    # Get joint positions
    joints = mano.get_posed_joints(pose_pca, global_R, global_T)
    print(f"\n21 Joint positions:\n{joints}")


if __name__ == "__main__":
    test_mano()

