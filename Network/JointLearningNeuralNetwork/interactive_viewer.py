#!/usr/bin/env python3
"""
Interactive 3D Hand Model Viewer

A web-based viewer that allows you to:
- View and rotate 3D hand mesh
- See skeleton joints
- Generate new meshes from depth images
- Real-time interaction

Usage:
    python interactive_viewer.py --port 5000
    
Then open http://localhost:5000 in your browser.
"""

import os
import json
import base64
import argparse
from flask import Flask, render_template_string, jsonify, request, send_from_directory
import numpy as np

app = Flask(__name__)

# HTML template with Three.js viewer
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Hand Model Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            overflow: hidden;
        }
        #container { 
            width: 100vw; 
            height: 100vh; 
            position: relative;
        }
        #info {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            max-width: 300px;
            z-index: 100;
        }
        #info h1 {
            font-size: 1.5em;
            margin-bottom: 10px;
            color: #4ecdc4;
        }
        #info p {
            font-size: 0.9em;
            line-height: 1.5;
            color: #ccc;
        }
        #controls {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 10px;
            z-index: 100;
        }
        button {
            background: #4ecdc4;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        button:hover {
            background: #45b7aa;
            transform: scale(1.05);
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.5em;
            color: #4ecdc4;
        }
        .checkbox-label {
            display: inline-flex;
            align-items: center;
            margin: 5px 10px;
            cursor: pointer;
        }
        .checkbox-label input {
            margin-right: 5px;
        }
        #stats {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 10px;
            font-size: 0.85em;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="loading">Loading 3D Hand Model...</div>
    </div>
    
    <div id="info">
        <h1>🖐️ 3D Hand Viewer</h1>
        <p>Interactive visualization of MANO hand model.</p>
        <p><strong>Controls:</strong></p>
        <p>• Left click + drag: Rotate</p>
        <p>• Scroll: Zoom</p>
        <p>• Right click + drag: Pan</p>
    </div>
    
    <div id="controls">
        <button onclick="resetCamera()">Reset View</button>
        <button onclick="toggleWireframe()">Toggle Wireframe</button>
        <button onclick="toggleJoints()">Toggle Joints</button>
        <button onclick="toggleSkeleton()">Toggle Skeleton</button>
        <br>
        <label class="checkbox-label">
            <input type="checkbox" id="autoRotate" onchange="toggleAutoRotate()"> Auto Rotate
        </label>
    </div>
    
    <div id="stats">
        <div>Vertices: <span id="vertexCount">0</span></div>
        <div>Faces: <span id="faceCount">0</span></div>
        <div>Joints: <span id="jointCount">0</span></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
    
    <script>
        let scene, camera, renderer, controls;
        let handMesh, jointSpheres = [], skeletonLines = [];
        let showWireframe = false, showJoints = true, showSkeleton = true;
        
        // Joint connections for skeleton
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],  // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],  // Index
            [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
            [0, 13], [13, 14], [14, 15], [15, 16],  // Ring
            [0, 17], [17, 18], [18, 19], [19, 20]   // Pinky
        ];
        
        function init() {
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
            camera.position.set(0, 0, 150);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(100, 100, 100);
            scene.add(directionalLight);
            
            const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
            backLight.position.set(-100, -100, -100);
            scene.add(backLight);
            
            // Grid helper
            const gridHelper = new THREE.GridHelper(200, 20, 0x444444, 0x333333);
            gridHelper.rotation.x = Math.PI / 2;
            scene.add(gridHelper);
            
            // Load mesh
            loadMesh();
            
            // Handle resize
            window.addEventListener('resize', onWindowResize);
            
            // Start animation
            animate();
        }
        
        function loadMesh() {
            fetch('/api/mesh')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    
                    // Create geometry
                    const geometry = new THREE.BufferGeometry();
                    const vertices = new Float32Array(data.vertices.flat());
                    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                    
                    // Add faces
                    const indices = new Uint32Array(data.faces.flat());
                    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                    geometry.computeVertexNormals();
                    
                    // Create material
                    const material = new THREE.MeshPhongMaterial({
                        color: 0xffccaa,
                        side: THREE.DoubleSide,
                        flatShading: false
                    });
                    
                    // Create mesh
                    handMesh = new THREE.Mesh(geometry, material);
                    scene.add(handMesh);
                    
                    // Center the mesh
                    geometry.computeBoundingBox();
                    const center = new THREE.Vector3();
                    geometry.boundingBox.getCenter(center);
                    handMesh.position.sub(center);
                    
                    // Update stats
                    document.getElementById('vertexCount').textContent = data.vertices.length;
                    document.getElementById('faceCount').textContent = data.faces.length;
                    
                    // Load joints
                    if (data.joints) {
                        createJoints(data.joints, center);
                        document.getElementById('jointCount').textContent = data.joints.length;
                    }
                })
                .catch(error => {
                    console.error('Error loading mesh:', error);
                    document.getElementById('loading').textContent = 'Error loading mesh';
                });
        }
        
        function createJoints(joints, offset) {
            const jointMaterial = new THREE.MeshBasicMaterial({ color: 0xff4444 });
            const jointGeometry = new THREE.SphereGeometry(2, 16, 16);
            
            joints.forEach((joint, i) => {
                const sphere = new THREE.Mesh(jointGeometry, jointMaterial);
                sphere.position.set(
                    joint[0] - offset.x,
                    joint[1] - offset.y,
                    joint[2] - offset.z
                );
                scene.add(sphere);
                jointSpheres.push(sphere);
            });
            
            // Create skeleton lines
            const lineMaterial = new THREE.LineBasicMaterial({ color: 0x4ecdc4, linewidth: 2 });
            
            connections.forEach(([i, j]) => {
                const points = [
                    jointSpheres[i].position.clone(),
                    jointSpheres[j].position.clone()
                ];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const line = new THREE.Line(geometry, lineMaterial);
                scene.add(line);
                skeletonLines.push(line);
            });
        }
        
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
        
        function resetCamera() {
            camera.position.set(0, 0, 150);
            controls.reset();
        }
        
        function toggleWireframe() {
            if (handMesh) {
                showWireframe = !showWireframe;
                handMesh.material.wireframe = showWireframe;
            }
        }
        
        function toggleJoints() {
            showJoints = !showJoints;
            jointSpheres.forEach(s => s.visible = showJoints);
        }
        
        function toggleSkeleton() {
            showSkeleton = !showSkeleton;
            skeletonLines.forEach(l => l.visible = showSkeleton);
        }
        
        function toggleAutoRotate() {
            controls.autoRotate = document.getElementById('autoRotate').checked;
            controls.autoRotateSpeed = 2.0;
        }
        
        init();
    </script>
</body>
</html>
'''


def load_obj(filepath):
    """Load OBJ file."""
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
                face = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1
                    face.append(idx)
                faces.append(face)
    
    return vertices, faces


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/mesh')
def get_mesh():
    """Return mesh data as JSON."""
    mesh_path = app.config.get('MESH_PATH', 'hand_from_depth.obj')
    joints_path = app.config.get('JOINTS_PATH', 'hand_from_depth_joints.obj')
    
    if not os.path.exists(mesh_path):
        # Try default mesh
        mesh_path = 'hand_rest.obj'
    
    if not os.path.exists(mesh_path):
        return jsonify({'error': 'Mesh not found'}), 404
    
    vertices, faces = load_obj(mesh_path)
    
    result = {
        'vertices': vertices,
        'faces': faces
    }
    
    # Load joints if available
    if os.path.exists(joints_path):
        joints, _ = load_obj(joints_path)
        result['joints'] = joints
    
    return jsonify(result)


@app.route('/api/generate', methods=['POST'])
def generate_mesh():
    """Generate new mesh from depth image."""
    # This could be extended to accept depth image and generate mesh
    return jsonify({'status': 'not implemented'})


def main():
    parser = argparse.ArgumentParser(description='Interactive 3D Hand Viewer')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--mesh', type=str, default='hand_from_depth.obj', help='Mesh file')
    parser.add_argument('--joints', type=str, default='hand_from_depth_joints.obj', help='Joints file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    
    args = parser.parse_args()
    
    app.config['MESH_PATH'] = args.mesh
    app.config['JOINTS_PATH'] = args.joints
    
    print(f"\n{'='*60}")
    print("🖐️  Interactive 3D Hand Model Viewer")
    print(f"{'='*60}")
    print(f"\nOpen your browser and go to:")
    print(f"  http://localhost:{args.port}")
    print(f"\nMesh: {args.mesh}")
    print(f"Joints: {args.joints}")
    print(f"\nPress Ctrl+C to stop the server")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

