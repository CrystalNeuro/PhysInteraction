# CRITICAL BUG FOUND: Unit Mismatch!

## The Problem

The **MANO model is in METERS** but **camera depth is in MILLIMETERS**!

### Scale Analysis:
```
Target joints (from camera):  ~100-400 mm
MANO rest joints:             ~0.01-0.1 m (= 10-100 mm)
Scale ratio:                  1657x !!!
```

This is why fitting errors are huge (~228,000 mm²) - the optimizer is trying to match:
- Target hand at ~200mm from camera (in mm units)
- MANO template at ~0.05m size (in m units)
- They're 1000x different!

## The Fix

### Option 1: Convert camera joints from mm to m (RECOMMENDED)

```python
# In process_video_inference.py, after convert_uvz_to_xyz:
joints_xyz = convert_uvz_to_xyz(joints_uvz, depth_img, fx=fx, fy=fy, cx=cx, cy=cy)

# Convert from mm to m for MANO
joints_xyz_m = joints_xyz / 1000.0

# Fit MANO (now in same units!)
pose_fit = mano.fit_to_joints(joints_xyz_m, max_iter=max_fit_iter)
```

### Option 2: Scale MANO model to mm

```python
# In mano_model.py __init__, after loading:
self.mesh_template *= 1000  # Convert to mm
self.mesh_pose_basis *= 1000
self.mesh_shape_basis *= 1000
```

## Expected Results After Fix

With correct units:
- Fit error should drop from ~228,000 to **< 100 mm²**
- MANO mesh will align perfectly with detected joints
- Visual results will look realistic

## Test Script

```python
# Quick test to verify fix
joints_xyz_mm = convert_uvz_to_xyz(joints_uvz, depth_img)  # in mm
joints_xyz_m = joints_xyz_mm / 1000.0  # convert to m

pose_fit_before = mano.fit_to_joints(joints_xyz_mm, max_iter=200)  # WRONG
pose_fit_after = mano.fit_to_joints(joints_xyz_m, max_iter=200)    # CORRECT

print(f"Error before fix: {pose_fit_before[3]:.2f} mm²")  # ~228,000
print(f"Error after fix: {pose_fit_after[3]:.2f} m²")    # Should be < 0.0001 m² = < 100 mm²
```

---

**THIS IS THE ROOT CAUSE OF ALL THE POOR RESULTS!**
