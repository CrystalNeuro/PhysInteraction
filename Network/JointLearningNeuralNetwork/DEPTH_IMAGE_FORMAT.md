# Depth Image Format Requirements

## Problem Summary

The JointLearningNeuralNetwork model expects depth images in a **specific format**. Dataset images in different formats will fail or produce poor results.

## Required Format

| Property | Value | Description |
|----------|-------|-------------|
| **Size** | 240×320 pixels | Height × Width (fixed input size) |
| **Bit depth** | 16-bit | uint16 (0-65535 range) |
| **Format** | PNG | Lossless format preserving depth values |
| **Values** | Millimeters | Depth in mm from camera |
| **Typical range** | 200-5000 mm | Hand at 20-500cm from camera |

## Common Issues

### ❌ Issue 1: Wrong Image Size

**Problem:** Dataset images often come in higher resolutions (e.g., 480×640)

**Symptom:** 
- Model fails to process
- Poor inference results
- Memory errors

**Solution:** Resize to 240×320 using INTER_NEAREST interpolation

```bash
python convert_depth_image.py --input your_depth.png --output converted.png
```

### ❌ Issue 2: Wrong Bit Depth

**Problem:** 8-bit depth images (0-255 range)

**Symptom:**
- Loss of depth precision
- Poor joint detection

**Solution:** Convert to 16-bit before processing

### ❌ Issue 3: Wrong Units

**Problem:** Depth in meters (float) instead of millimeters (uint16)

**Symptom:**
- All depth values < 10
- Model produces incorrect results

**Solution:** Multiply by 1000 and convert to uint16

## Comparison: Working vs Non-Working Images

### Sample Image (✅ Works)
```
Shape:     240×320
Dtype:     uint16
Range:     2654-3463 mm
Non-zero:  7.1%
```

### Dataset Image (❌ Doesn't Work)
```
Shape:     480×640  ← WRONG SIZE!
Dtype:     uint16
Range:     515-3629 mm
Non-zero:  90.1%
```

### Converted Dataset Image (✅ Works)
```
Shape:     240×320  ← FIXED!
Dtype:     uint16
Range:     515-3629 mm
Non-zero:  90.0%
```

## How to Convert Your Images

### Method 1: Using the Conversion Script (Recommended)

```bash
cd Network/JointLearningNeuralNetwork

# Convert a single image
python convert_depth_image.py \
    --input your_depth.png \
    --output converted_depth.png

# Batch convert all images in a directory
for img in /path/to/dataset/*.png; do
    python convert_depth_image.py --input "$img" --output "converted_$(basename $img)"
done
```

### Method 2: Using Python Code

```python
import cv2
import numpy as np

def convert_depth(input_path, output_path):
    # Load depth image
    depth = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    # Resize to 240×320 (use INTER_NEAREST to preserve depth values)
    depth_resized = cv2.resize(depth, (320, 240), interpolation=cv2.INTER_NEAREST)
    
    # Ensure uint16
    if depth_resized.dtype != np.uint16:
        depth_resized = depth_resized.astype(np.uint16)
    
    # Save
    cv2.imwrite(output_path, depth_resized)

convert_depth("dataset_depth.png", "converted_depth.png")
```

## Testing Your Converted Image

After conversion, test with the inference server:

```bash
# 1. Start the inference server (if not already running)
python inference_server.py --gpu 0

# 2. Test the converted image (in another terminal)
python test_server.py --depth converted_depth.png

# 3. Generate hand mesh
python generate_hand_mesh.py --depth_image converted_depth.png --output hand_mesh.obj
```

## Expected Results

After proper conversion, you should see:
- ✅ 21 hand joints detected
- ✅ Hand/object segmentation mask generated
- ✅ Joint positions with reasonable coordinates (u: 0-320, v: 0-240, z: 200-5000mm)

## Troubleshooting

### Issue: Still getting poor results after conversion

**Check:**
1. Is there a hand visible in the depth image?
2. Is the hand in the foreground (not behind an object)?
3. Are depth values reasonable (200-5000mm range)?
4. Is the image properly saved as 16-bit PNG?

**Verify your image:**
```python
import cv2
img = cv2.imread("converted_depth.png", cv2.IMREAD_UNCHANGED)
print(f"Shape: {img.shape}")  # Should be (240, 320)
print(f"Dtype: {img.dtype}")  # Should be uint16
print(f"Range: [{img.min()}, {img.max()}]")  # Should be in mm
```

## Reference Camera Setup

The model was trained using:
- **Camera:** Intel RealSense SR300
- **Resolution:** 640×480 (then center-cropped/resized to 320×240)
- **Depth range:** 200-1500mm optimal
- **Frame rate:** 30 FPS

If using a different camera, ensure your depth images are converted to match this format.

## Files Generated

After running the conversion and testing:

```
frame_000018_depth_converted.png    ← Converted depth image (240×320)
frame_000018_result.png             ← Visualization with detected joints
depth_format_comparison.png         ← Side-by-side comparison
depth_histogram_comparison.png      ← Depth value distribution
```

## Summary

✅ **Always resize to 240×320 before inference**  
✅ **Use 16-bit PNG format**  
✅ **Depth values in millimeters**  
✅ **Use INTER_NEAREST interpolation when resizing**  

---

For more information, see:
- `convert_depth_image.py` - Conversion script
- `test_server.py` - Testing script
- `generate_hand_mesh.py` - Full pipeline script
