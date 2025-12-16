#!/home/ckelley/bundle-adjustment-compare/venv/bin/python3
"""
Create video from reprojection error frames using matplotlib animation.
Alternative to ffmpeg for systems without it installed.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, 'output')
frames_dir = os.path.join(output_dir, 'reproj_frames')

# Find all frame images
frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])

if not frame_files:
    print(f"No frames found in {frames_dir}")
    exit(1)

print(f"Found {len(frame_files)} frames")

# Load first frame to get dimensions
first_frame = np.array(Image.open(os.path.join(frames_dir, frame_files[0])))

# Create figure
fig, ax = plt.subplots(figsize=(12, 9))
ax.axis('off')
im = ax.imshow(first_frame)

def update_frame(frame_num):
    """Update function for animation"""
    frame_path = os.path.join(frames_dir, frame_files[frame_num])
    frame = np.array(Image.open(frame_path))
    im.set_data(frame)
    return [im]

# Create animation - 5 fps (200ms per frame)
anim = animation.FuncAnimation(
    fig, 
    update_frame, 
    frames=len(frame_files),
    interval=200,  # milliseconds between frames
    blit=True,
    repeat=True
)

# Save as MP4
output_path = os.path.join(output_dir, 'reprojection_errors.mp4')
print(f"Creating video at {output_path}...")

# Use pillow writer (doesn't require ffmpeg)
writer = animation.PillowWriter(fps=5)
anim.save(output_path.replace('.mp4', '.gif'), writer=writer)
print(f"Created GIF: {output_path.replace('.mp4', '.gif')}")

# Try ffmpeg writer if available
try:
    writer = animation.FFMpegWriter(fps=5, bitrate=2000, codec='libx264')
    anim.save(output_path, writer=writer)
    print(f"Created MP4: {output_path}")
except Exception as e:
    print(f"Could not create MP4 (ffmpeg not available): {e}")
    print("GIF created successfully as alternative")

plt.close()
print("Done!")
