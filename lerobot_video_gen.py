import os
import cv2
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import io
from PIL import Image

# Path to your LeRobot dataset
lerobot_dataset_path = '/home/wangxianhao/data/project/reasoning/openpi/EeureKaaaa/tabletop_dataset'

# Create video directories if they don't exist
os.makedirs(os.path.join(lerobot_dataset_path, 'videos/chunk-000/observation.images.ego_view'), exist_ok=True)
os.makedirs(os.path.join(lerobot_dataset_path, 'videos/chunk-000/observation.images.wrist_view'), exist_ok=True)

# Get all parquet files
parquet_files = sorted(glob.glob(os.path.join(lerobot_dataset_path, 'data/chunk-000/episode_*.parquet')))

# Process each episode
for parquet_file in tqdm(parquet_files, desc="Processing episodes"):
    # Extract episode index from filename
    episode_idx = int(os.path.basename(parquet_file).split('_')[1].split('.')[0])
    
    # Read the parquet file
    df = pd.read_parquet(parquet_file)
    
    # Print columns for the first file to help debugging
    if parquet_file == parquet_files[0]:
        print(f"Available columns in first parquet file: {df.columns.tolist()}")
    
    # Process main camera (image)
    main_frames = []
    wrist_frames = []
    
    # Extract images from each row
    for _, row in df.iterrows():
        # Process main camera image
        if 'image' in row:
            image_data = row['image']
            
            # Check if image_data is a dictionary with 'bytes' key
            if isinstance(image_data, dict) and 'bytes' in image_data:
                # Convert bytes to numpy array using PIL
                img_bytes = image_data['bytes']
                img = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img)
                main_frames.append(img_array)
            else:
                if parquet_file == parquet_files[0]:
                    print(f"Unexpected main image format: {type(image_data)}")
        
        # Process wrist camera image
        if 'wrist_image' in row:
            wrist_image_data = row['wrist_image']
            
            # Check if wrist_image_data is a dictionary with 'bytes' key
            if isinstance(wrist_image_data, dict) and 'bytes' in wrist_image_data:
                # Convert bytes to numpy array using PIL
                img_bytes = wrist_image_data['bytes']
                img = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img)
                wrist_frames.append(img_array)
            else:
                if parquet_file == parquet_files[0]:
                    print(f"Unexpected wrist image format: {type(wrist_image_data)}")
    
    # Process main camera frames
    if main_frames:
        # Create video path for main camera
        main_video_path = os.path.join(lerobot_dataset_path, f'videos/chunk-000/observation.images.ego_view/episode_{episode_idx:06d}.mp4')
        
        # Get image dimensions
        height, width = main_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Check if image is grayscale or color
        if len(main_frames[0].shape) == 2 or main_frames[0].shape[2] == 1:
            # Grayscale
            main_video = cv2.VideoWriter(main_video_path, fourcc, 10.0, (width, height), isColor=False)
        else:
            # Color
            main_video = cv2.VideoWriter(main_video_path, fourcc, 10.0, (width, height))
        
        # Write frames to video
        for frame in main_frames:
            # Convert to uint8 if needed
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Handle color conversion if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # OpenCV uses BGR, but PIL images are RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            main_video.write(frame)
        
        # Release video writer
        main_video.release()
        
        print(f"Created main camera video for episode {episode_idx}")
    else:
        print(f"No main camera images found in episode {episode_idx}")
    
    # Process wrist camera frames
    if wrist_frames:
        # Create video path for wrist camera
        wrist_video_path = os.path.join(lerobot_dataset_path, f'videos/chunk-000/observation.images.wrist_view/episode_{episode_idx:06d}.mp4')
        
        # Get image dimensions
        height, width = wrist_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Check if image is grayscale or color
        if len(wrist_frames[0].shape) == 2 or wrist_frames[0].shape[2] == 1:
            # Grayscale
            wrist_video = cv2.VideoWriter(wrist_video_path, fourcc, 10.0, (width, height), isColor=False)
        else:
            # Color
            wrist_video = cv2.VideoWriter(wrist_video_path, fourcc, 10.0, (width, height))
        
        # Write frames to video
        for frame in wrist_frames:
            # Convert to uint8 if needed
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Handle color conversion if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # OpenCV uses BGR, but PIL images are RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            wrist_video.write(frame)
        
        # Release video writer
        wrist_video.release()
        
        print(f"Created wrist camera video for episode {episode_idx}")
    else:
        print(f"No wrist camera images found in episode {episode_idx}")

print("All videos created successfully!")