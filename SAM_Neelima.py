import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import torch
from xml.dom import minidom
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm
import time
import json
from multiprocessing import Pool, cpu_count
import gc

# --- OPTIMIZED CONFIG ---
TRAINING_DIR = 'training'
SAM_CHECKPOINT = 'sam_vit_b_01ec64.pth'
OUTPUT_BASE_DIR = 'batch_processed_dataset'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Speed optimization settings
BATCH_PROCESSING = True
REDUCED_VISUALIZATION = True  # Only save final results, not debug
PARALLEL_PROCESSING = True
MAX_WORKERS = min(4, cpu_count())  # Limit to prevent memory issues
MEMORY_OPTIMIZATION = True


# Output directories
BATCH_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, 'images_SAM_Neelima')
BATCH_ANNOTATIONS_DIR = os.path.join(OUTPUT_BASE_DIR, 'annotations_SAM_Neelima')
BATCH_SUMMARY_DIR = os.path.join(OUTPUT_BASE_DIR, 'summary')
BATCH_LOGS_DIR = os.path.join(OUTPUT_BASE_DIR, 'logs')

for dir_path in [OUTPUT_BASE_DIR, BATCH_IMAGES_DIR, BATCH_ANNOTATIONS_DIR, BATCH_SUMMARY_DIR, BATCH_LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

class OptimizedPreprocessor:
    """
    Lightweight version of preprocessor for batch processing
    """
    
    def __init__(self):
        self.hexbug_colors = {
            'blue': [{'lower': np.array([100, 50, 50]), 'upper': np.array([130, 255, 255])}],
            'green': [{'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255])}],
            'red': [{'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
                   {'lower': np.array([170, 50, 50]), 'upper': np.array([180, 255, 255])}],
            'yellow': [{'lower': np.array([20, 50, 50]), 'upper': np.array([30, 255, 255])}],
            'purple': [{'lower': np.array([130, 50, 50]), 'upper': np.array([170, 255, 255])}],
            'orange': [{'lower': np.array([10, 50, 50]), 'upper': np.array([20, 255, 255])}]
        }
    
    def quick_background_analysis(self, frame):
        """Lightweight background analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 60:
            return 'dark_road'
        elif mean_brightness > 160:
            return 'bright_arena'
        else:
            return 'mixed'
    
    def fast_preprocess(self, frame, bg_type):
        """Fast preprocessing without heavy operations"""
        if bg_type == 'dark_road':
            # Quick brightness enhancement
            enhanced = cv2.convertScaleAbs(frame, alpha=1.8, beta=40)
            return cv2.bilateralFilter(enhanced, 5, 50, 50)
        elif bg_type == 'bright_arena':
            # Quick contrast adjustment
            enhanced = cv2.convertScaleAbs(frame, alpha=0.9, beta=-15)
            return enhanced
        else:
            # Standard enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def detect_colors_fast(self, frame, points, max_samples=5):
        """Fast color detection with limited samples"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_colors = []
        
        # Limit samples for speed
        sample_points = points[:max_samples] if len(points) > max_samples else points
        
        for x, y in sample_points:
            sample_radius = 15
            x1, y1 = max(0, x-sample_radius), max(0, y-sample_radius)
            x2, y2 = min(frame.shape[1], x+sample_radius), min(frame.shape[0], y+sample_radius)
            
            roi_hsv = hsv[y1:y2, x1:x2]
            if roi_hsv.size == 0:
                continue
            
            for color_name, color_ranges in self.hexbug_colors.items():
                for color_range in color_ranges:
                    mask = cv2.inRange(roi_hsv, color_range['lower'], color_range['upper'])
                    if np.sum(mask) > 30:  # Reduced threshold for speed
                        if color_name not in detected_colors:
                            detected_colors.append(color_name)
                        break
                if color_name in detected_colors:
                    break
        
        return detected_colors[:3]  # Limit to 3 colors for speed

def optimized_sam_prediction(frame, x, y, predictor, preprocessor, bg_type):
    """
    Optimized SAM prediction for batch processing
    """
    try:
        # Fast preprocessing
        processed_frame = preprocessor.fast_preprocess(frame, bg_type)
        
        # Optimal resizing for SAM
        h, w = processed_frame.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            processed_frame = cv2.resize(processed_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            x_scaled, y_scaled = int(x * scale), int(y * scale)
        else:
            x_scaled, y_scaled = x, y
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Set SAM image
        predictor.set_image(rgb_frame)
        
        # Simplified prompting strategy for speed
        radius = 10
        points = np.array([
            [x_scaled, y_scaled],
            [x_scaled-radius, y_scaled],
            [x_scaled+radius, y_scaled],
            [x_scaled, y_scaled-radius],
            [x_scaled, y_scaled+radius]
        ])
        
        # Ensure points are within bounds
        h_new, w_new = rgb_frame.shape[:2]
        points[:, 0] = np.clip(points[:, 0], 0, w_new-1)
        points[:, 1] = np.clip(points[:, 1], 0, h_new-1)
        
        labels = np.array([1, 1, 1, 1, 1])
        
        # SAM prediction
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        
        # Quick mask selection
        best_mask = None
        best_score = -1
        
        for mask, score in zip(masks, scores):
            if not mask[y_scaled, x_scaled]:
                continue
            
            mask_area = np.sum(mask)
            
            # Simplified criteria for speed
            min_score = 0.15 if bg_type == 'dark_road' else 0.2
            min_area = 10
            
            if score > min_score and mask_area > min_area:
                if score > best_score:
                    best_score = score
                    best_mask = mask
        
        # Scale mask back if needed
        if best_mask is not None and max(h, w) > 1024:
            original_size = (frame.shape[1], frame.shape[0])
            best_mask = cv2.resize(best_mask.astype(np.uint8), original_size, 
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        
        return best_mask, best_score
        
    except Exception as e:
        return None, 0.0

def create_voc_xml_fast(filename, width, height, bbox, save_path):
    """Optimized XML creation"""
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = filename
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    obj = ET.SubElement(annotation, 'object')
    ET.SubElement(obj, 'name').text = 'hexbug'
    bndbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = str(bbox[0])
    ET.SubElement(bndbox, 'ymin').text = str(bbox[1])
    ET.SubElement(bndbox, 'xmax').text = str(bbox[2])
    ET.SubElement(bndbox, 'ymax').text = str(bbox[3])
    
    # Fast write without pretty formatting
    tree = ET.ElementTree(annotation)
    tree.write(save_path)

def process_single_video_batch(video_name, sam_predictor, preprocessor):
    """
    Optimized processing for single video in batch mode
    """
    start_time = time.time()
    
    # File paths
    video_path = os.path.join(TRAINING_DIR, f"{video_name}.mp4")
    csv_path = os.path.join(TRAINING_DIR, f"{video_name}.csv")
    
    if not os.path.exists(video_path) or not os.path.exists(csv_path):
        return {
            'video': video_name,
            'status': 'error',
            'message': 'Files not found',
            'hexbugs_processed': 0,
            'hexbugs_successful': 0,
            'processing_time': 0
        }
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Group by frame
        frame_groups = df.groupby('t')
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Quick color analysis (limited frames for speed)
        sample_points = []
        sample_frames_analyzed = 0
        max_frames_for_color = 3  # Reduced for speed
        
        for frame_num, frame_data in frame_groups:
            if sample_frames_analyzed >= max_frames_for_color:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                for _, row in frame_data.iterrows():
                    sample_points.append((int(row['x']), int(row['y'])))
                
                # Quick color detection
                if sample_frames_analyzed == 0:  # Only analyze first frame for speed
                    detected_colors = preprocessor.detect_colors_fast(frame, sample_points)
                
                sample_frames_analyzed += 1
        
        # Process all frames
        total_hexbugs = 0
        successful_hexbugs = 0
        processed_frames = 0
        
        for frame_num, frame_data in frame_groups:
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Quick background analysis
            bg_type = preprocessor.quick_background_analysis(frame)
            
            # Process each hexbug in frame
            for _, row in frame_data.iterrows():
                hexbug_id = int(row['hexbug'])
                x, y = int(row['x']), int(row['y'])
                
                total_hexbugs += 1
                
                # SAM prediction
                mask, score = optimized_sam_prediction(frame, x, y, sam_predictor, preprocessor, bg_type)
                
                if mask is not None and score > 0.1:
                    # Get bounding box
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(largest_contour)
                        bbox = (x_bbox, y_bbox, x_bbox + w_bbox, y_bbox + h_bbox)
                        
                        # Save results
                        img_name = f"{video_name}_frame{frame_num:04d}_hexbug{hexbug_id}.jpg"
                        img_path = os.path.join(BATCH_IMAGES_DIR, img_name)
                        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])  # Reduced quality for speed
                        
                        xml_path = os.path.join(BATCH_ANNOTATIONS_DIR, img_name.replace('.jpg', '.xml'))
                        create_voc_xml_fast(img_name, width, height, bbox, xml_path)
                        
                        successful_hexbugs += 1
            
            processed_frames += 1
            
            # Memory cleanup every 50 frames
            if MEMORY_OPTIMIZATION and processed_frames % 50 == 0:
                gc.collect()
        
        cap.release()
        
        processing_time = time.time() - start_time
        success_rate = (successful_hexbugs / total_hexbugs * 100) if total_hexbugs > 0 else 0
        
        return {
            'video': video_name,
            'status': 'success',
            'total_frames': processed_frames,
            'hexbugs_processed': total_hexbugs,
            'hexbugs_successful': successful_hexbugs,
            'success_rate': success_rate,
            'detected_colors': detected_colors if 'detected_colors' in locals() else [],
            'processing_time': processing_time,
            'fps': processed_frames / processing_time if processing_time > 0 else 0
        }
        
    except Exception as e:
        return {
            'video': video_name,
            'status': 'error',
            'message': str(e),
            'hexbugs_processed': 0,
            'hexbugs_successful': 0,
            'processing_time': time.time() - start_time
        }

def batch_process_all_videos():
    """
    Process entire dataset in batch mode with optimizations
    """
    print("🚀 BATCH PROCESSING ENTIRE DATASET")
    print("=" * 60)
    
    # Load SAM once
    print("Loading SAM model...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    
    # Initialize preprocessor
    preprocessor = OptimizedPreprocessor()
    
    # Find all video files
    video_files = glob(os.path.join(TRAINING_DIR, "*.mp4"))
    video_names = [os.path.splitext(os.path.basename(f))[0] for f in video_files]
    
    # Filter for videos that have corresponding CSV files
    valid_videos = []
    for video_name in video_names:
        csv_path = os.path.join(TRAINING_DIR, f"{video_name}.csv")
        if os.path.exists(csv_path):
            valid_videos.append(video_name)
    
    print(f"📊 Found {len(valid_videos)} valid video-CSV pairs")
    print(f"🔧 Processing settings:")
    print(f"   Device: {DEVICE}")
    print(f"   Batch processing: {BATCH_PROCESSING}")
    print(f"   Memory optimization: {MEMORY_OPTIMIZATION}")
    print(f"   Output directory: {OUTPUT_BASE_DIR}")
    
    # Process all videos
    all_results = []
    total_start_time = time.time()
    
    print(f"\n🎬 Processing {len(valid_videos)} videos...")
    
    for i, video_name in enumerate(tqdm(valid_videos, desc="Processing videos")):
        print(f"\n📹 [{i+1}/{len(valid_videos)}] Processing {video_name}...")
        
        result = process_single_video_batch(video_name, predictor, preprocessor)
        all_results.append(result)
        
        # Print quick status
        if result['status'] == 'success':
            print(f"   ✅ {result['hexbugs_successful']}/{result['hexbugs_processed']} hexbugs successful ({result['success_rate']:.1f}%) in {result['processing_time']:.1f}s")
        else:
            print(f"   ❌ Failed: {result['message']}")
        
        # Memory cleanup
        if MEMORY_OPTIMIZATION and i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    total_time = time.time() - total_start_time
    
    # Compile results
    successful_videos = [r for r in all_results if r['status'] == 'success']
    failed_videos = [r for r in all_results if r['status'] == 'error']
    
    total_hexbugs = sum(r['hexbugs_processed'] for r in successful_videos)
    total_successful = sum(r['hexbugs_successful'] for r in successful_videos)
    overall_success_rate = (total_successful / total_hexbugs * 100) if total_hexbugs > 0 else 0
    
    # Save detailed results
    summary = {
        'processing_summary': {
            'total_videos_found': len(valid_videos),
            'successful_videos': len(successful_videos),
            'failed_videos': len(failed_videos),
            'total_hexbugs_processed': total_hexbugs,
            'total_hexbugs_successful': total_successful,
            'overall_success_rate': overall_success_rate,
            'total_processing_time': total_time,
            'average_time_per_video': total_time / len(valid_videos) if valid_videos else 0
        },
        'video_results': all_results,
        'output_directories': {
            'images': BATCH_IMAGES_DIR,
            'annotations': BATCH_ANNOTATIONS_DIR,
            'summary': BATCH_SUMMARY_DIR
        }
    }
    
    # Save summary
    summary_path = os.path.join(BATCH_SUMMARY_DIR, 'batch_processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create CSV report
    csv_results = []
    for result in all_results:
        csv_results.append({
            'video_name': result['video'],
            'status': result['status'],
            'hexbugs_processed': result['hexbugs_processed'],
            'hexbugs_successful': result['hexbugs_successful'],
            'success_rate': result.get('success_rate', 0),
            'processing_time': result['processing_time'],
            'detected_colors': ','.join(result.get('detected_colors', []))
        })
    
    csv_report_path = os.path.join(BATCH_SUMMARY_DIR, 'video_processing_report.csv')
    pd.DataFrame(csv_results).to_csv(csv_report_path, index=False)
    
    # Print final summary
    print(f"\n🏆 BATCH PROCESSING COMPLETE!")
    print(f"=" * 60)
    print(f"📊 OVERALL RESULTS:")
    print(f"   Videos processed: {len(successful_videos)}/{len(valid_videos)} successful")
    print(f"   HexBugs processed: {total_hexbugs}")
    print(f"   HexBugs successful: {total_successful}")
    print(f"   Overall success rate: {overall_success_rate:.1f}%")
    print(f"   Total processing time: {total_time/60:.1f} minutes")
    print(f"   Average per video: {total_time/len(valid_videos):.1f} seconds")
    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"   Images: {BATCH_IMAGES_DIR}")
    print(f"   Annotations: {BATCH_ANNOTATIONS_DIR}")
    print(f"   Summary: {summary_path}")
    print(f"   CSV Report: {csv_report_path}")
    
    if failed_videos:
        print(f"\n❌ FAILED VIDEOS ({len(failed_videos)}):")
        for result in failed_videos:
            print(f"   {result['video']}: {result['message']}")
    
    return summary

def get_processing_stats():
    """
    Get statistics about already processed files (for resume functionality)
    """
    if not os.path.exists(BATCH_IMAGES_DIR):
        return 0, []
    
    processed_files = glob(os.path.join(BATCH_IMAGES_DIR, "*.jpg"))
    processed_videos = set()
    
    for file in processed_files:
        filename = os.path.basename(file)
        video_name = filename.split('_frame')[0]
        processed_videos.add(video_name)
    
    return len(processed_files), list(processed_videos)

if __name__ == "__main__":
    # Check if any processing was already done
    processed_count, processed_videos = get_processing_stats()
    
    if processed_count > 0:
        print(f"📊 Found {processed_count} already processed images from {len(processed_videos)} videos")
        response = input("Continue and process remaining videos? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            exit()
    
    # Start batch processing
    summary = batch_process_all_videos()
    
    print(f"\n✅ All done! Check {OUTPUT_BASE_DIR} for results.")
