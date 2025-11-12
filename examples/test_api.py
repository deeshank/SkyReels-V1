#!/usr/bin/env python3
"""
Test script for SkyReels Video Generation API
"""
import requests
import time
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "https://dr9b1f4vx6l8rp-8000.proxy.runpod.net"
OUTPUT_DIR = Path("./generated_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

def test_t2v_generation():
    """Test text-to-video generation with async polling"""
    print("=" * 60)
    print("Testing Text-to-Video Generation")
    print("=" * 60)
    
    # Prepare request
    payload = {
        "prompt": "FPS-24, A serene lake at sunset with mountains in the background",
        "task_type": "t2v",
        "height": 544,
        "width": 960,
        "num_frames": 97,
        "num_inference_steps": 30,
        "guidance_scale": 6.0,
        "embedded_guidance_scale": 1.0,
        "seed": -1,
        "fps": 24,
        "negative_prompt": "Aerial view, overexposed, low quality, deformation, bad composition",
        "cfg_for": False
    }
    
    print(f"\nPrompt: {payload['prompt']}")
    print(f"Resolution: {payload['width']}x{payload['height']}")
    print(f"Frames: {payload['num_frames']}")
    print(f"Steps: {payload['num_inference_steps']}")
    
    # Send generation request
    print("\nSending generation request...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        job_id = result['job_id']
        print(f"✓ Job queued successfully")
        print(f"✓ Job ID: {job_id}")
        print(f"✓ Status: {result['status']}")
        
        # Poll for completion
        print("\nPolling for completion...")
        poll_interval = 5  # seconds
        max_wait_time = 600  # 10 minutes
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait_time:
                print(f"✗ Timeout: Job did not complete within {max_wait_time}s")
                return False
            
            # Check status
            status_response = requests.get(
                f"{API_BASE_URL}/status/{job_id}",
                timeout=10
            )
            status_response.raise_for_status()
            status = status_response.json()
            
            current_status = status['status']
            progress = status.get('progress', 'Unknown')
            
            print(f"  [{int(elapsed)}s] Status: {current_status} - {progress}")
            
            if current_status == 'completed':
                print(f"\n✓ Video generation completed!")
                print(f"✓ Total time: {elapsed:.2f}s")
                print(f"✓ Generation time: {status.get('generation_time', 'N/A')}s")
                
                # Download video
                video_url = f"{API_BASE_URL}{status['video_path']}"
                output_file = OUTPUT_DIR / f"{job_id}.mp4"
                
                print(f"\nDownloading video from {video_url}...")
                video_response = requests.get(video_url, timeout=60)
                video_response.raise_for_status()
                
                with open(output_file, 'wb') as f:
                    f.write(video_response.content)
                
                print(f"✓ Video saved to: {output_file}")
                print(f"✓ File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
                
                return True
                
            elif current_status == 'failed':
                print(f"\n✗ Generation failed: {status.get('message', 'Unknown error')}")
                return False
            
            # Wait before next poll
            time.sleep(poll_interval)
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def check_health():
    """Check API health"""
    print("\nChecking API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        health = response.json()
        print(f"✓ Status: {health['status']}")
        print(f"✓ Available models: {health['available_models']}")
        print(f"✓ GPU count: {health['gpu_count']}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("\n🎬 SkyReels API Test Script\n")
    
    # Check health first
    if not check_health():
        print("\n❌ API is not healthy. Exiting.")
        sys.exit(1)
    
    # Run T2V test
    print("\n")
    success = test_t2v_generation()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Test failed")
        print("=" * 60)
        sys.exit(1)
