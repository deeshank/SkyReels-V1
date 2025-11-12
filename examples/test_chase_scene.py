#!/usr/bin/env python3
"""
Generate a dramatic chase scene: Tiger chasing cow, cow chasing elephant
"""
import requests
import time
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "https://dr9b1f4vx6l8rp-8000.proxy.runpod.net"
OUTPUT_DIR = Path("./generated_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_chase_scene():
    """Generate an epic chase scene video"""
    print("=" * 70)
    print("🎬 Generating Epic Chase Scene: Tiger → Cow → Elephant")
    print("=" * 70)
    
    # Craft a concise prompt (under 77 tokens for CLIP)
    # Simplified to focus on clear, achievable action
    prompt = (
        "FPS-24, A tiger chasing a cow running forward through grassland, "
        "the cow chasing an elephant running forward, all moving left to right, "
        "golden hour lighting, cinematic tracking shot"
    )
    
    # Prepare request with optimized settings
    payload = {
        "prompt": prompt,
        "task_type": "t2v",
        "height": 544,
        "width": 960,
        "num_frames": 97,  # ~4 seconds at 24fps
        "num_inference_steps": 30,
        "guidance_scale": 6.0,
        "embedded_guidance_scale": 1.0,
        "seed": -1,  # Random seed for variety
        "fps": 24,
        "negative_prompt": (
            "Aerial view, static shot, blurry, low quality, deformation, "
            "bad anatomy, unnatural movements, cartoon, animated, "
            "poor composition, overexposed, underexposed"
        ),
        "cfg_for": False
    }
    
    print(f"\n📝 Prompt:")
    print(f"   {prompt[:100]}...")
    print(f"\n⚙️  Settings:")
    print(f"   Resolution: {payload['width']}x{payload['height']}")
    print(f"   Duration: ~4 seconds ({payload['num_frames']} frames @ {payload['fps']} fps)")
    print(f"   Quality: {payload['num_inference_steps']} inference steps")
    
    # Send generation request
    print("\n🚀 Submitting generation request...")
    
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
        
        # Poll for completion
        print("\n⏳ Generating video (this may take a few minutes)...")
        poll_interval = 25
        max_wait_time = 3000  # 10 minutes
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait_time:
                print(f"\n✗ Timeout: Job did not complete within {max_wait_time}s")
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
            
            # Show progress with animation
            dots = "." * (int(elapsed) % 4)
            print(f"\r  [{int(elapsed)}s] {current_status.upper()}{dots:<3} - {progress}", end="", flush=True)
            
            if current_status == 'completed':
                print(f"\n\n✅ Video generation completed!")
                print(f"⏱️  Total time: {elapsed:.2f}s")
                print(f"🎯 Generation time: {status.get('generation_time', 'N/A')}s")
                
                # Download video
                video_url = f"{API_BASE_URL}{status['video_path']}"
                output_file = OUTPUT_DIR / f"chase_scene_{job_id}.mp4"
                
                print(f"\n⬇️  Downloading video...")
                video_response = requests.get(video_url, timeout=60)
                video_response.raise_for_status()
                
                with open(output_file, 'wb') as f:
                    f.write(video_response.content)
                
                file_size_mb = output_file.stat().st_size / (1024*1024)
                print(f"✓ Video saved to: {output_file}")
                print(f"✓ File size: {file_size_mb:.2f} MB")
                
                print("\n" + "=" * 70)
                print("🎉 Chase scene generated successfully!")
                print("=" * 70)
                
                return True
                
            elif current_status == 'failed':
                print(f"\n\n✗ Generation failed: {status.get('message', 'Unknown error')}")
                return False
            
            # Wait before next poll
            time.sleep(poll_interval)
            
    except requests.exceptions.Timeout:
        print("\n✗ Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Request failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("\n🐯 SkyReels Chase Scene Generator\n")
    
    success = generate_chase_scene()
    
    sys.exit(0 if success else 1)
