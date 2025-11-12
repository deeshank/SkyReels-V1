#!/usr/bin/env python3
"""
Generate cinematic movie-style scenes with SkyReels
"""
import requests
import time
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "https://dr9b1f4vx6l8rp-8000.proxy.runpod.net"
OUTPUT_DIR = Path("./generated_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

# Comprehensive cinematic prompts (LLaMA encoder supports up to 256 tokens)
MOVIE_SCENES = {
    "1": {
        "name": "Dramatic Close-Up",
        "prompt": (
            "FPS-24, Extreme close-up shot of a young woman's face with tears streaming down her cheeks, "
            "capturing raw emotion and vulnerability. Her eyes glisten with moisture, reflecting soft natural light "
            "from a nearby window. Shallow depth of field with bokeh background. Cinematic color grading with "
            "desaturated tones and subtle film grain. Professional Hollywood cinematography with perfect focus on "
            "her expressive eyes. Emotional drama style reminiscent of award-winning films."
        ),
        "description": "Intimate emotional drama with award-winning cinematography"
    },
    "2": {
        "name": "Action Hero Walk",
        "prompt": (
            "FPS-24, A confident man in a tailored black suit and sunglasses walking directly toward the camera "
            "in dramatic slow motion. Behind him, a massive explosion erupts with orange flames and black smoke "
            "billowing into the sky. He doesn't look back at the explosion, maintaining a stoic expression. "
            "Low angle camera shot emphasizing his powerful presence. Cinematic action movie style with high contrast "
            "lighting, lens flares from the fire, and debris floating in the air. Michael Bay style explosive action sequence."
        ),
        "description": "Epic action hero moment with explosive background"
    },
    "3": {
        "name": "Film Noir Detective",
        "prompt": (
            "FPS-24, A hardboiled detective wearing a classic beige trench coat and dark fedora hat walks through "
            "a rain-soaked city street at night. Neon signs in red and blue reflect off the wet pavement, creating "
            "colorful puddle reflections. Heavy rain falls, illuminated by street lamps creating dramatic shafts of light. "
            "Moody film noir atmosphere with high contrast black and white aesthetic, deep shadows, and mysterious ambiance. "
            "1940s detective movie style with cigarette smoke wisps and foggy air. Classic noir cinematography."
        ),
        "description": "Atmospheric noir detective scene with rain and neon"
    },
    "4": {
        "name": "Romantic Moment",
        "prompt": (
            "FPS-24, A young couple in love holding hands while walking barefoot along a pristine beach at golden hour sunset. "
            "The woman wears a flowing white dress that moves gently in the ocean breeze, the man in casual linen shirt. "
            "Warm golden sunlight bathes the scene with a romantic glow, creating long shadows on the sand. "
            "Gentle waves lap at their feet. Soft focus romantic cinematography with warm color palette, lens flares from the sun, "
            "and dreamy bokeh. Professional wedding film style with emotional depth and beautiful composition."
        ),
        "description": "Romantic beach sunset with cinematic warmth"
    },
    "5": {
        "name": "Sci-Fi Corridor",
        "prompt": (
            "FPS-24, An astronaut in a detailed white spacesuit with glowing helmet visor walks through a sleek futuristic "
            "spaceship corridor. The hallway features smooth metallic walls with blue LED strip lighting running along the edges. "
            "Holographic displays and control panels glow with cyan and white light. Atmospheric mist near the floor adds depth. "
            "Science fiction movie cinematography with cool color temperature, lens flares from bright lights, and high-tech details. "
            "Interstellar or The Martian style realistic space aesthetic with professional VFX quality."
        ),
        "description": "High-tech sci-fi spaceship with realistic details"
    },
    "6": {
        "name": "Horror Suspense",
        "prompt": (
            "FPS-24, A terrified young woman slowly walks down a dark, narrow hallway holding a flickering flashlight "
            "that barely illuminates the oppressive darkness. The beam of light shakes with her trembling hands. "
            "Ominous shadows move and stretch along the walls, suggesting unseen presence. Peeling wallpaper and "
            "decrepit surroundings add to the dread. Suspenseful horror movie atmosphere with desaturated colors, "
            "heavy grain, and claustrophobic framing. Tense cinematography reminiscent of The Conjuring or Hereditary. "
            "Building tension with every step forward into the unknown darkness."
        ),
        "description": "Terrifying horror atmosphere with building dread"
    },
    "7": {
        "name": "Western Showdown",
        "prompt": (
            "FPS-24, A rugged cowboy with weathered face and stern expression stands in the middle of a dusty desert town "
            "main street at high noon. He wears a worn leather vest, dark hat, and his hand hovers near his holster, "
            "ready to draw. Tumbleweeds roll past in the hot wind. The sun beats down creating harsh shadows and lens flares. "
            "Classic western movie cinematography with warm sepia tones, dust particles visible in the air, and wide-angle "
            "composition. Sergio Leone spaghetti western style with dramatic tension and iconic gunslinger posture. "
            "Cinematic showdown moment before the quick draw."
        ),
        "description": "Iconic western standoff with classic cinematography"
    },
    "8": {
        "name": "Dance Performance",
        "prompt": (
            "FPS-24, An elegant ballet dancer in a pristine white tutu performs a graceful pirouette on a dark stage. "
            "A single dramatic spotlight illuminates her from above, creating a perfect circle of light while the background "
            "fades to black. Her movements are fluid and precise, arms extended in perfect form, standing en pointe. "
            "Slow motion captures the flowing fabric and elegant lines of her body. Cinematic performance cinematography "
            "with high contrast lighting, shallow depth of field, and artistic composition. Professional dance film style "
            "reminiscent of Black Swan, emphasizing beauty, grace, and technical perfection."
        ),
        "description": "Elegant ballet performance with dramatic stage lighting"
    }
}

def generate_scene(scene_id, scene_data):
    """Generate a single movie scene"""
    print("\n" + "=" * 70)
    print(f"🎬 Scene {scene_id}: {scene_data['name']}")
    print("=" * 70)
    print(f"📝 {scene_data['description']}")
    
    payload = {
        "prompt": scene_data['prompt'],
        "task_type": "t2v",
        "height": 544,
        "width": 960,
        "num_frames": 97,
        "num_inference_steps": 30,
        "guidance_scale": 6.0,
        "embedded_guidance_scale": 1.0,
        "seed": -1,
        "fps": 24,
        "negative_prompt": (
            "Aerial view, low quality, blurry, deformation, bad composition, "
            "overexposed, cartoon, animated, unrealistic"
        ),
        "cfg_for": False
    }
    
    print(f"\n💬 Prompt: {scene_data['prompt']}")
    
    try:
        # Submit job
        print("\n🚀 Submitting generation request...")
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        job_id = result['job_id']
        print(f"✓ Job ID: {job_id}")
        
        # Poll for completion
        print("⏳ Generating (this may take a few minutes)...")
        poll_interval = 25
        max_wait_time = 6000
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait_time:
                print(f"\n✗ Timeout after {max_wait_time}s")
                return False
            
            status_response = requests.get(
                f"{API_BASE_URL}/status/{job_id}",
                timeout=10
            )
            status_response.raise_for_status()
            status = status_response.json()
            
            current_status = status['status']
            progress = status.get('progress', 'Unknown')
            
            dots = "." * (int(elapsed) % 4)
            print(f"\r  [{int(elapsed)}s] {current_status.upper()}{dots:<3}", end="", flush=True)
            
            if current_status == 'completed':
                print(f"\n✅ Completed in {elapsed:.2f}s")
                
                # Download video
                video_url = f"{API_BASE_URL}{status['video_path']}"
                scene_name = scene_data['name'].lower().replace(' ', '_')
                output_file = OUTPUT_DIR / f"{scene_name}_{job_id}.mp4"
                
                print(f"⬇️  Downloading...")
                video_response = requests.get(video_url, timeout=60)
                video_response.raise_for_status()
                
                with open(output_file, 'wb') as f:
                    f.write(video_response.content)
                
                file_size_mb = output_file.stat().st_size / (1024*1024)
                print(f"✓ Saved: {output_file.name} ({file_size_mb:.2f} MB)")
                
                return True
                
            elif current_status == 'failed':
                print(f"\n✗ Failed: {status.get('message', 'Unknown error')}")
                return False
            
            time.sleep(poll_interval)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def show_menu():
    """Display scene selection menu"""
    print("\n" + "=" * 70)
    print("🎥 SKYREELS CINEMATIC SCENE GENERATOR")
    print("=" * 70)
    print("\nAvailable Scenes:\n")
    
    for scene_id, scene_data in sorted(MOVIE_SCENES.items()):
        print(f"  [{scene_id}] {scene_data['name']:<25} - {scene_data['description']}")
    
    print(f"\n  [A] Generate ALL scenes")
    print(f"  [Q] Quit")
    print("\n" + "=" * 70)

def main():
    """Main function"""
    while True:
        show_menu()
        choice = input("\nSelect scene (1-8, A for all, Q to quit): ").strip().upper()
        
        if choice == 'Q':
            print("\n👋 Goodbye!")
            sys.exit(0)
        
        elif choice == 'A':
            print("\n🎬 Generating ALL scenes...")
            success_count = 0
            
            for scene_id, scene_data in sorted(MOVIE_SCENES.items()):
                if generate_scene(scene_id, scene_data):
                    success_count += 1
                print()  # Spacing between scenes
            
            print("\n" + "=" * 70)
            print(f"🎉 Generated {success_count}/{len(MOVIE_SCENES)} scenes successfully!")
            print("=" * 70)
            
            input("\nPress Enter to continue...")
        
        elif choice in MOVIE_SCENES:
            generate_scene(choice, MOVIE_SCENES[choice])
            input("\nPress Enter to continue...")
        
        else:
            print("❌ Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
