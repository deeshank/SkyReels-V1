import os
import time
import random
import uuid
import logging
import threading
import queue
from typing import Optional, Literal, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from diffusers.utils import export_to_video, load_image
from PIL import Image
import base64
import io

from skyreelsinfer import TaskType
from skyreelsinfer.offload import OffloadConfig
from skyreelsinfer.skyreels_video_infer import SkyReelsVideoInfer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instances
predictors = {}

# Job tracking
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()
generation_queue = None  # Will be initialized in lifespan
worker_thread = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize predictors on startup and cleanup on shutdown."""
    global generation_queue, worker_thread
    
    gpu_num = int(os.getenv("GPU_NUM", "1"))
    use_quant = os.getenv("USE_QUANT", "true").lower() == "true"
    use_offload = os.getenv("USE_OFFLOAD", "true").lower() == "true"
    high_cpu_memory = os.getenv("HIGH_CPU_MEMORY", "true").lower() == "true"
    parameters_level = os.getenv("PARAMETERS_LEVEL", "true").lower() == "true"
    
    # Initialize T2V by default
    task_types = os.getenv("TASK_TYPES", "t2v").split(",")
    
    for task_type in task_types:
        task_type = task_type.strip()
        try:
            predictors[task_type] = initialize_predictor(
                task_type=task_type,
                gpu_num=gpu_num,
                use_quant=use_quant,
                use_offload=use_offload,
                high_cpu_memory=high_cpu_memory,
                parameters_level=parameters_level
            )
        except Exception as e:
            logger.error(f"Failed to initialize {task_type} predictor: {e}")
    
    # Initialize job queue and worker thread
    generation_queue = queue.Queue()
    worker_thread = threading.Thread(target=process_video_generation_worker, daemon=True)
    worker_thread.start()
    logger.info("Started video generation worker thread")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down...")
    if generation_queue:
        generation_queue.put(None)  # Signal worker to stop
    if worker_thread:
        worker_thread.join(timeout=5)

app = FastAPI(
    title="SkyReels Video Generation API",
    description="FastAPI server for SkyReels AI video generation (T2V and I2V)",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration
OUTPUT_DIR = Path("./api_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation (should start with 'FPS-24, ')")
    task_type: Literal["t2v", "i2v"] = Field("t2v", description="Task type: text-to-video or image-to-video")
    height: int = Field(544, ge=256, le=1024, description="Video height")
    width: int = Field(960, ge=256, le=1920, description="Video width")
    num_frames: int = Field(97, ge=1, le=289, description="Number of frames")
    num_inference_steps: int = Field(30, ge=1, le=100, description="Number of inference steps")
    guidance_scale: float = Field(6.0, ge=1.0, le=20.0, description="Guidance scale")
    embedded_guidance_scale: float = Field(1.0, ge=0.0, le=10.0, description="Embedded guidance scale")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    negative_prompt: str = Field(
        "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
        description="Negative prompt"
    )
    fps: int = Field(24, ge=1, le=60, description="Output video FPS")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image for I2V (required if task_type=i2v)")
    cfg_for: bool = Field(False, description="Enable sequential batch CFG")

class VideoGenerationResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    video_path: Optional[str] = None
    message: Optional[str] = None
    generation_time: Optional[float] = None
    progress: Optional[str] = None

def get_model_id(task_type: str) -> str:
    """Get the appropriate model ID based on task type."""
    return "Skywork/SkyReels-V1-Hunyuan-I2V" if task_type == "i2v" else "Skywork/SkyReels-V1-Hunyuan-T2V"

def initialize_predictor(task_type: str, gpu_num: int = 1, use_quant: bool = True, 
                        use_offload: bool = True, high_cpu_memory: bool = True,
                        parameters_level: bool = True, compiler_transformer: bool = False):
    """Initialize the video generation predictor."""
    logger.info(f"Initializing predictor for {task_type} with {gpu_num} GPU(s)")
    
    predictor = SkyReelsVideoInfer(
        task_type=TaskType.I2V if task_type == "i2v" else TaskType.T2V,
        model_id=get_model_id(task_type),
        quant_model=use_quant,
        world_size=gpu_num,
        is_offload=use_offload,
        offload_config=OffloadConfig(
            high_cpu_memory=high_cpu_memory,
            parameters_level=parameters_level,
            compiler_transformer=compiler_transformer,
        ),
        enable_cfg_parallel=True,
    )
    
    logger.info(f"Predictor initialized for {task_type}")
    return predictor

def process_video_generation_worker():
    """Background worker thread that processes video generation jobs from queue."""
    while True:
        try:
            # Get job from queue (blocking)
            job_data = generation_queue.get()
            
            if job_data is None:  # Shutdown signal
                break
            
            job_id = job_data["job_id"]
            task_type = job_data["task_type"]
            kwargs = job_data["kwargs"]
            fps = job_data["fps"]
            
            try:
                with jobs_lock:
                    jobs[job_id]["status"] = JobStatus.PROCESSING
                    jobs[job_id]["progress"] = "Generating video..."
                
                logger.info(f"Starting video generation for job {job_id}")
                start_time = time.time()
                
                predictor = predictors[task_type]
                output = predictor.inference(kwargs)
                
                # Save video
                output_path = OUTPUT_DIR / f"{job_id}.mp4"
                export_to_video(output, str(output_path), fps=fps)
                
                generation_time = time.time() - start_time
                
                with jobs_lock:
                    jobs[job_id]["status"] = JobStatus.COMPLETED
                    jobs[job_id]["video_path"] = f"/video/{job_id}"
                    jobs[job_id]["generation_time"] = generation_time
                    jobs[job_id]["progress"] = "Completed"
                
                logger.info(f"Video generated successfully for job {job_id} in {generation_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error generating video for job {job_id}: {str(e)}")
                with jobs_lock:
                    jobs[job_id]["status"] = JobStatus.FAILED
                    jobs[job_id]["message"] = str(e)
                    jobs[job_id]["progress"] = "Failed"
            
            finally:
                generation_queue.task_done()
                
        except Exception as e:
            logger.error(f"Worker thread error: {e}")
            time.sleep(1)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SkyReels Video Generation API",
        "version": "1.0.0",
        "available_models": list(predictors.keys()),
        "endpoints": {
            "/generate": "POST - Generate video",
            "/health": "GET - Health check",
            "/video/{job_id}": "GET - Download generated video"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "available_models": list(predictors.keys()),
        "gpu_count": int(os.getenv("GPU_NUM", "1"))
    }

@app.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(request: VideoGenerationRequest):
    """Generate video from text or image+text (async - returns immediately)."""
    job_id = str(uuid.uuid4())
    
    try:
        # Validate task type
        if request.task_type not in predictors:
            raise HTTPException(
                status_code=400,
                detail=f"Task type '{request.task_type}' not initialized. Available: {list(predictors.keys())}"
            )
        
        # Validate I2V requirements
        if request.task_type == "i2v" and not request.image_base64:
            raise HTTPException(
                status_code=400,
                detail="image_base64 is required for I2V task"
            )
        
        # Handle seed
        seed = request.seed
        if seed == -1:
            random.seed(time.time())
            seed = int(random.randrange(4294967294))
        
        # Prepare kwargs
        kwargs = {
            "prompt": request.prompt,
            "height": request.height,
            "width": request.width,
            "num_frames": request.num_frames,
            "num_inference_steps": request.num_inference_steps,
            "seed": seed,
            "guidance_scale": request.guidance_scale,
            "embedded_guidance_scale": request.embedded_guidance_scale,
            "negative_prompt": request.negative_prompt,
            "cfg_for": request.cfg_for,
        }
        
        # Handle image for I2V
        if request.task_type == "i2v":
            try:
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(io.BytesIO(image_data))
                kwargs["image"] = image
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Initialize job
        with jobs_lock:
            jobs[job_id] = {
                "status": JobStatus.PENDING,
                "task_type": request.task_type,
                "prompt": request.prompt,
                "created_at": time.time(),
                "progress": "Queued"
            }
        
        # Add job to queue (worker thread will process it)
        generation_queue.put({
            "job_id": job_id,
            "task_type": request.task_type,
            "kwargs": kwargs,
            "fps": request.fps
        })
        
        logger.info(f"Job {job_id} queued for processing")
        
        return VideoGenerationResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Job queued for processing. Use /status/{job_id} to check progress."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queueing job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a video generation job."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id].copy()
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        video_path=job.get("video_path"),
        message=job.get("message"),
        generation_time=job.get("generation_time"),
        progress=job.get("progress")
    )

@app.get("/video/{job_id}")
async def get_video(job_id: str):
    """Download generated video by job ID."""
    video_path = OUTPUT_DIR / f"{job_id}.mp4"
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
    )

@app.post("/generate-multipart")
async def generate_video_multipart(
    prompt: str = Form(...),
    task_type: Literal["t2v", "i2v"] = Form("t2v"),
    height: int = Form(544),
    width: int = Form(960),
    num_frames: int = Form(97),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(6.0),
    embedded_guidance_scale: float = Form(1.0),
    seed: int = Form(-1),
    fps: int = Form(24),
    image: Optional[UploadFile] = File(None),
    negative_prompt: str = Form("Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"),
    cfg_for: bool = Form(False)
):
    """Generate video using multipart form data (async - returns immediately)."""
    job_id = str(uuid.uuid4())
    
    try:
        # Validate task type
        if task_type not in predictors:
            raise HTTPException(
                status_code=400,
                detail=f"Task type '{task_type}' not initialized. Available: {list(predictors.keys())}"
            )
        
        # Validate I2V requirements
        if task_type == "i2v" and not image:
            raise HTTPException(
                status_code=400,
                detail="image file is required for I2V task"
            )
        
        # Handle seed
        if seed == -1:
            random.seed(time.time())
            seed = int(random.randrange(4294967294))
        
        # Prepare kwargs
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "embedded_guidance_scale": embedded_guidance_scale,
            "negative_prompt": negative_prompt,
            "cfg_for": cfg_for,
        }
        
        # Handle image for I2V
        if task_type == "i2v" and image:
            try:
                image_data = await image.read()
                pil_image = Image.open(io.BytesIO(image_data))
                kwargs["image"] = pil_image
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Initialize job
        with jobs_lock:
            jobs[job_id] = {
                "status": JobStatus.PENDING,
                "task_type": task_type,
                "prompt": prompt,
                "created_at": time.time(),
                "progress": "Queued"
            }
        
        # Add job to queue (worker thread will process it)
        generation_queue.put({
            "job_id": job_id,
            "task_type": task_type,
            "kwargs": kwargs,
            "fps": fps
        })
        
        logger.info(f"Job {job_id} queued for processing")
        
        return {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Job queued for processing. Use /status/{job_id} to check progress."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queueing job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
