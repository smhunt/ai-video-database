"""
Video Chat Server - Main FastAPI application.

A local web server that allows uploading videos, automatic AI analysis,
and conversational interaction with video content.
"""
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import uuid
from loguru import logger
import json

from src.models.database import get_db, VideoDatabase
from src.services.video_processor import VideoProcessor
from src.services.video_embeddings import VideoEmbeddingService
from src.services.audio_transcription import AudioTranscriptionService
import anthropic

# Import cost calculator directly
import sys
import importlib.util
cost_calc_spec = importlib.util.spec_from_file_location(
    "cost_calculator",
    "src/utils/cost_calculator.py"
)
cost_calc_module = importlib.util.module_from_spec(cost_calc_spec)
cost_calc_spec.loader.exec_module(cost_calc_module)
calculate_claude_cost = cost_calc_module.calculate_claude_cost
calculate_openai_embedding_cost = cost_calc_module.calculate_openai_embedding_cost

# Import VideoAnalyzer without loading other tools that need smolagents
import importlib.util
spec = importlib.util.spec_from_file_location(
    "video_analyzer",
    "src/tools/video_analyzer.py"
)
video_analyzer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_analyzer_module)
VideoAnalyzer = video_analyzer_module.VideoAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Video Chat Server",
    description="Upload and chat with your videos using AI",
    version="1.0.0",
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
db = get_db()
video_processor = VideoProcessor()
video_analyzer = VideoAnalyzer()
embedding_service = VideoEmbeddingService()
transcription_service = AudioTranscriptionService()

# Create static directory if it doesn't exist
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models for API
class VideoUploadResponse(BaseModel):
    video_id: int
    filename: str
    status: str
    message: str


class ChatRequest(BaseModel):
    video_id: int
    query: str
    context_window: Optional[int] = 5  # Number of previous messages to consider


class ChatResponse(BaseModel):
    answer: str
    relevant_frames: List[Dict[str, Any]]
    timestamp: Optional[float] = None


class VideoInfo(BaseModel):
    id: int
    filename: str
    duration_seconds: float
    status: str
    upload_date: str
    thumbnail_url: Optional[str] = None


class FrameInfo(BaseModel):
    frame_id: int
    timestamp_seconds: float
    description: str
    thumbnail_url: str
    excitement_score: int


# Background task for processing video
async def process_video_task(video_id: int, video_path: str):
    """Background task to process uploaded video."""
    try:
        logger.info(f"Starting background processing for video {video_id}")

        # Get video metadata
        video_info = video_processor.get_video_info(video_path)
        db.update_video_info(
            video_id,
            duration_seconds=video_info["duration_seconds"],
            fps=video_info["fps"],
            width=video_info["width"],
            height=video_info["height"],
        )

        # Extract frames (every 2 seconds, max 500 frames)
        frames = video_processor.extract_frames(
            video_path, video_id, interval_seconds=2.0, max_frames=500
        )
        logger.info(f"Extracted {len(frames)} frames")

        # Also extract keyframes at scene changes
        keyframes = video_processor.extract_keyframes_at_scenes(
            video_path, video_id, max_keyframes=50
        )
        logger.info(f"Extracted {len(keyframes)} keyframes")

        # Save frames to database
        all_frames = frames + keyframes
        frame_paths = []
        frame_ids = []

        for frame_data in all_frames:
            frame_id = db.add_frame(
                video_id=video_id,
                timestamp_seconds=frame_data["timestamp_seconds"],
                frame_number=frame_data["frame_number"],
                frame_path=frame_data["frame_path"],
                is_keyframe=frame_data.get("is_keyframe", False),
            )
            frame_paths.append(frame_data["frame_path"])
            frame_ids.append(frame_id)

        logger.info(f"Saved {len(all_frames)} frames to database")

        # Extract and transcribe audio (if video has audio)
        audio_path = None
        transcription_cost = 0.0
        if video_processor.has_audio(video_path):
            try:
                db.update_video_status(video_id, "transcribing")
                logger.info("Extracting audio from video...")

                # Extract audio to MP3
                audio_path = video_processor.extract_audio(video_path, format="mp3")

                # Transcribe audio with timestamps
                logger.info("Transcribing audio with Whisper...")
                segments = transcription_service.transcribe_with_timestamps(audio_path)

                # Save transcript segments to database
                for segment in segments:
                    db.add_transcript_segment(
                        video_id=video_id,
                        start_time=segment["start_time"],
                        end_time=segment["end_time"],
                        text=segment["text"],
                        confidence=segment["confidence"],
                        model_name="whisper-1",
                    )

                logger.info(f"Saved {len(segments)} transcript segments to database")

                # Calculate and track transcription cost
                transcription_cost = transcription_service.estimate_cost(
                    video_info["duration_seconds"]
                )
                db.add_api_cost(
                    video_id=video_id,
                    operation_type="transcription",
                    api_provider="openai",
                    model_name="whisper-1",
                    estimated_cost_usd=transcription_cost,
                    details=f"Transcribed {len(segments)} segments, {video_info['duration_seconds']:.1f}s audio",
                )
                logger.info(f"Transcription cost: ${transcription_cost:.4f}")

                # Clean up extracted audio file
                if audio_path and Path(audio_path).exists():
                    Path(audio_path).unlink()
                    logger.info("Cleaned up temporary audio file")

            except Exception as e:
                logger.warning(f"Audio transcription failed: {e}")
                # Continue processing even if transcription fails

        # Analyze frames with Claude Vision
        db.update_video_status(video_id, "analyzing")

        # Create fresh analyzer for this video to track costs
        analyzer = VideoAnalyzer()
        analysis_results = analyzer.analyze_frames(
            frame_paths[:100]  # Analyze first 100 frames to manage costs
        )

        # Get usage stats and calculate cost
        usage = analyzer.get_usage_stats()
        analysis_cost = calculate_claude_cost(
            model=analyzer.model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            num_images=usage["num_images"],
        )

        # Track analysis cost
        db.add_api_cost(
            video_id=video_id,
            operation_type="frame_analysis",
            api_provider="anthropic",
            model_name=analyzer.model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            num_images=usage["num_images"],
            estimated_cost_usd=analysis_cost,
            details=f"Analyzed {len(analysis_results)} frames",
        )

        logger.info(f"Frame analysis cost: ${analysis_cost:.4f}")

        # Update database with analysis (including excitement scores)
        for analysis in analysis_results:
            if analysis.frame_index < len(frame_ids):
                frame_id = frame_ids[analysis.frame_index]
                # Add excitement_score column if it doesn't exist
                try:
                    db.connection.execute("ALTER TABLE frames ADD COLUMN excitement_score INTEGER DEFAULT 5")
                except:
                    pass  # Column already exists

                # Update frame record
                db.connection.execute(
                    """
                    UPDATE frames
                    SET description = ?, objects = ?, actions = ?, scene_type = ?, excitement_score = ?
                    WHERE id = ?
                """,
                    (
                        analysis.description,
                        json.dumps(analysis.objects),
                        json.dumps(analysis.actions),
                        analysis.scene_type,
                        analysis.excitement_score,
                        frame_id,
                    ),
                )
        db.connection.commit()

        logger.info(f"Updated {len(analysis_results)} frames with AI analysis")

        # Index in Qdrant for semantic search
        db.update_video_status(video_id, "indexing")
        frame_data_for_indexing = []
        for i, analysis in enumerate(analysis_results):
            if analysis.frame_index < len(frame_ids):
                frame_data_for_indexing.append(
                    {
                        "frame_id": frame_ids[analysis.frame_index],
                        "description": analysis.description,
                        "timestamp_seconds": all_frames[analysis.frame_index][
                            "timestamp_seconds"
                        ],
                        "objects": analysis.objects,
                        "actions": analysis.actions,
                        "scene_type": analysis.scene_type,
                        "excitement_score": analysis.excitement_score,
                    }
                )

        embedding_service.index_frames(video_id, frame_data_for_indexing)

        # Track embedding cost
        embedding_cost = calculate_openai_embedding_cost(
            model="text-embedding-3-large",
            num_embeddings=len(frame_data_for_indexing),
            avg_tokens_per_text=40,
        )

        db.add_api_cost(
            video_id=video_id,
            operation_type="embeddings",
            api_provider="openai",
            model_name="text-embedding-3-large",
            num_embeddings=len(frame_data_for_indexing),
            estimated_cost_usd=embedding_cost,
            details=f"Generated embeddings for {len(frame_data_for_indexing)} frame descriptions",
        )

        logger.info(f"Embedding cost: ${embedding_cost:.4f}")

        # Generate thumbnail strip
        try:
            video_obj = db.get_video(video_id)
            thumb_path = f"data/thumbnails/video_{video_id}_strip.jpg"
            Path("data/thumbnails").mkdir(parents=True, exist_ok=True)
            video_processor.generate_thumbnail_strip(
                video_path, thumb_path, num_thumbs=10
            )
            logger.info(f"Generated thumbnail strip")
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail strip: {e}")

        # Mark as complete
        db.update_video_status(video_id, "ready")
        logger.info(f"Video {video_id} processing complete")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        db.update_video_status(video_id, "error")


@app.get("/")
async def root():
    """Serve the main web UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return JSONResponse(
            {
                "message": "Video Chat Server is running",
                "docs": "/docs",
                "note": "Create static/index.html for web UI",
            }
        )


@app.post("/api/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a video file for processing.

    The video will be processed in the background:
    1. Extract frames at regular intervals
    2. Analyze frames with Claude Vision
    3. Index for semantic search
    4. Generate timeline thumbnails
    """
    # Validate file size (2GB limit)
    MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2GB in bytes

    # Generate unique filename
    original_filename = file.filename
    file_ext = Path(original_filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    save_path = video_processor.output_dir / unique_filename

    try:
        # Save uploaded file
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = save_path.stat().st_size
        if file_size > MAX_SIZE:
            save_path.unlink()  # Delete file
            raise HTTPException(
                status_code=413, detail=f"File too large (max 2GB)"
            )

        # Validate video file
        is_valid, error_msg = video_processor.validate_video(str(save_path))
        if not is_valid:
            save_path.unlink()
            raise HTTPException(status_code=400, detail=error_msg)

        # Add to database
        video_id = db.add_video(
            filename=unique_filename,
            original_filename=original_filename,
            path=str(save_path),
            size_bytes=file_size,
        )

        # Start background processing
        background_tasks.add_task(process_video_task, video_id, str(save_path))

        logger.info(
            f"Video uploaded: {original_filename} -> ID {video_id}, processing started"
        )

        return VideoUploadResponse(
            video_id=video_id,
            filename=original_filename,
            status="processing",
            message="Video uploaded successfully. Processing in background.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos", response_model=List[VideoInfo])
async def list_videos():
    """Get list of all uploaded videos."""
    videos = db.get_all_videos()
    return [
        VideoInfo(
            id=v["id"],
            filename=v["original_filename"],
            duration_seconds=v["duration_seconds"] or 0,
            status=v["status"],
            upload_date=v["upload_date"],
            thumbnail_url=f"/api/videos/{v['id']}/thumbnail"
            if v["status"] == "ready"
            else None,
        )
        for v in videos
    ]


@app.get("/api/videos/{video_id}")
async def get_video_info(video_id: int):
    """Get detailed information about a video."""
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    keyframes = db.get_keyframes_for_video(video_id)

    return {
        "id": video["id"],
        "filename": video["original_filename"],
        "duration_seconds": video["duration_seconds"],
        "fps": video["fps"],
        "width": video["width"],
        "height": video["height"],
        "status": video["status"],
        "upload_date": video["upload_date"],
        "num_frames": len(db.get_frames_for_video(video_id)),
        "num_keyframes": len(keyframes),
        "video_url": f"/api/videos/{video_id}/stream",
        "thumbnail_url": f"/api/videos/{video_id}/thumbnail",
    }


@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: int):
    """Stream video file."""
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(video["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=video["original_filename"],
    )


@app.get("/api/videos/{video_id}/thumbnail")
async def get_thumbnail(video_id: int):
    """Get thumbnail strip for video."""
    thumb_path = Path(f"data/thumbnails/video_{video_id}_strip.jpg")
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(thumb_path, media_type="image/jpeg")


@app.get("/api/videos/{video_id}/frames", response_model=List[FrameInfo])
async def get_video_frames(
    video_id: int,
    keyframes_only: bool = Query(False, description="Return only keyframes"),
):
    """Get frames for a video with AI analysis."""
    if keyframes_only:
        frames = db.get_keyframes_for_video(video_id)
    else:
        frames = db.get_frames_for_video(video_id)

    return [
        FrameInfo(
            frame_id=f["id"],
            timestamp_seconds=f["timestamp_seconds"],
            description=f["description"] or "Not yet analyzed",
            thumbnail_url=f"/api/frames/{f['id']}/image",
            excitement_score=0,  # TODO: Store this in DB
        )
        for f in frames
    ]


@app.get("/api/frames/{frame_id}/image")
async def get_frame_image(frame_id: int):
    """Get frame image."""
    cursor = db.connection.cursor()
    cursor.execute("SELECT frame_path FROM frames WHERE id = ?", (frame_id,))
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Frame not found")

    frame_path = Path(row[0])
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame image not found")

    return FileResponse(frame_path, media_type="image/jpeg")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_video(request: ChatRequest):
    """
    Chat with a video using natural language.

    Supports:
    - Timestamp queries: "what happens at 1:23?"
    - Content search: "find explosions"
    - General questions: "summarize this video"
    """
    video = db.get_video(request.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video["status"] != "ready":
        raise HTTPException(
            status_code=400, detail=f"Video not ready (status: {video['status']})"
        )

    query = request.query.lower()

    try:
        # Handle timestamp queries
        if "at" in query and (":" in query or "second" in query or "minute" in query):
            timestamp = extract_timestamp_from_query(query)
            if timestamp is not None:
                return await handle_timestamp_query(request.video_id, timestamp, query)

        # Handle search queries
        if any(word in query for word in ["find", "show", "search", "when"]):
            return await handle_search_query(request.video_id, query)

        # Handle general chat
        return await handle_general_query(request.video_id, query)

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_timestamp_query(
    video_id: int, timestamp: float, query: str
) -> ChatResponse:
    """Handle query about specific timestamp."""
    # Find frame closest to timestamp
    frame = db.get_frame_at_timestamp(video_id, timestamp, tolerance=2.0)

    if not frame:
        return ChatResponse(
            answer=f"No frame found near {timestamp:.1f} seconds.",
            relevant_frames=[],
            timestamp=timestamp,
        )

    # Analyze specific frame if no description
    if not frame["description"]:
        result = video_analyzer.analyze_single_frame(frame["frame_path"], query)
        answer = result["answer"]
    else:
        # Use Claude to answer based on existing description
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": f"Based on this frame description: '{frame['description']}'\n\nAnswer this question: {query}",
                }
            ],
        )
        answer = response.content[0].text

    relevant_frames = [
        {
            "frame_id": frame["id"],
            "timestamp": frame["timestamp_seconds"],
            "description": frame["description"] or answer,
            "image_url": f"/api/frames/{frame['id']}/image",
        }
    ]

    return ChatResponse(
        answer=answer, relevant_frames=relevant_frames, timestamp=timestamp
    )


async def handle_search_query(video_id: int, query: str) -> ChatResponse:
    """Handle semantic search query."""
    query_lower = query.lower()

    # Check if asking for exciting moments/highlights
    if any(word in query_lower for word in ["exciting", "highlight", "best", "interesting", "action"]):
        # Get exciting frames
        results = embedding_service.get_exciting_frames(
            video_id=video_id,
            threshold=7,  # Excitement score >= 7
            limit=10,
        )

        if not results:
            return ChatResponse(
                answer="I couldn't find any particularly exciting moments in this video (threshold: 7/10).",
                relevant_frames=[],
            )

        # Format response
        relevant_frames = [
            {
                "frame_id": r["frame_id"],
                "timestamp": r["timestamp_seconds"],
                "description": r["description"],
                "image_url": f"/api/frames/{r['frame_id']}/image",
                "excitement_score": r["excitement_score"],
            }
            for r in results
        ]

        timestamps = [f"{r['timestamp_seconds']:.1f}s (score: {r['excitement_score']}/10)" for r in results[:5]]
        answer = f"Found {len(results)} exciting moments:\n\n" + "\n".join(timestamps)

        return ChatResponse(answer=answer, relevant_frames=relevant_frames)

    # Regular semantic search
    # Extract search terms (remove "find", "show", etc.)
    search_terms = query
    for word in ["find", "show", "search", "when", "where"]:
        search_terms = search_terms.replace(word, "")
    search_terms = search_terms.strip()

    # Search in Qdrant
    results = embedding_service.search_frames(
        query=search_terms, video_id=video_id, limit=5
    )

    if not results:
        return ChatResponse(
            answer=f"I couldn't find any frames matching '{search_terms}' in this video.",
            relevant_frames=[],
        )

    # Format response
    relevant_frames = [
        {
            "frame_id": r["frame_id"],
            "timestamp": r["timestamp_seconds"],
            "description": r["description"],
            "image_url": f"/api/frames/{r['frame_id']}/image",
            "similarity": r["similarity_score"],
        }
        for r in results
    ]

    timestamps = [f"{r['timestamp_seconds']:.1f}s" for r in results[:3]]
    answer = f"Found {len(results)} moments matching '{search_terms}' at: {', '.join(timestamps)}\n\nTop match: {results[0]['description']}"

    return ChatResponse(answer=answer, relevant_frames=relevant_frames)


async def handle_general_query(video_id: int, query: str) -> ChatResponse:
    """Handle general conversation about video."""
    # Get video frames summary
    frames_data = embedding_service.get_frames_for_video(video_id)[:20]

    if not frames_data:
        return ChatResponse(
            answer="This video hasn't been analyzed yet.", relevant_frames=[]
        )

    # Build context from frame descriptions
    context = "Video content summary:\n"
    for i, frame in enumerate(frames_data[:10]):
        context += f"- {frame['timestamp_seconds']:.1f}s: {frame['description']}\n"

    # Ask Claude
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"{context}\n\nBased on this video content, {query}",
            }
        ],
    )

    answer = response.content[0].text

    # Include relevant frames
    relevant_frames = [
        {
            "frame_id": f["frame_id"],
            "timestamp": f["timestamp_seconds"],
            "description": f["description"],
            "image_url": f"/api/frames/{f['frame_id']}/image",
        }
        for f in frames_data[:3]
    ]

    return ChatResponse(answer=answer, relevant_frames=relevant_frames)


def extract_timestamp_from_query(query: str) -> Optional[float]:
    """Extract timestamp from query like 'at 1:23' or 'at 45 seconds'."""
    import re

    # Match patterns like "1:23", "1:23:45", "45s", "45 seconds"
    patterns = [
        r"(\d+):(\d+):(\d+)",  # hours:minutes:seconds
        r"(\d+):(\d+)",  # minutes:seconds
        r"(\d+)\s*s",  # seconds with 's'
        r"(\d+)\s*sec",  # seconds with 'sec'
    ]

    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            groups = match.groups()
            if len(groups) == 3:  # h:m:s
                return int(groups[0]) * 3600 + int(groups[1]) * 60 + int(groups[2])
            elif len(groups) == 2:  # m:s
                return int(groups[0]) * 60 + int(groups[1])
            elif len(groups) == 1:  # seconds
                return int(groups[0])

    return None


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    videos = db.get_all_videos()
    total_videos = len(videos)
    ready_videos = len([v for v in videos if v["status"] == "ready"])

    qdrant_stats = embedding_service.get_collection_stats()
    cost_stats = db.get_total_costs()

    return {
        "total_videos": total_videos,
        "ready_videos": ready_videos,
        "total_frames_indexed": qdrant_stats["total_frames"],
        "embedding_model": "mxbai-embed-large-v1",
        "analysis_model": "claude-3-haiku-20240307",
        "costs": cost_stats,
    }


@app.get("/api/costs")
async def get_costs():
    """Get detailed cost breakdown."""
    total_costs = db.get_total_costs()

    # Get all cost records for detailed log
    cursor = db.connection.cursor()
    cursor.execute("""
        SELECT
            ac.*,
            v.original_filename
        FROM api_costs ac
        LEFT JOIN videos v ON ac.video_id = v.id
        ORDER BY ac.timestamp DESC
        LIMIT 100
    """)
    recent_costs = [dict(row) for row in cursor.fetchall()]

    return {
        "summary": total_costs,
        "recent_operations": recent_costs,
    }


@app.get("/api/videos/{video_id}/costs")
async def get_video_costs(video_id: int):
    """Get costs for a specific video."""
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    costs = db.get_costs_for_video(video_id)
    total = sum(c["estimated_cost_usd"] for c in costs)

    return {
        "video_id": video_id,
        "filename": video["original_filename"],
        "total_cost_usd": round(total, 4),
        "operations": costs,
    }


@app.get("/api/videos/{video_id}/transcript")
async def get_video_transcript(video_id: int):
    """
    Get full transcript for a video.

    Returns all transcript segments with timestamps.
    """
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    segments = db.get_transcripts_for_video(video_id)

    if not segments:
        raise HTTPException(
            status_code=404,
            detail="No transcript available for this video"
        )

    return {
        "video_id": video_id,
        "filename": video["original_filename"],
        "segment_count": len(segments),
        "segments": segments,
        "full_text": " ".join(seg["text"] for seg in segments),
    }


@app.get("/api/videos/{video_id}/transcript/search")
async def search_transcript(video_id: int, q: str = Query(..., min_length=1)):
    """
    Search for text within a video's transcript.

    Args:
        video_id: Video ID
        q: Search query

    Returns matching transcript segments.
    """
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    results = db.search_transcript(video_id, q)

    return {
        "video_id": video_id,
        "query": q,
        "result_count": len(results),
        "results": results,
    }


@app.get("/api/videos/{video_id}/transcript/at/{timestamp}")
async def get_transcript_at_timestamp(video_id: int, timestamp: float):
    """
    Get transcript segment at a specific timestamp.

    Args:
        video_id: Video ID
        timestamp: Time in seconds

    Returns transcript segment containing that timestamp.
    """
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    segment = db.get_transcript_at_timestamp(video_id, timestamp)

    if not segment:
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found at timestamp {timestamp}s"
        )

    return {
        "video_id": video_id,
        "timestamp": timestamp,
        "segment": segment,
    }


@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: int):
    """Delete a video and all its associated data and files."""
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        # Delete video file
        video_path = Path(video["path"])
        if video_path.exists():
            video_path.unlink()
            logger.info(f"Deleted video file: {video_path}")

        # Delete frame files
        frames_dir = Path("data/frames") / f"video_{video_id}"
        if frames_dir.exists():
            import shutil
            shutil.rmtree(frames_dir)
            logger.info(f"Deleted frames directory: {frames_dir}")

        # Delete thumbnail
        thumbnail_path = Path("data/videos") / f"thumbnail_{video_id}.jpg"
        if thumbnail_path.exists():
            thumbnail_path.unlink()
            logger.info(f"Deleted thumbnail: {thumbnail_path}")

        # Delete from Qdrant
        try:
            embedding_service.client.delete(
                collection_name="video_frames",
                points_selector={
                    "filter": {
                        "must": [{"key": "video_id", "match": {"value": video_id}}]
                    }
                },
            )
            logger.info(f"Deleted video {video_id} embeddings from Qdrant")
        except Exception as e:
            logger.warning(f"Could not delete embeddings from Qdrant: {e}")

        # Delete from database (cascades to frames, embeddings, etc.)
        db.delete_video(video_id)

        return {"message": f"Video {video_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/videos/{video_id}/reprocess")
async def reprocess_video(video_id: int, background_tasks: BackgroundTasks):
    """Reprocess a video that failed or needs re-analysis."""
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(video["path"])
    if not video_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Video file not found. Cannot reprocess. Please re-upload."
        )

    # Reset video status
    db.update_video_status(video_id, "processing")

    # Clear old frame data
    try:
        cursor = db.connection.cursor()
        # Delete old frames
        cursor.execute("DELETE FROM frames WHERE video_id = ?", (video_id,))
        # Delete old transcripts
        cursor.execute("DELETE FROM transcripts WHERE video_id = ?", (video_id,))
        db.connection.commit()
        logger.info(f"Cleared old data for video {video_id}")
    except Exception as e:
        logger.error(f"Error clearing old data: {e}")

    # Delete old embeddings from Qdrant
    try:
        embedding_service.client.delete(
            collection_name="video_frames",
            points_selector={
                "filter": {
                    "must": [{"key": "video_id", "match": {"value": video_id}}]
                }
            },
        )
        logger.info(f"Cleared old embeddings for video {video_id}")
    except Exception as e:
        logger.warning(f"Could not clear embeddings: {e}")

    # Start reprocessing in background
    background_tasks.add_task(process_video_task, video_id, str(video_path))

    logger.info(f"Started reprocessing video {video_id}")

    return {
        "message": "Video reprocessing started",
        "video_id": video_id,
        "status": "processing"
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Video Chat Server...")
    logger.info("Web UI will be available at http://localhost:8000")
    logger.info("API docs available at http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
