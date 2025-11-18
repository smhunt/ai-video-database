"""Video processing service using ffmpeg for frame extraction and analysis."""
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import math


class VideoProcessor:
    """Handles video processing operations using ffmpeg."""

    def __init__(self, output_dir: str = "data/videos", frames_dir: str = "data/frames"):
        """Initialize video processor with output directories."""
        self.output_dir = Path(output_dir)
        self.frames_dir = Path(frames_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Extract video metadata using ffprobe.

        Returns:
            Dict with duration_seconds, fps, width, height, codec, bitrate
        """
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Find video stream
            video_stream = next(
                (s for s in data["streams"] if s["codec_type"] == "video"), None
            )

            if not video_stream:
                raise ValueError("No video stream found")

            # Parse FPS (can be fraction like "30/1")
            fps_str = video_stream.get("r_frame_rate", "30/1")
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_str)

            info = {
                "duration_seconds": float(data["format"].get("duration", 0)),
                "fps": fps,
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "codec": video_stream.get("codec_name", "unknown"),
                "bitrate": int(data["format"].get("bit_rate", 0)),
                "size_bytes": int(data["format"].get("size", 0)),
            }

            logger.info(
                f"Video info: {info['duration_seconds']:.1f}s, "
                f"{info['fps']:.2f}fps, {info['width']}x{info['height']}"
            )
            return info

        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe error: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise

    def extract_frames(
        self,
        video_path: str,
        video_id: int,
        interval_seconds: float = 2.0,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from video at regular intervals.

        Args:
            video_path: Path to input video
            video_id: Database ID for organizing frames
            interval_seconds: Time interval between frames
            max_frames: Maximum number of frames to extract (None = all)

        Returns:
            List of dicts with frame_number, timestamp_seconds, frame_path
        """
        video_info = self.get_video_info(video_path)
        duration = video_info["duration_seconds"]
        fps = video_info["fps"]

        # Calculate frame extraction parameters
        total_possible_frames = int(duration / interval_seconds)
        if max_frames and total_possible_frames > max_frames:
            # Adjust interval to stay under max_frames
            interval_seconds = duration / max_frames
            total_possible_frames = max_frames

        logger.info(
            f"Extracting ~{total_possible_frames} frames at {interval_seconds:.2f}s intervals"
        )

        # Create output directory for this video
        video_frames_dir = self.frames_dir / f"video_{video_id}"
        video_frames_dir.mkdir(parents=True, exist_ok=True)

        # ffmpeg command to extract frames at intervals
        # Using fps filter to sample at specific rate
        sample_fps = 1.0 / interval_seconds
        output_pattern = str(video_frames_dir / "frame_%06d.jpg")

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps={sample_fps}",
            "-q:v",
            "2",  # High quality JPEG
            "-y",  # Overwrite existing files
            output_pattern,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=300
            )
            logger.info(f"Frame extraction completed")

            # Build frame list
            frames = []
            for i, frame_file in enumerate(sorted(video_frames_dir.glob("frame_*.jpg"))):
                timestamp = i * interval_seconds
                frame_number = int(timestamp * fps)

                frames.append(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": timestamp,
                        "frame_path": str(frame_file),
                    }
                )

            logger.info(f"Extracted {len(frames)} frames")
            return frames

        except subprocess.TimeoutExpired:
            logger.error("Frame extraction timed out")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg error: {e.stderr}")
            raise

    def extract_single_frame(
        self, video_path: str, timestamp: float, output_path: str
    ) -> str:
        """
        Extract a single frame at specific timestamp.

        Args:
            video_path: Path to input video
            timestamp: Time in seconds
            output_path: Where to save the frame

        Returns:
            Path to extracted frame
        """
        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            logger.info(f"Extracted frame at {timestamp}s to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to extract frame at {timestamp}s: {e}")
            raise

    def detect_scenes(
        self, video_path: str, threshold: float = 0.3
    ) -> List[Dict[str, float]]:
        """
        Detect scene changes in video using ffmpeg.

        Args:
            video_path: Path to input video
            threshold: Scene detection sensitivity (0-1, lower = more sensitive)

        Returns:
            List of scene timestamps with score
        """
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"select='gt(scene,{threshold})',metadata=print:file=-",
            "-f",
            "null",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=300
            )

            # Parse scene changes from metadata output
            scenes = []
            for line in result.stderr.split("\n"):
                if "pts_time:" in line and "scene" in line:
                    # Extract timestamp
                    try:
                        pts_time = float(line.split("pts_time:")[1].split()[0])
                        score_part = line.split("scene:")[1].split()[0] if "scene:" in line else "0"
                        score = float(score_part) if score_part else 0
                        scenes.append({"timestamp": pts_time, "score": score})
                    except (IndexError, ValueError):
                        continue

            logger.info(f"Detected {len(scenes)} scene changes")
            return scenes

        except subprocess.TimeoutExpired:
            logger.error("Scene detection timed out")
            return []
        except subprocess.CalledProcessError as e:
            logger.error(f"Scene detection failed: {e.stderr}")
            return []

    def extract_keyframes_at_scenes(
        self, video_path: str, video_id: int, max_keyframes: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Extract keyframes at scene changes.

        Args:
            video_path: Path to input video
            video_id: Database ID for organizing frames
            max_keyframes: Maximum number of keyframes

        Returns:
            List of keyframe info dicts
        """
        scenes = self.detect_scenes(video_path)

        if not scenes:
            logger.warning("No scenes detected, falling back to uniform sampling")
            return self.extract_frames(video_path, video_id, interval_seconds=5.0, max_frames=max_keyframes)

        # Sort by score and take top N
        scenes.sort(key=lambda x: x["score"], reverse=True)
        selected_scenes = scenes[:max_keyframes]
        selected_scenes.sort(key=lambda x: x["timestamp"])  # Re-sort by time

        logger.info(f"Extracting {len(selected_scenes)} keyframes at scene changes")

        # Create output directory
        video_frames_dir = self.frames_dir / f"video_{video_id}" / "keyframes"
        video_frames_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames at scene timestamps
        keyframes = []
        video_info = self.get_video_info(video_path)
        fps = video_info["fps"]

        for i, scene in enumerate(selected_scenes):
            timestamp = scene["timestamp"]
            output_path = video_frames_dir / f"keyframe_{i:04d}.jpg"

            try:
                self.extract_single_frame(video_path, timestamp, str(output_path))
                frame_number = int(timestamp * fps)

                keyframes.append(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": timestamp,
                        "frame_path": str(output_path),
                        "is_keyframe": True,
                        "scene_score": scene["score"],
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to extract keyframe at {timestamp}s: {e}")
                continue

        logger.info(f"Extracted {len(keyframes)} keyframes")
        return keyframes

    def generate_thumbnail_strip(
        self, video_path: str, output_path: str, num_thumbs: int = 10, thumb_width: int = 160
    ) -> str:
        """
        Generate a thumbnail strip image for timeline preview.

        Args:
            video_path: Path to input video
            output_path: Where to save thumbnail strip
            num_thumbs: Number of thumbnails
            thumb_width: Width of each thumbnail

        Returns:
            Path to generated strip
        """
        video_info = self.get_video_info(video_path)
        duration = video_info["duration_seconds"]

        # Calculate thumbnail interval
        interval = duration / num_thumbs

        # Generate thumbnails and stitch them
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps=1/{interval},scale={thumb_width}:-1,tile={num_thumbs}x1",
            "-y",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
            logger.info(f"Generated thumbnail strip with {num_thumbs} frames")
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate thumbnail strip: {e}")
            raise

    def extract_audio(
        self, video_path: str, output_path: Optional[str] = None, format: str = "mp3"
    ) -> str:
        """
        Extract audio from video file.

        Args:
            video_path: Path to input video
            output_path: Where to save audio (None = auto-generate)
            format: Audio format (mp3, wav, m4a, etc.)

        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(
                self.output_dir / f"{video_path_obj.stem}_audio.{format}"
            )

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",  # No video
            "-acodec",
            "libmp3lame" if format == "mp3" else "copy",
            "-q:a",
            "2",  # High quality
            "-y",  # Overwrite
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=300
            )
            logger.info(f"Extracted audio to {output_path}")
            return output_path
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr}")
            raise

    def has_audio(self, video_path: str) -> bool:
        """
        Check if video has an audio stream.

        Args:
            video_path: Path to video file

        Returns:
            True if video has audio stream
        """
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            video_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Check if any audio stream exists
            audio_stream = next(
                (s for s in data["streams"] if s["codec_type"] == "audio"), None
            )
            return audio_stream is not None

        except Exception as e:
            logger.error(f"Failed to check audio stream: {e}")
            return False

    def validate_video(self, video_path: str, max_size_gb: float = 2.0) -> Tuple[bool, str]:
        """
        Validate video file.

        Args:
            video_path: Path to video
            max_size_gb: Maximum allowed size in GB

        Returns:
            (is_valid, error_message)
        """
        path = Path(video_path)

        # Check file exists
        if not path.exists():
            return False, "File does not exist"

        # Check size
        size_gb = path.stat().st_size / (1024**3)
        if size_gb > max_size_gb:
            return False, f"File too large: {size_gb:.2f}GB (max {max_size_gb}GB)"

        # Check if valid video with ffprobe
        try:
            info = self.get_video_info(video_path)
            if info["duration_seconds"] <= 0:
                return False, "Invalid video duration"
            return True, "Valid"
        except Exception as e:
            return False, f"Invalid video file: {str(e)}"
