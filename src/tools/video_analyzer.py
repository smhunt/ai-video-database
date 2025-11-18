"""Video content analyzer using Claude Vision for understanding and indexing."""
import anthropic
import instructor
import base64
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path

MAX_IMAGES_PER_BATCH = 100

FRAME_ANALYSIS_SYSTEM_PROMPT = """
You are a video content analyzer. You analyze frames extracted from videos and provide structured descriptions.

Your task is to analyze each frame and provide:
1. **Description**: A concise 1-2 sentence description of what's happening
2. **Objects**: List of key objects, people, or elements visible (max 10)
3. **Actions**: List of actions or activities taking place (max 5)
4. **Scene Type**: Category like "indoor", "outdoor", "close-up", "wide shot", "action", "dialogue", "landscape", etc.
5. **Excitement Score**: 1-10 rating of how interesting/exciting/highlight-worthy this moment is

Be accurate and specific. Focus on visual content, not assumptions.
"""


class FrameAnalysisResult(BaseModel):
    """Analysis result for a single frame."""

    frame_index: int = Field(..., description="Index of the frame in the batch")
    description: str = Field(..., description="Concise description of frame content")
    objects: List[str] = Field(
        default_factory=list, description="Key objects or people visible"
    )
    actions: List[str] = Field(default_factory=list, description="Actions taking place")
    scene_type: str = Field(..., description="Category of scene")
    excitement_score: int = Field(
        ..., ge=1, le=10, description="Highlight-worthiness rating 1-10"
    )


class BatchFrameAnalysis(BaseModel):
    """Analysis results for a batch of frames."""

    frames: List[FrameAnalysisResult] = Field(
        default_factory=list, description="Analysis for each frame"
    )


class VideoAnalyzer:
    """Analyzes video frames using Claude Vision."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        max_batch_size: int = MAX_IMAGES_PER_BATCH,
    ):
        """Initialize analyzer with Claude client."""
        base_client = anthropic.Anthropic()
        self.client = instructor.from_anthropic(base_client)
        self.model = model
        self.max_batch_size = max_batch_size
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_images = 0
        logger.info(f"VideoAnalyzer initialized with model {model}")

    def get_usage_stats(self) -> dict:
        """Get usage statistics for cost tracking."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "num_images": self.total_images,
        }

    def _load_image_as_base64(self, image_path: str) -> str:
        """Load image file and encode as base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_frames(
        self, frame_paths: List[str], context: Optional[str] = None
    ) -> List[FrameAnalysisResult]:
        """
        Analyze multiple frames in batch.

        Args:
            frame_paths: List of paths to frame images
            context: Optional context about the video

        Returns:
            List of analysis results, one per frame
        """
        if not frame_paths:
            return []

        all_results = []
        total_frames = len(frame_paths)

        logger.info(f"Analyzing {total_frames} frames...")

        # Process in batches
        for batch_start in range(0, total_frames, self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, total_frames)
            batch_paths = frame_paths[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start // self.max_batch_size + 1}: "
                f"frames {batch_start}-{batch_end - 1}"
            )

            batch_results = self._analyze_batch(batch_paths, batch_start, context)
            all_results.extend(batch_results)

        logger.info(f"Completed analysis of {len(all_results)} frames")
        return all_results

    def _analyze_batch(
        self, frame_paths: List[str], start_index: int, context: Optional[str]
    ) -> List[FrameAnalysisResult]:
        """Analyze a single batch of frames."""
        # Build message content with all frames
        message_content = []
        images_in_batch = 0

        for i, frame_path in enumerate(frame_paths):
            try:
                image_data = self._load_image_as_base64(frame_path)
                message_content.extend(
                    [
                        {
                            "type": "text",
                            "text": f"Frame {start_index + i}:",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                    ]
                )
                images_in_batch += 1
            except Exception as e:
                logger.error(f"Failed to load frame {frame_path}: {e}")
                continue

        # Add context if provided
        prompt = "Analyze each frame and provide structured information."
        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        message_content.append({"type": "text", "text": prompt})

        # Retry logic for rate limits
        import time
        max_retries = 3
        retry_delay = 60  # Start with 60 second delay

        for attempt in range(max_retries):
            try:
                # Call Claude with structured output
                batch_analysis = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=FRAME_ANALYSIS_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": message_content}],
                    response_model=BatchFrameAnalysis,
                )

                # Track usage for cost calculation
                self.total_images += images_in_batch
                # Approximate token usage (actual usage not available with instructor)
                self.total_input_tokens += images_in_batch * 1568 + 100  # ~1568 tokens per image
                self.total_output_tokens += len(batch_analysis.frames) * 150  # ~150 tokens per frame result

                logger.info(f"Analyzed {len(batch_analysis.frames)} frames in batch")
                return batch_analysis.frames

            except Exception as e:
                error_str = str(e)

                # Check if it's a rate limit error
                if "rate_limit_error" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                            f"Waiting {retry_delay}s before retry..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error("Max retries reached for rate limit. Skipping batch.")
                else:
                    logger.error(f"Failed to analyze batch: {e}")

                # Return empty results for this batch if all retries failed
                return []

        return []

    def analyze_single_frame(
        self, frame_path: str, query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single frame with optional specific query.

        Args:
            frame_path: Path to frame image
            query: Optional specific question about the frame

        Returns:
            Dict with description and answer (if query provided)
        """
        try:
            image_data = self._load_image_as_base64(frame_path)

            # Build prompt based on whether there's a query
            if query:
                prompt = f"Question: {query}\n\nProvide a detailed answer based on what you see in this frame."
            else:
                prompt = "Describe this frame in detail, including objects, actions, setting, and any notable elements."

            message_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt},
            ]

            # Use regular Claude call for flexible response
            response = anthropic.Anthropic().messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": message_content}],
            )

            answer = response.content[0].text

            logger.info(f"Analyzed single frame: {frame_path}")

            return {
                "frame_path": frame_path,
                "query": query,
                "answer": answer,
            }

        except Exception as e:
            logger.error(f"Failed to analyze frame {frame_path}: {e}")
            return {
                "frame_path": frame_path,
                "query": query,
                "answer": f"Error analyzing frame: {str(e)}",
            }

    def find_exciting_moments(
        self, analysis_results: List[FrameAnalysisResult], threshold: int = 7, top_n: Optional[int] = None
    ) -> List[int]:
        """
        Find frame indices of exciting/highlight-worthy moments.

        Args:
            analysis_results: List of frame analysis results
            threshold: Minimum excitement score (1-10)
            top_n: Optional limit to top N most exciting moments

        Returns:
            List of frame indices sorted by excitement score
        """
        # Filter by threshold
        exciting = [
            (result.frame_index, result.excitement_score)
            for result in analysis_results
            if result.excitement_score >= threshold
        ]

        # Sort by score descending
        exciting.sort(key=lambda x: x[1], reverse=True)

        # Limit to top N if specified
        if top_n:
            exciting = exciting[:top_n]

        frame_indices = [idx for idx, _ in exciting]
        logger.info(
            f"Found {len(frame_indices)} exciting moments "
            f"(threshold={threshold}, top_n={top_n})"
        )

        return frame_indices

    def search_frames_by_query(
        self, analysis_results: List[FrameAnalysisResult], query: str
    ) -> List[int]:
        """
        Search for frames matching a text query using simple keyword matching.

        Args:
            analysis_results: List of frame analysis results
            query: Search query

        Returns:
            List of matching frame indices
        """
        query_lower = query.lower()
        matching_indices = []

        for result in analysis_results:
            # Check description, objects, actions
            description = result.description.lower()
            objects = " ".join(result.objects).lower()
            actions = " ".join(result.actions).lower()
            scene_type = result.scene_type.lower()

            combined = f"{description} {objects} {actions} {scene_type}"

            if query_lower in combined:
                matching_indices.append(result.frame_index)

        logger.info(f"Found {len(matching_indices)} frames matching '{query}'")
        return matching_indices


if __name__ == "__main__":
    # Test with sample frames
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_analyzer.py <frame1.jpg> [frame2.jpg] ...")
        sys.exit(1)

    frame_paths = sys.argv[1:]
    analyzer = VideoAnalyzer()

    results = analyzer.analyze_frames(frame_paths, context="Test video frames")

    print("\n=== Analysis Results ===\n")
    for result in results:
        print(f"Frame {result.frame_index}:")
        print(f"  Description: {result.description}")
        print(f"  Objects: {', '.join(result.objects)}")
        print(f"  Actions: {', '.join(result.actions)}")
        print(f"  Scene Type: {result.scene_type}")
        print(f"  Excitement: {result.excitement_score}/10")
        print()

    # Find exciting moments
    exciting = analyzer.find_exciting_moments(results, threshold=7)
    print(f"Exciting moments (score >= 7): {exciting}")
