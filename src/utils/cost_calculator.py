"""Cost calculation utilities for API usage tracking."""

# Pricing as of January 2025 (in USD)

PRICING = {
    "anthropic": {
        "claude-3-haiku-20240307": {
            "input_per_mtok": 0.25,
            "output_per_mtok": 1.25,
            "image_per_image": 0.0004,  # Approximation based on token equivalent
        },
        "claude-3-sonnet-20240229": {
            "input_per_mtok": 3.0,
            "output_per_mtok": 15.0,
            "image_per_image": 0.0048,
        },
        "claude-3-5-sonnet-20241022": {
            "input_per_mtok": 3.0,
            "output_per_mtok": 15.0,
            "image_per_image": 0.0048,
        },
    },
    "openai": {
        "text-embedding-3-large": {
            "per_mtok": 0.13,
        },
        "text-embedding-3-small": {
            "per_mtok": 0.02,
        },
        "text-embedding-ada-002": {
            "per_mtok": 0.10,
        },
    },
}


def calculate_claude_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    num_images: int = 0,
) -> float:
    """
    Calculate cost for Claude API usage.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        num_images: Number of images processed

    Returns:
        Estimated cost in USD
    """
    if model not in PRICING["anthropic"]:
        # Default to Haiku pricing if model not found
        model = "claude-3-haiku-20240307"

    pricing = PRICING["anthropic"][model]

    # Calculate token costs
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_mtok"]

    # Calculate image costs
    image_cost = num_images * pricing["image_per_image"]

    total_cost = input_cost + output_cost + image_cost

    return round(total_cost, 6)


def calculate_openai_embedding_cost(
    model: str,
    num_embeddings: int,
    avg_tokens_per_text: int = 50,
) -> float:
    """
    Calculate cost for OpenAI embedding API usage.

    Args:
        model: Model name
        num_embeddings: Number of texts embedded
        avg_tokens_per_text: Average tokens per text (default 50)

    Returns:
        Estimated cost in USD
    """
    if model not in PRICING["openai"]:
        # Default to text-embedding-3-large
        model = "text-embedding-3-large"

    pricing = PRICING["openai"][model]

    # Calculate total tokens
    total_tokens = num_embeddings * avg_tokens_per_text

    # Calculate cost
    cost = (total_tokens / 1_000_000) * pricing["per_mtok"]

    return round(cost, 6)


def estimate_frame_analysis_cost(num_frames: int, model: str = "claude-3-haiku-20240307") -> dict:
    """
    Estimate cost for analyzing video frames with Claude Vision.

    Args:
        num_frames: Number of frames to analyze
        model: Claude model to use

    Returns:
        Dict with cost breakdown
    """
    # Approximate tokens per frame analysis
    # Each frame generates ~1568 tokens for the image
    # Plus text tokens for prompt and response
    tokens_per_frame_image = 1568
    tokens_per_frame_prompt = 100
    tokens_per_frame_response = 150

    total_input_tokens = num_frames * (tokens_per_frame_image + tokens_per_frame_prompt)
    total_output_tokens = num_frames * tokens_per_frame_response

    cost = calculate_claude_cost(
        model=model,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        num_images=num_frames,
    )

    return {
        "num_frames": num_frames,
        "estimated_cost_usd": cost,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost_per_frame": round(cost / num_frames, 6) if num_frames > 0 else 0,
    }


def estimate_embedding_cost(num_descriptions: int, model: str = "text-embedding-3-large") -> dict:
    """
    Estimate cost for generating embeddings.

    Args:
        num_descriptions: Number of descriptions to embed
        model: Embedding model to use

    Returns:
        Dict with cost breakdown
    """
    # Average description is ~30-50 tokens
    avg_tokens = 40

    cost = calculate_openai_embedding_cost(
        model=model,
        num_embeddings=num_descriptions,
        avg_tokens_per_text=avg_tokens,
    )

    return {
        "num_embeddings": num_descriptions,
        "estimated_cost_usd": cost,
        "avg_tokens_per_text": avg_tokens,
        "total_tokens": num_descriptions * avg_tokens,
    }


def estimate_chat_query_cost(
    query_length: int = 50,
    response_length: int = 200,
    num_frames: int = 0,
    model: str = "claude-3-haiku-20240307",
) -> dict:
    """
    Estimate cost for a chat query.

    Args:
        query_length: Length of user query in tokens
        response_length: Expected response length in tokens
        num_frames: Number of frames included in context
        model: Claude model to use

    Returns:
        Dict with cost breakdown
    """
    # Base tokens for query
    input_tokens = query_length

    # Add frame context (description text, not images)
    input_tokens += num_frames * 100  # Approximate tokens per frame description

    output_tokens = response_length

    cost = calculate_claude_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_images=0,  # Chat doesn't reprocess images
    )

    return {
        "estimated_cost_usd": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def estimate_video_processing_cost(
    duration_seconds: float,
    frame_interval: float = 2.0,
    max_frames_to_analyze: int = 100,
) -> dict:
    """
    Estimate total cost for processing a video.

    Args:
        duration_seconds: Video duration in seconds
        frame_interval: Seconds between frame extractions
        max_frames_to_analyze: Maximum frames to analyze with AI

    Returns:
        Dict with complete cost breakdown
    """
    # Calculate number of frames
    total_frames = int(duration_seconds / frame_interval)
    frames_to_analyze = min(total_frames, max_frames_to_analyze)

    # Frame analysis cost
    analysis = estimate_frame_analysis_cost(frames_to_analyze)

    # Embedding cost
    embeddings = estimate_embedding_cost(frames_to_analyze)

    # Total cost
    total_cost = analysis["estimated_cost_usd"] + embeddings["estimated_cost_usd"]

    return {
        "video_duration_seconds": duration_seconds,
        "total_frames_extracted": total_frames,
        "frames_analyzed": frames_to_analyze,
        "frame_analysis_cost_usd": analysis["estimated_cost_usd"],
        "embedding_cost_usd": embeddings["estimated_cost_usd"],
        "total_cost_usd": round(total_cost, 4),
        "cost_per_minute": round(total_cost / (duration_seconds / 60), 4),
    }
