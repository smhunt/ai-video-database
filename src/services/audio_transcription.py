"""Audio transcription service using OpenAI Whisper API."""
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from openai import OpenAI
import os


class AudioTranscriptionService:
    """Handles audio transcription using OpenAI Whisper API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        """
        Initialize transcription service.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Whisper model to use (whisper-1)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        logger.info(f"Initialized AudioTranscriptionService with model {model}")

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "verbose_json",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper API.

        Args:
            audio_path: Path to audio file (mp3, mp4, wav, etc.)
            language: ISO-639-1 language code (e.g., 'en', 'es')
            prompt: Optional text to guide the model's style
            response_format: Response format (json, text, srt, verbose_json, vtt)
            temperature: Sampling temperature (0-1)

        Returns:
            Transcription result with text and metadata
        """
        audio_file = Path(audio_path)

        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio file: {audio_path}")

        try:
            with open(audio_path, "rb") as audio:
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio,
                    language=language,
                    prompt=prompt,
                    response_format=response_format,
                    temperature=temperature,
                )

            # Handle different response formats
            if response_format == "verbose_json":
                result = {
                    "text": response.text,
                    "language": response.language,
                    "duration": response.duration,
                    "segments": [
                        {
                            "id": seg.id,
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                            "temperature": seg.temperature,
                            "avg_logprob": seg.avg_logprob,
                            "compression_ratio": seg.compression_ratio,
                            "no_speech_prob": seg.no_speech_prob,
                        }
                        for seg in response.segments
                    ],
                }
                logger.info(
                    f"Transcription complete: {len(response.segments)} segments, "
                    f"{response.duration:.1f}s, language: {response.language}"
                )
            elif response_format == "json":
                result = {"text": response.text}
                logger.info("Transcription complete (JSON format)")
            else:
                result = {"text": response}
                logger.info(f"Transcription complete ({response_format} format)")

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def transcribe_with_timestamps(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio and return segments with timestamps.

        Args:
            audio_path: Path to audio file
            language: Optional language code

        Returns:
            List of transcript segments with start/end times and text
        """
        result = self.transcribe_audio(
            audio_path, language=language, response_format="verbose_json"
        )

        segments = [
            {
                "start_time": seg["start"],
                "end_time": seg["end"],
                "text": seg["text"].strip(),
                "confidence": 1.0 - seg["no_speech_prob"],  # Rough confidence score
            }
            for seg in result["segments"]
        ]

        return segments

    def estimate_cost(self, audio_duration_seconds: float) -> float:
        """
        Estimate transcription cost based on audio duration.

        Whisper API pricing: $0.006 per minute (as of 2024)

        Args:
            audio_duration_seconds: Duration in seconds

        Returns:
            Estimated cost in USD
        """
        duration_minutes = audio_duration_seconds / 60.0
        cost_per_minute = 0.006
        estimated_cost = duration_minutes * cost_per_minute
        return round(estimated_cost, 4)

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported audio formats for Whisper API.

        Returns:
            List of file extensions
        """
        return [
            "mp3",
            "mp4",
            "mpeg",
            "mpga",
            "m4a",
            "wav",
            "webm",
            "flac",
        ]

    def validate_audio_file(self, audio_path: str, max_size_mb: float = 25.0) -> tuple[bool, str]:
        """
        Validate audio file for transcription.

        Whisper API limits: 25MB file size

        Args:
            audio_path: Path to audio file
            max_size_mb: Maximum file size in MB

        Returns:
            (is_valid, error_message)
        """
        path = Path(audio_path)

        if not path.exists():
            return False, "File does not exist"

        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)"

        # Check file extension
        extension = path.suffix.lower().lstrip(".")
        if extension not in self.get_supported_formats():
            return False, f"Unsupported format: {extension}"

        return True, "Valid"


# Example usage
if __name__ == "__main__":
    # Initialize service
    service = AudioTranscriptionService()

    # Example: Transcribe an audio file
    audio_path = "data/videos/sample_audio.mp3"

    # Basic transcription
    result = service.transcribe_audio(audio_path)
    print(f"Full text: {result['text']}")

    # Transcription with timestamps
    segments = service.transcribe_with_timestamps(audio_path)
    for seg in segments[:5]:  # Show first 5 segments
        print(f"[{seg['start_time']:.1f}s - {seg['end_time']:.1f}s] {seg['text']}")

    # Cost estimation
    duration = 600  # 10 minutes
    cost = service.estimate_cost(duration)
    print(f"Estimated cost for {duration}s audio: ${cost:.4f}")
