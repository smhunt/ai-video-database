"""Test script for audio transcription feature."""
import sys
from pathlib import Path
from loguru import logger

from src.services.video_processor import VideoProcessor
from src.services.audio_transcription import AudioTranscriptionService
from src.models.database import get_db


def test_audio_extraction(video_path: str):
    """Test audio extraction from video."""
    logger.info("=" * 60)
    logger.info("Testing Audio Extraction")
    logger.info("=" * 60)

    processor = VideoProcessor()

    # Check if video has audio
    has_audio = processor.has_audio(video_path)
    logger.info(f"Video has audio: {has_audio}")

    if not has_audio:
        logger.warning("Video does not have audio track!")
        return None

    # Extract audio
    logger.info("Extracting audio...")
    audio_path = processor.extract_audio(video_path, format="mp3")
    logger.info(f"Audio extracted to: {audio_path}")

    # Check file size
    audio_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
    logger.info(f"Audio file size: {audio_size_mb:.2f} MB")

    return audio_path


def test_transcription(audio_path: str):
    """Test audio transcription."""
    logger.info("=" * 60)
    logger.info("Testing Audio Transcription")
    logger.info("=" * 60)

    service = AudioTranscriptionService()

    # Validate audio file
    is_valid, message = service.validate_audio_file(audio_path)
    logger.info(f"Audio validation: {is_valid} - {message}")

    if not is_valid:
        logger.error(f"Audio file invalid: {message}")
        return None

    # Transcribe with timestamps
    logger.info("Transcribing audio (this may take a while)...")
    try:
        segments = service.transcribe_with_timestamps(audio_path)

        logger.info(f"Transcription complete!")
        logger.info(f"Total segments: {len(segments)}")

        # Show first 5 segments
        logger.info("\nFirst 5 segments:")
        for i, seg in enumerate(segments[:5], 1):
            logger.info(
                f"{i}. [{seg['start_time']:.2f}s - {seg['end_time']:.2f}s] "
                f"(confidence: {seg['confidence']:.2%}) {seg['text']}"
            )

        # Show last segment
        if len(segments) > 5:
            last = segments[-1]
            logger.info(
                f"... Last segment: [{last['start_time']:.2f}s - {last['end_time']:.2f}s] "
                f"{last['text']}"
            )

        # Estimate cost
        duration = segments[-1]["end_time"] if segments else 0
        cost = service.estimate_cost(duration)
        logger.info(f"\nEstimated cost: ${cost:.4f}")

        return segments

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None


def test_database_integration(video_id: int, segments: list):
    """Test saving transcripts to database."""
    logger.info("=" * 60)
    logger.info("Testing Database Integration")
    logger.info("=" * 60)

    db = get_db()

    # Save segments to database
    logger.info(f"Saving {len(segments)} segments to database...")
    for segment in segments:
        db.add_transcript_segment(
            video_id=video_id,
            start_time=segment["start_time"],
            end_time=segment["end_time"],
            text=segment["text"],
            confidence=segment["confidence"],
            model_name="whisper-1",
        )

    logger.info("Segments saved successfully!")

    # Retrieve all segments
    retrieved = db.get_transcripts_for_video(video_id)
    logger.info(f"Retrieved {len(retrieved)} segments from database")

    # Test full transcript
    full_text = db.get_full_transcript_text(video_id)
    logger.info(f"Full transcript length: {len(full_text)} characters")
    logger.info(f"Preview: {full_text[:200]}...")

    # Test timestamp lookup
    test_timestamp = 10.0
    segment_at_time = db.get_transcript_at_timestamp(video_id, test_timestamp)
    if segment_at_time:
        logger.info(f"\nSegment at {test_timestamp}s: {segment_at_time['text']}")

    # Test search
    search_term = "the"
    results = db.search_transcript(video_id, search_term)
    logger.info(f'\nSearch results for "{search_term}": {len(results)} matches')
    if results:
        logger.info(f"First match: {results[0]['text']}")


def main():
    """Run all tests."""
    if len(sys.argv) < 2:
        logger.error("Usage: python test_transcription.py <video_path> [video_id]")
        logger.info("Example: python test_transcription.py data/videos/sample.mp4 1")
        sys.exit(1)

    video_path = sys.argv[1]
    video_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    logger.info(f"Testing transcription for: {video_path}")
    logger.info(f"Video ID: {video_id}")

    # Test 1: Audio extraction
    audio_path = test_audio_extraction(video_path)
    if not audio_path:
        logger.error("Audio extraction failed or no audio track")
        sys.exit(1)

    # Test 2: Transcription
    segments = test_transcription(audio_path)
    if not segments:
        logger.error("Transcription failed")
        # Clean up audio file
        if Path(audio_path).exists():
            Path(audio_path).unlink()
        sys.exit(1)

    # Test 3: Database integration
    test_database_integration(video_id, segments)

    # Clean up
    logger.info("\nCleaning up temporary audio file...")
    if Path(audio_path).exists():
        Path(audio_path).unlink()
        logger.info("Cleanup complete!")

    logger.info("\n" + "=" * 60)
    logger.info("All tests completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
