# 🎙️ Audio Transcription Feature

Automatic speech-to-text transcription for video files using OpenAI Whisper API.

---

## 🌟 Features

- **Automatic Transcription**: Audio is automatically transcribed when videos are uploaded
- **Timestamped Segments**: Each transcript segment includes precise start/end timestamps
- **Full-Text Search**: Search for specific words or phrases in transcripts
- **Timestamp Lookup**: Find what was said at any specific moment
- **Multi-language Support**: Whisper supports 99+ languages
- **Cost Tracking**: Automatic tracking of transcription costs

---

## 🚀 Setup

### 1. Environment Variables

Add your OpenAI API key to `.env`:

```bash
OPENAI_API_KEY=sk-...
```

### 2. Dependencies

All required dependencies are already included in the project:
- `openai>=1.60.2` - OpenAI Whisper API client
- `ffmpeg` - Audio extraction (should already be installed)

---

## 📖 How It Works

### Automatic Processing Pipeline

When you upload a video, the system automatically:

1. **Checks for Audio**: Detects if video has an audio track
2. **Extracts Audio**: Uses ffmpeg to extract audio as MP3
3. **Transcribes**: Sends audio to OpenAI Whisper API
4. **Stores Segments**: Saves timestamped segments to database
5. **Tracks Costs**: Records transcription costs for billing

### Database Schema

Transcripts are stored in the `transcripts` table:

```sql
CREATE TABLE transcripts (
    id INTEGER PRIMARY KEY,
    video_id INTEGER,
    start_time REAL,      -- Segment start (seconds)
    end_time REAL,        -- Segment end (seconds)
    text TEXT,            -- Transcribed text
    language TEXT,        -- Detected language
    confidence REAL,      -- Confidence score (0-1)
    model_name TEXT,      -- "whisper-1"
    created_at TIMESTAMP
)
```

---

## 🔧 API Endpoints

### Get Full Transcript

```bash
GET /api/videos/{video_id}/transcript
```

**Response:**
```json
{
  "video_id": 1,
  "filename": "my_video.mp4",
  "segment_count": 42,
  "segments": [
    {
      "id": 1,
      "start_time": 0.0,
      "end_time": 3.5,
      "text": "Welcome to this video",
      "confidence": 0.95,
      "language": "en"
    },
    ...
  ],
  "full_text": "Welcome to this video..."
}
```

### Search Transcript

```bash
GET /api/videos/{video_id}/transcript/search?q=keyword
```

**Example:**
```bash
curl "http://localhost:8000/api/videos/1/transcript/search?q=machine+learning"
```

**Response:**
```json
{
  "video_id": 1,
  "query": "machine learning",
  "result_count": 3,
  "results": [
    {
      "id": 15,
      "start_time": 45.2,
      "end_time": 48.7,
      "text": "Machine learning is transforming AI",
      "confidence": 0.97
    },
    ...
  ]
}
```

### Get Transcript at Timestamp

```bash
GET /api/videos/{video_id}/transcript/at/{timestamp}
```

**Example:**
```bash
curl "http://localhost:8000/api/videos/1/transcript/at/120.5"
```

**Response:**
```json
{
  "video_id": 1,
  "timestamp": 120.5,
  "segment": {
    "id": 30,
    "start_time": 118.0,
    "end_time": 122.5,
    "text": "Now let's look at the implementation details",
    "confidence": 0.94
  }
}
```

---

## 💻 Programmatic Usage

### Extract and Transcribe Audio

```python
from src.services.video_processor import VideoProcessor
from src.services.audio_transcription import AudioTranscriptionService

# Initialize services
video_processor = VideoProcessor()
transcription_service = AudioTranscriptionService()

# Extract audio
video_path = "data/videos/my_video.mp4"
audio_path = video_processor.extract_audio(video_path, format="mp3")

# Transcribe with timestamps
segments = transcription_service.transcribe_with_timestamps(audio_path)

# Print results
for seg in segments:
    print(f"[{seg['start_time']:.2f}s] {seg['text']}")
```

### Access from Database

```python
from src.models.database import get_db

db = get_db()

# Get all segments for a video
segments = db.get_transcripts_for_video(video_id=1)

# Get full transcript as text
full_text = db.get_full_transcript_text(video_id=1)

# Find what was said at a specific time
segment = db.get_transcript_at_timestamp(video_id=1, timestamp=45.0)

# Search for keywords
results = db.search_transcript(video_id=1, search_term="important")
```

---

## 🧪 Testing

### Run Test Script

```bash
# Test with a video file
python test_transcription.py data/videos/sample.mp4 1

# The script will:
# 1. Extract audio from video
# 2. Transcribe using Whisper
# 3. Save to database
# 4. Test all database operations
```

### Manual Testing

```bash
# Start the server
python video_chat_server.py

# Upload a video (will auto-transcribe)
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/upload

# Get transcript (replace {video_id} with actual ID)
curl http://localhost:8000/api/videos/1/transcript

# Search transcript
curl "http://localhost:8000/api/videos/1/transcript/search?q=hello"
```

---

## 💰 Cost Information

### OpenAI Whisper Pricing

- **Cost**: $0.006 per minute of audio
- **File Limit**: 25 MB max file size

### Cost Examples

| Video Duration | Estimated Cost |
|---------------|---------------|
| 5 minutes     | $0.03         |
| 10 minutes    | $0.06         |
| 30 minutes    | $0.18         |
| 1 hour        | $0.36         |

### Cost Tracking

All transcription costs are automatically tracked:

```bash
# Get costs for a video
curl http://localhost:8000/api/videos/1/costs
```

**Response includes:**
```json
{
  "operations": [
    {
      "operation_type": "transcription",
      "api_provider": "openai",
      "model_name": "whisper-1",
      "estimated_cost_usd": 0.0360,
      "details": "Transcribed 42 segments, 360.0s audio"
    }
  ]
}
```

---

## 🌍 Supported Languages

Whisper supports 99+ languages including:

- English, Spanish, French, German, Italian
- Chinese (Mandarin), Japanese, Korean
- Arabic, Russian, Portuguese
- And many more...

**Auto-detection**: Language is automatically detected if not specified.

---

## 🎛️ Configuration

### Custom Whisper Settings

Edit `src/services/audio_transcription.py`:

```python
# Temperature (0-1): Higher = more creative, Lower = more deterministic
temperature = 0.0  # Default: deterministic transcription

# Language (optional): Force specific language
language = "en"  # Or None for auto-detection

# Prompt (optional): Guide the model's style
prompt = "This is a technical video about machine learning."
```

### Audio Format

By default, audio is extracted as MP3. You can change this in `video_processor.py`:

```python
# Supported formats: mp3, wav, m4a, flac, webm
audio_path = video_processor.extract_audio(video_path, format="mp3")
```

---

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"

**Solution**: Add your OpenAI API key to `.env`:
```bash
OPENAI_API_KEY=sk-...
```

### "Audio file too large (max 25MB)"

**Solution**: For videos longer than ~40 minutes, the audio file may exceed Whisper's 25MB limit.

**Workaround**:
1. Split video into chunks
2. Use lower bitrate audio extraction
3. Or implement chunked transcription (not yet implemented)

### "No audio track found"

**Solution**: Video doesn't have an audio stream. Check with:
```bash
ffprobe -v error -show_streams video.mp4
```

### Transcription Quality Issues

**Tips for better quality**:
- Use high-quality audio sources
- Minimize background noise
- Provide a prompt for technical terms
- Specify language if auto-detection fails

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Speaker Diarization**: Identify different speakers
- [ ] **Custom Vocabulary**: Support for technical terms
- [ ] **Subtitle Export**: Generate SRT/VTT subtitle files
- [ ] **Real-time Transcription**: Transcribe as video uploads
- [ ] **Chunked Processing**: Handle videos > 25MB
- [ ] **Semantic Search**: Index transcripts in Qdrant for better search
- [ ] **Translation**: Translate transcripts to other languages

---

## 📊 Performance

### Benchmarks

| Task | Time | Details |
|------|------|---------|
| Audio Extraction | ~5s | 10 min video → MP3 |
| Transcription | ~30-60s | Depends on audio length & API |
| Database Storage | <1s | Save all segments |

**Total processing time**: ~1-2 minutes for 10-minute video

---

## 🔗 Integration with Existing Features

### Video Chat Enhancement

Transcripts can enhance the chat experience:

```python
# Include transcript context when answering questions
transcript = db.get_full_transcript_text(video_id)

# Use transcript to answer "what was said" questions
query = "What did they say about AI?"
matching_segments = db.search_transcript(video_id, "AI")
```

### Semantic Search Upgrade

Future: Index transcript embeddings in Qdrant for semantic search:

```python
# Search both visual frames AND audio transcript
visual_results = embedding_service.search_frames("explosion", video_id)
audio_results = search_transcript_semantically("loud noise", video_id)
```

---

## 🤝 Contributing

Found a bug? Have ideas for improvements?

**Key areas for contribution**:
- Speaker diarization support
- Better handling of long videos
- Subtitle file export
- Multi-language UI support

---

## 📄 License

Part of the AI Video Database project.

---

## 🙏 Acknowledgments

Built with:
- **OpenAI Whisper** - State-of-the-art speech recognition
- **FFmpeg** - Audio extraction
- **FastAPI** - API framework
- **SQLite** - Transcript storage

---

**Made with ❤️ for the AI Video Database project**
