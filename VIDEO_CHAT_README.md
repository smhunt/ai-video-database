# 💬 Video Chat System

**Talk to your videos using AI!** Upload videos and have natural conversations about their content using Claude Vision and semantic search.

---

## 🌟 Features

### ✅ Implemented
- **Video Upload**: Drag & drop or click to upload videos up to 2GB
- **Auto-Indexing**: Automatic frame extraction and AI analysis on upload
- **Semantic Search**: "Find explosions" - searches video content intelligently
- **Timestamp Q&A**: "What happens at 1:23?" - analyzes specific moments
- **Visual Timeline**: Thumbnail preview strip with clickable keyframes
- **Chat Interface**: Natural conversation about video content
- **Scene Detection**: Automatically identifies and extracts keyframes at scene changes
- **Excitement Scoring**: AI rates each frame for highlight-worthiness (1-10)

---

## 🚀 Quick Start

### 1. Install System Dependencies

**FFmpeg** (required for video processing):

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 2. Install Python Dependencies

```bash
# Install FastAPI and related packages
pip install -r requirements_video_chat.txt

# Main dependencies should already be installed
# (anthropic, instructor, qdrant-client, etc.)
```

### 3. Setup Environment Variables

Make sure your `.env` file has:

```bash
ANTHROPIC_API_KEY=your_key_here
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=your_key_here  # For embeddings
```

### 4. Start Qdrant (Vector Database)

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install locally: https://qdrant.tech/documentation/quick-start/
```

### 5. Run the Server

```bash
python video_chat_server.py
```

The server will start at: **http://localhost:8000**

---

## 📖 Usage Guide

### Upload a Video

1. Open http://localhost:8000 in your browser
2. Drag & drop a video file (or click to select)
3. Wait for processing (2-5 minutes depending on video length)
4. When status shows "Ready" ✅, start chatting!

### Chat Examples

**Timestamp Queries:**
- "What happens at 1:23?"
- "Describe the scene at 45 seconds"
- "What's going on at 2:15?"

**Content Search:**
- "Find explosions"
- "Show me outdoor scenes"
- "When do people appear?"
- "Find exciting moments"

**General Questions:**
- "Summarize this video"
- "What are the main themes?"
- "Describe the setting"

### Timeline Navigation

- Click thumbnails in the keyframe grid to jump to that moment
- Hover over frames in chat responses to see descriptions
- Click frame previews to seek video player

---

## 🏗️ Architecture

### Backend Components

```
video_chat_server.py          # FastAPI server (main entry point)
│
├── src/models/database.py    # SQLite database models
├── src/services/
│   ├── video_processor.py    # FFmpeg video operations
│   └── video_embeddings.py   # Qdrant semantic search
├── src/tools/
│   └── video_analyzer.py     # Claude Vision analysis
└── static/
    ├── index.html             # Web UI
    └── app.js                 # Frontend logic
```

### Data Flow

1. **Upload** → Video saved to `data/videos/`
2. **Extract** → Frames saved to `data/frames/video_{id}/`
3. **Analyze** → Claude Vision analyzes frames (batch of 100)
4. **Index** → Embeddings stored in Qdrant
5. **Ready** → User can chat with video

### Storage

- **SQLite Database**: `data/video_chat.db`
  - Videos metadata
  - Frames with timestamps
  - Chat history
  - Frame analysis results

- **Qdrant Collection**: `video_frames`
  - Frame description embeddings
  - Semantic search index
  - Excitement scores

- **File System**:
  - `data/videos/` - Uploaded videos
  - `data/frames/video_{id}/` - Extracted frames
  - `data/thumbnails/` - Timeline strips

---

## 🔧 API Endpoints

### Upload & Video Management

```bash
# Upload video
POST /api/upload
Content-Type: multipart/form-data
Body: file=<video_file>

# List all videos
GET /api/videos

# Get video info
GET /api/videos/{video_id}

# Stream video
GET /api/videos/{video_id}/stream

# Get thumbnail strip
GET /api/videos/{video_id}/thumbnail

# Get frames
GET /api/videos/{video_id}/frames?keyframes_only=true
```

### Chat

```bash
# Chat with video
POST /api/chat
Content-Type: application/json
Body: {
  "video_id": 1,
  "query": "What happens at 1:23?"
}
```

### System

```bash
# Get statistics
GET /api/stats
```

---

## 💰 Cost Estimates

Using Claude Haiku + MixedBread AI embeddings:

### Per Video (10 min, analyzing 100 frames)

- **Frame Analysis**: ~$0.04 (100 images × 1568 tokens)
- **Embeddings**: ~$0.01 (100 descriptions)
- **Chat Queries**: ~$0.01 per query
- **Total Processing**: **~$0.05 per video**

### Optimizations

- Extract frames every 2s (vs every frame = much cheaper)
- Analyze first 100 frames only (configurable)
- Batch API calls (100 images at once)
- Keyframes at scene changes only

---

## 🎛️ Configuration

Edit `video_chat_server.py` to customize:

```python
# Frame extraction interval (seconds)
interval_seconds=2.0  # Extract frame every 2 seconds

# Maximum frames to analyze
max_frames=500  # Up to 500 frames (16 min @ 2s interval)

# Frames to analyze with AI
frame_paths[:100]  # Analyze first 100 frames

# Excitement threshold for highlights
threshold=7  # Minimum score 7/10
```

---

## 🧪 Testing

### Test Video Processing

```bash
# Test frame extraction
python -c "
from src.services.video_processor import VideoProcessor
vp = VideoProcessor()
info = vp.get_video_info('assets/big_buck_bunny_1080p_30fps.mp4')
print(info)
"
```

### Test AI Analysis

```bash
# Analyze sample frames
python src/tools/video_analyzer.py data/frames/video_1/frame_000001.jpg
```

### Test Search

```bash
# Python test script
python -c "
from src.services.video_embeddings import VideoEmbeddingService
service = VideoEmbeddingService()
results = service.search_frames('explosion', video_id=1, limit=5)
print(f'Found {len(results)} matching frames')
"
```

---

## 🐛 Troubleshooting

### "ffmpeg not found"

**Solution**: Install ffmpeg (see Quick Start step 1)

```bash
# Verify installation
ffmpeg -version
```

### "Qdrant connection failed"

**Solution**: Start Qdrant server

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### "Video processing stuck"

**Check logs** in console. Common issues:
- Video codec not supported (try MP4/H.264)
- File corrupted
- Out of disk space

**Workaround**: Re-upload video or try different format

### "Claude API error"

**Check**:
- `ANTHROPIC_API_KEY` in `.env` is valid
- You have API credits
- Rate limits not exceeded

---

## 🔮 Future Enhancements

### Planned Features (from TODO)

- [ ] **Audio Analysis**: Speech-to-text, music detection
- [ ] **Multi-video Chat**: Compare across videos
- [ ] **Export Highlights**: Generate highlight reels automatically
- [ ] **Collaborative Annotations**: Share video insights
- [ ] **Advanced Filters**: Filter by scene type, objects, actions
- [ ] **Real-time Processing**: Stream analysis as video uploads

### Integration Ideas

- **VideoLLaMA3**: Add local video understanding (if you have GPU)
- **Whisper**: Transcribe audio for better search
- **BM25 Hybrid Search**: Combine semantic + keyword search
- **MCP Protocol**: Standardized tool interface

---

## 📊 Performance

### Benchmarks (MacBook Pro M1, 10min video)

| Task | Time | Cost |
|------|------|------|
| Upload | 30s | Free |
| Frame Extraction | 45s | Free |
| Scene Detection | 60s | Free |
| AI Analysis (100 frames) | 90s | $0.04 |
| Embedding & Indexing | 20s | $0.01 |
| **Total Processing** | **~4 min** | **$0.05** |
| Chat Query | 2-5s | $0.01 |

---

## 🤝 Contributing

Found a bug? Have ideas? PRs welcome!

**Key Areas for Improvement:**
- Better scene detection algorithms
- Optimize API costs
- Mobile-responsive UI
- More chat query types

---

## 📄 License

Part of the AI Video Database project.

---

## 🙏 Acknowledgments

Built with:
- **FastAPI** - Web framework
- **Claude Vision** - Video understanding
- **Qdrant** - Vector search
- **FFmpeg** - Video processing
- **MixedBread AI** - Embeddings

---

## 📞 Support

**Issues**: https://github.com/diffusionstudio/agent/issues

**Questions**: Ask in the video chat system itself! 😉

---

**Made with ❤️ by the Diffusion Studio team**
