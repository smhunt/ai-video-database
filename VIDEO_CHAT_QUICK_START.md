# 🚀 Video Chat System - Quick Start Guide

You now have a fully functional **local video chat system** that lets you upload videos (up to 2GB) and have intelligent conversations about their content!

---

## ✨ What Was Built

### Complete System Components

1. **Backend Server** (`video_chat_server.py`)
   - FastAPI web server with 15+ endpoints
   - Video upload with streaming support
   - Background processing pipeline
   - Chat API with intelligent routing

2. **Video Processing** (`src/services/video_processor.py`)
   - Frame extraction at configurable intervals
   - Scene detection and keyframe identification
   - Thumbnail strip generation
   - FFmpeg integration

3. **AI Analysis** (`src/tools/video_analyzer.py`)
   - Claude Vision batch analysis (100 frames at once)
   - Structured outputs with Pydantic
   - Excitement scoring (1-10 for each frame)
   - Single frame Q&A capability

4. **Semantic Search** (`src/services/video_embeddings.py`)
   - Qdrant vector database integration
   - MixedBread AI embeddings (1024-dim)
   - Natural language queries like "Show me the first dance"
   - Excitement threshold filtering for highlights

5. **Database** (`src/models/database.py`)
   - SQLite with comprehensive schema
   - Videos, frames, embeddings, chat history
   - Efficient indexes for fast queries

6. **Web UI** (`static/index.html` + `app.js`)
   - Drag & drop video upload
   - Real-time processing status
   - Video player with timeline
   - Chat interface with frame previews
   - Clickable keyframes for seeking

---

## 🎯 How To Start (2 Minutes)

### Option 1: Easy Start (Recommended)

```bash
./start_video_chat.sh
```

This script automatically:
- ✅ Checks all dependencies
- ✅ Starts Qdrant if needed
- ✅ Verifies API keys
- ✅ Launches the server

### Option 2: Manual Start

```bash
# 1. Start Qdrant (in separate terminal)
docker run -p 6333:6333 qdrant/qdrant

# 2. Install dependencies (if not already done)
pip install -r requirements_video_chat.txt

# 3. Start server
python video_chat_server.py
```

### Then Open Your Browser

**🌐 http://localhost:8000**

---

## 📹 Using the System

### 1. Upload a Video

- Drag & drop a video file (MP4, MOV, AVI, etc.)
- Or click the upload area to select
- Max size: 2GB
- Wait 2-5 minutes for processing

### 2. Watch the Processing

Status will update automatically:
- **Processing** → Extracting frames
- **Analyzing** → Claude Vision analyzing content
- **Indexing** → Creating searchable embeddings
- **Ready** ✅ → Start chatting!

### 3. Chat with Your Video

**Example Queries:**

```
"Show me when we walked down the aisle"
"What happens at 1:23?"
"Find the first kiss"
"When did the best man give his speech?"
"Find all the dancing moments"
"Show me when grandma arrived"
```

The system automatically:
- Identifies query type (timestamp, search, general)
- Searches semantically through frame descriptions
- Returns relevant frames with timestamps
- Links directly to video moments (click to seek)

---

## 💡 Cool Features

### Automatic Scene Detection

The system identifies scene changes and extracts keyframes automatically. Click any keyframe thumbnail to jump to that moment!

### Excitement Scoring

Every frame gets a 1-10 "excitement score" from Claude Vision. Use this to find highlights:

```
"Find exciting moments"  → Returns frames with score ≥7
```

### Visual Timeline

Thumbnail strip shows 10 evenly-spaced frames for quick video overview. Generated automatically on upload.

### Frame Previews in Chat

Responses include clickable frame thumbnails - click to seek the video player to that exact moment.

### Semantic Search

Not just keyword matching - understands concepts:

```
"Show the ceremony"     → Matches: "vows", "rings", "altar", "officiant"
"Find emotional moments" → Matches: "tears", "hugging", "smiling", "laughter"
"Show outdoor scenes"    → Matches: "garden", "sky", "venue exterior"
```

---

## 📊 System Architecture

```
┌─────────────────┐
│  Web Browser    │  Upload video, chat interface
└────────┬────────┘
         │ HTTP
┌────────▼────────┐
│  FastAPI Server │  Routes, business logic
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼──┐ ┌───▼───┐ ┌──▼───┐ ┌─▼────┐
│SQLite│ │FFmpeg │ │Claude│ │Qdrant│
│  DB  │ │Process│ │Vision│ │Vector│
└──────┘ └───────┘ └──────┘ └──────┘
```

### Data Flow

1. **Upload** → Saved to `data/videos/`
2. **Extract** → Frames to `data/frames/video_{id}/`
3. **Analyze** → Claude Vision processes 100 frames
4. **Embed** → Descriptions → 1024-dim vectors
5. **Index** → Qdrant stores embeddings
6. **Chat** → Semantic search + Claude answers

---

## 💰 Cost Estimate

**Per 10-minute video:**

| Operation | Cost |
|-----------|------|
| Frame extraction (200 frames @ 2s) | Free (local) |
| Claude Vision analysis (100 frames) | ~$0.04 |
| Embeddings (100 descriptions) | ~$0.01 |
| **Total Processing** | **~$0.05** |
| Each chat query | ~$0.01 |

**Optimizations in place:**
- Only analyze first 100 frames (configurable)
- Batch API calls (100 images at once)
- Efficient frame sampling (every 2s, not every frame)
- Local processing (FFmpeg, Qdrant)

---

## 🔧 Configuration

Edit `video_chat_server.py` to customize:

```python
# Line 68: Frame extraction interval
interval_seconds=2.0  # Change to 1.0 for more frames, 5.0 for fewer

# Line 69: Max frames
max_frames=500  # Increase for longer videos

# Line 91: Frames to analyze with AI
frame_paths[:100]  # Change to [:200] for more analysis

# Line 102: Excitement threshold
excitement_threshold=7  # Lower for more highlights
```

---

## 🗂️ File Structure

```
video_chat_server.py              # Main server (entry point)
start_video_chat.sh               # Easy startup script
VIDEO_CHAT_README.md              # Full documentation
VIDEO_CHAT_QUICK_START.md         # This file

src/
├── models/
│   └── database.py               # SQLite models & queries
├── services/
│   ├── video_processor.py        # FFmpeg operations
│   └── video_embeddings.py       # Qdrant search
└── tools/
    └── video_analyzer.py         # Claude Vision analysis

static/
├── index.html                    # Web UI (HTML)
└── app.js                        # Frontend logic (JavaScript)

data/                             # Created on first run
├── video_chat.db                 # SQLite database
├── videos/                       # Uploaded videos
├── frames/                       # Extracted frames
└── thumbnails/                   # Timeline strips
```

---

## 🐛 Troubleshooting

### "Connection refused" on http://localhost:8000

**Check:** Is the server running?

```bash
python video_chat_server.py
```

### "Qdrant connection failed"

**Start Qdrant:**

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Verify it's running:**

```bash
curl http://localhost:6333/health
```

### "Video stuck processing"

**Check server logs** in terminal. Common issues:
- Unsupported codec (try converting to MP4/H.264)
- Out of disk space
- FFmpeg not installed

**Fix:** Re-upload or try different video format

### "Chat not working"

**Check:**
1. Video status is "Ready" ✅ (not "Processing")
2. `ANTHROPIC_API_KEY` is set in `.env`
3. Server logs for error messages

---

## 📚 Learn More

- **Full Documentation**: `VIDEO_CHAT_README.md`
- **API Docs**: http://localhost:8000/docs (when server running)
- **System Stats**: http://localhost:8000/api/stats

---

## 🎉 Next Steps

1. **Upload your first video** - Try it out!
2. **Experiment with queries** - See what works best
3. **Check the costs** - Monitor API usage on Anthropic dashboard
4. **Customize settings** - Adjust frame extraction, thresholds
5. **Build features** - Add your own query types!

---

## 🤝 Need Help?

- **Issue?** Check `VIDEO_CHAT_README.md` troubleshooting section
- **Question?** Ask in the video chat system itself! 😉
- **Bug?** Open issue on GitHub

---

**Made with ❤️ for local video understanding**

Happy chatting! 🎬✨
