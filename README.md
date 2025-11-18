<p align="center">
  <h1 align="center">🎬 AI Video Database</h1>
  <p align="center"><i>Formerly known as "AI Video Composer" - because composing videos was too mainstream. We now organize, search, and chat with them instead. Much cooler. 😎</i></p>
</p>

<p align="center">
  <strong>Talk to your videos using AI. Upload, analyze, and search through video content with natural language.</strong>
</p>

<br/>

## 🙏 Credits

Built on top of the excellent [Diffusion Studio Video Composer Agent](https://github.com/diffusionstudio/agent) - we took their video editing foundation and went wild with AI-powered video search and chat. Thanks for the head start! 🚀

<br/>

## 🚀 What This Does

Upload your videos and **talk to them** like they're your best friend who remembers everything. The AI analyzes your videos, creates searchable transcripts, and lets you ask questions in plain English.

### Key Features

- 📹 **Video Upload**: Drag & drop videos up to 2GB
- 🤖 **AI Analysis**: Claude Vision analyzes every frame
- 🎙️ **Audio Transcription**: Whisper converts speech to searchable text
- 🔍 **Semantic Search**: Find scenes by describing what you want
- 💬 **Video Chat**: Ask questions about your videos in natural language
- ⚡ **Highlight Detection**: Automatically identifies exciting moments
- 📊 **Cost Tracking**: Monitor AI API usage and costs

## 📖 Documentation

- **[Video Chat System Guide](VIDEO_CHAT_README.md)** - Complete guide to the video chat features
- **[Quick Start Guide](VIDEO_CHAT_QUICK_START.md)** - Get up and running in 5 minutes
- **[Transcription Feature](TRANSCRIPTION_FEATURE.md)** - Details on audio transcription

## ⚡ Quick Start

### 1. Install Dependencies

```bash
# Install FFmpeg (required for video processing)
brew install ffmpeg  # macOS
# sudo apt-get install ffmpeg  # Ubuntu/Debian

# Install Python dependencies
pip install -r requirements_video_chat.txt
```

### 2. Set Up Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Required API keys:
- `ANTHROPIC_API_KEY` - For Claude Vision analysis
- `OPENAI_API_KEY` - For embeddings and Whisper transcription
- `QDRANT_URL` - Vector database URL (default: http://localhost:6333)

### 3. Start Qdrant (Vector Database)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Run the Server

```bash
python video_chat_server.py
```

Open http://localhost:8000 and start chatting with your videos!

## 🎯 Use Cases

- **Personal Memories**: Search through your wedding video: "Show me when we said our vows" or "Find the cake cutting moment"
- **Video Archives**: Make your life's precious moments searchable
- **Content Creation**: Find specific moments in long recordings
- **Research**: Analyze interview footage and find key quotes
- **Family Videos**: "When did grandma tell that story about dad?"

## 📁 Project Structure

```
├── video_chat_server.py       # Main FastAPI server
├── static/                     # Web UI (HTML/JS)
├── src/
│   ├── models/                 # Database models
│   ├── services/               # Video processing, embeddings, transcription
│   ├── tools/                  # AI analysis tools
│   └── utils/                  # Cost calculation, etc.
├── data/                       # Videos, frames, database (gitignored)
└── archive/                    # Original video editing examples
```

## 🎬 About the Video Editing Features

This project was forked from the excellent [Diffusion Studio Video Composer Agent](https://github.com/diffusionstudio/agent), which focuses on AI-powered video *composition* and *editing*.

We kept their foundation but pivoted to video *database* and *search* capabilities. If you're interested in the original video editing features, check out the `archive/` folder or visit their repo!

## 🛠️ Development

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and guidelines.

## 📋 Roadmap

- [ ] Multi-video search across entire library
- [ ] Export highlight reels automatically
- [ ] Real-time processing as videos upload
- [ ] Advanced filters (scene type, objects, actions)
- [ ] Collaborative video annotations
- [ ] BM25 hybrid search integration
- [ ] Support for video understanding models like VideoLLaMA
