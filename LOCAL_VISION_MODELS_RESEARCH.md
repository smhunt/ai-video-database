# Local Vision Model Research for Video Enrichment

**Goal**: Run local LLM with vision capabilities to analyze ALL video frames asynchronously, enriching the database with detailed descriptions without API costs.

---

## 🎯 Why Local Models?

### Advantages
- ✅ **Zero API Costs** - No per-frame charges
- ✅ **Analyze ALL Frames** - Not limited to 50 frames like Claude
- ✅ **Privacy** - Data never leaves your machine
- ✅ **Async Enrichment** - Background processing doesn't block uploads
- ✅ **Super Fine Detail** - Can analyze every single frame

### Use Cases
1. **Initial Fast Pass** - Claude analyzes 50 keyframes (quick, decent quality)
2. **Deep Enrichment** - Local model analyzes ALL 1000+ frames overnight
3. **Wedding Video** - Find every single moment, every person, every detail
4. **Search Enhancement** - More detailed descriptions = better search results

---

## 🤖 Best Local Vision Models

### 1. **LLaVA (Best Overall)**
- **Model**: LLaVA-v1.6-34B (or 13B for faster)
- **Capabilities**: Image understanding, detailed descriptions
- **Performance**: ~5-10 seconds per frame on M1/M2 Mac
- **Quality**: Very good, comparable to GPT-4V on many tasks
- **Framework**: Ollama or HuggingFace Transformers

**Installation**:
```bash
# Using Ollama (easiest)
ollama pull llava:13b
ollama pull llava:34b  # Better quality, slower
```

**Pros**:
- Excellent description quality
- Fast with Ollama
- Easy to setup
- Good for wedding videos (recognizes people, events, emotions)

**Cons**:
- Memory intensive (16GB+ RAM recommended)
- Slower than API calls

---

### 2. **BakLLaVA (Faster Alternative)**
- **Model**: BakLLaVA-1
- **Capabilities**: Similar to LLaVA but optimized
- **Performance**: ~3-5 seconds per frame
- **Quality**: Slightly lower than LLaVA but still good

**Installation**:
```bash
ollama pull bakllava
```

---

### 3. **Video-LLaMA (Already in Roadmap!)**
- **Model**: Video-LLaMA-2
- **Capabilities**: **Understands temporal context across frames**
- **Performance**: Slower but can analyze frame sequences
- **Quality**: Best for understanding video flow (not just individual frames)

**Special Feature**: Can understand:
- "The bride walks down the aisle" (motion/temporal)
- "People are dancing together" (group dynamics)
- "The ceremony transitions to reception" (event flow)

**Note**: More complex setup, requires specific environment

---

### 4. **MiniGPT-4 (Lightweight)**
- **Model**: MiniGPT-4
- **Capabilities**: Basic vision understanding
- **Performance**: Very fast (~1-2 seconds per frame)
- **Quality**: Lower than LLaVA but good for basic tagging

---

## 🏗️ Proposed Architecture

```
┌─────────────────────────────────────────────────┐
│                 Video Upload                     │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Extract ALL Frames (1000+)               │
└────────────────┬────────────────────────────────┘
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────────┐
│ FAST PASS    │    │ DEEP ENRICHMENT  │
│ (Claude)     │    │ (Local LLM)      │
│              │    │                  │
│ • 50 frames  │    │ • ALL frames     │
│ • Immediate  │    │ • Async/Queue    │
│ • $0.049     │    │ • $0.00          │
│ • Ready in   │    │ • Completes      │
│   2 minutes  │    │   overnight      │
└──────────────┘    └──────────────────┘
       │                   │
       └─────────┬─────────┘
                 ▼
       ┌──────────────────┐
       │ Enhanced Database │
       │ - Quick results   │
       │ - Deep results    │
       └──────────────────┘
```

---

## 📋 Implementation Plan

### Phase 1: Ollama + LLaVA Setup
```bash
# Install Ollama
brew install ollama

# Pull LLaVA model
ollama pull llava:13b

# Test it
ollama run llava:13b
```

### Phase 2: Create Local Vision Analyzer Service

**New File**: `src/services/local_vision_analyzer.py`

```python
class LocalVisionAnalyzer:
    """Async local vision model for deep frame analysis"""

    def __init__(self, model='llava:13b'):
        self.model = model
        self.queue = asyncio.Queue()

    async def analyze_frame_deep(self, frame_path):
        """Analyze single frame with local model"""
        # Call Ollama API
        # Return detailed description

    async def enrich_video(self, video_id):
        """Background task: analyze all frames"""
        # Get all frames for video
        # Process each frame
        # Update database with enriched descriptions

    async def worker(self):
        """Background worker processing queue"""
        # Continuously process enrichment tasks
```

### Phase 3: Add Enrichment Queue

**Database Changes**:
```sql
-- Add enrichment status to videos
ALTER TABLE videos ADD COLUMN enrichment_status TEXT DEFAULT 'pending';
-- Values: pending, enriching, enriched

-- Add local analysis to frames
ALTER TABLE frames ADD COLUMN local_description TEXT;
ALTER TABLE frames ADD COLUMN local_tags TEXT;
ALTER TABLE frames ADD COLUMN enriched_at TIMESTAMP;
```

### Phase 4: Background Worker

**New Script**: `enrichment_worker.py`

```python
# Runs separately from main server
# Polls database for videos needing enrichment
# Processes frames asynchronously
# Updates database as it goes
```

### Phase 5: UI Updates

**Status Panel Card 5**: "🧠 Enrichment Progress"
- Videos enriched: 2/5
- Frames analyzed: 1,234/5,000
- Current: wedding.mp4 (45% complete)

---

## 🎯 Recommended Approach

### **Start with: LLaVA 13B + Ollama**

**Why?**
1. Easiest setup (one command)
2. Good quality/speed balance
3. Works great on M1/M2 Macs
4. Can upgrade to 34B later for quality

**Performance Estimates** (M1/M2 Mac):
- 1000 frames × 5 seconds = ~1.4 hours
- Run overnight → wake up to enriched database
- 40-minute wedding video = ~2 hours enrichment time

---

## 💡 Advanced Features

### 1. **Hybrid Analysis**
- Claude (fast): Event type, excitement score
- Local (deep): Detailed descriptions, all objects, relationships

### 2. **Smart Enrichment**
- Only enrich frames user searches for
- Prioritize keyframes and highlights
- Background enrich rest over time

### 3. **Temporal Analysis** (Video-LLaMA)
- Understand frame sequences
- "Bride walks down aisle" (multiple frames)
- Better for video context vs static images

### 4. **Custom Fine-tuning**
- Fine-tune on your wedding videos
- Learns to recognize your family/friends
- Better descriptions over time

---

## 🚀 Next Steps

1. **Install Ollama** - `brew install ollama`
2. **Pull LLaVA** - `ollama pull llava:13b`
3. **Test locally** - Analyze a few frames manually
4. **Build service** - Create `LocalVisionAnalyzer` class
5. **Add queue** - Background enrichment worker
6. **Update UI** - Show enrichment progress

---

## 📊 Cost Comparison

**Current (Claude only)**:
- 50 frames @ $0.049 = Fast, good quality
- Total: **$0.049 per video**

**With Local Enrichment**:
- 50 frames @ $0.049 (Claude - immediate)
- 1000 frames @ $0.00 (Local - overnight)
- Total: **$0.049 + 0 electricity**

**Result**: 20x more analysis for same price! 🎉

---

## 🤔 Should We Build This?

**Vote**:
- ✅ **YES** - If you want super detailed search
- ⏸️ **LATER** - If current quality is good enough
- 🎯 **PARTIAL** - Start with manual testing, automate if useful

Let me know and I'll start building! This would be a **game-changer** for wedding video search.
