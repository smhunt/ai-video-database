# 3TB Video Archive - Master Indexing Plan

**Goal**: Index, embed, and make searchable 10 years of personal video content (3TB) running entirely on your Mac.

**Vision**: "Show me all videos where we went to the beach" → Instant results from a decade of memories.

---

## 📊 The Challenge

### What We're Working With
- **Total Size**: 3TB of video
- **Timespan**: 10 years of content
- **Content**: Personal videos, family events, trips, daily life
- **Hardware**: Mac M1/M2 with 644GB available disk space
- **Network**: Local processing (no cloud dependency)

### Estimated Content
Assuming average video specs:
- **~3,000 hours** of video (rough estimate)
- **~6-10 million frames** at 1 frame/2 seconds
- **~500-1000 videos** (varies by length)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  3TB VIDEO ARCHIVE                       │
│              (External Drive Recommended)                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             VIDEO INTAKE & CATALOGING                    │
│  • Scan all video files                                  │
│  • Extract metadata (date, duration, resolution)         │
│  • Generate unique IDs                                   │
│  • Priority scoring (newest/favorites first)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           SMART FRAME EXTRACTION                         │
│  • Adaptive sampling (2-10 sec intervals)                │
│  • Scene detection for keyframes                         │
│  • Motion analysis (skip static frames)                  │
│  • Deduplication (skip similar frames)                   │
│  • Store only thumbnails (not full frames)               │
└────────────────────┬────────────────────────────────────┘
                     │
           ┌─────────┴─────────┐
           │                   │
           ▼                   ▼
┌──────────────────┐  ┌──────────────────────┐
│ TIER 1: FAST     │  │ TIER 2: DEEP         │
│ (Strategic)      │  │ (Comprehensive)      │
│                  │  │                      │
│ LLaVA 13B        │  │ LLaVA 34B / Qwen-VL  │
│ • Key events     │  │ • All frames         │
│ • Important      │  │ • High detail        │
│   videos         │  │ • Background queue   │
│ • ~100k frames   │  │ • ~6M frames         │
└──────────────────┘  └──────────────────────┘
           │                   │
           └─────────┬─────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│          LOCAL EMBEDDING GENERATION                      │
│  • Sentence Transformers (all-MiniLM-L6-v2)             │
│  • Or: BGE-large (better quality)                        │
│  • Run on Mac GPU                                        │
│  • ~1000 embeddings/second                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              VECTOR DATABASE                             │
│  Option 1: Qdrant (current, good for millions)          │
│  Option 2: FAISS (faster, file-based)                   │
│  Option 3: ChromaDB (simpler, embedded)                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          METADATA DATABASE (SQLite/PostgreSQL)           │
│  • 10M+ frame records                                    │
│  • Video metadata                                        │
│  • Timeline indexes                                      │
│  • Search cache                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 💾 Storage Strategy

### Current Limitations
- **Mac Internal**: 644GB available
- **3TB Videos**: Need external storage
- **Processed Data**: Will generate significant data

### Storage Requirements Breakdown

| Component | Size per Frame | Total (6M frames) | Notes |
|-----------|----------------|-------------------|-------|
| **Thumbnails** (320x180 JPEG) | ~15 KB | ~90 GB | Stored locally |
| **Embeddings** (768 dims) | ~3 KB | ~18 GB | Vector DB |
| **Descriptions** (text) | ~0.5 KB | ~3 GB | SQLite |
| **Metadata** | ~0.2 KB | ~1.2 GB | SQLite |
| **Indexes** | - | ~5 GB | DB indexes |
| **Models** (LLaVA, embeddings) | - | ~15 GB | Ollama cache |
| **Total Processed** | | **~132 GB** | ✅ Fits! |

### Recommended Setup
```
📁 /Users/seanhunt/VideoArchive/
├── processed/
│   ├── thumbnails/     (90 GB)
│   ├── database/       (25 GB)
│   └── vectors/        (18 GB)
├── models/             (15 GB)  # Ollama models
└── cache/              (10 GB)  # Temp processing

📁 /Volumes/External/    (3TB videos - read-only)
└── raw_videos/
```

---

## ⚡ Processing Strategy

### Phase 1: Catalog & Prioritize (1-2 hours)
```python
# Scan all videos, extract metadata
for video in find_all_videos('/Volumes/External/'):
    - Get duration, resolution, date
    - Calculate priority score
    - Add to processing queue

Priority Rules:
1. Newest videos first (recent memories)
2. Special events (holidays, birthdays) - detect from filename/date
3. Longer videos (likely important events)
4. High resolution (4K > 1080p > 720p)
```

### Phase 2: Smart Frame Extraction
```python
# Adaptive sampling based on video characteristics

Short videos (<5 min):     1 frame/second   → High detail
Medium videos (5-30 min):  1 frame/2 sec    → Balanced
Long videos (>30 min):     1 frame/5 sec    → Efficient
Static videos (low motion): 1 frame/10 sec  → Skip redundant

+ Scene change keyframes (always capture)
+ First/last 30 seconds (context)

Result: ~6M frames instead of 10M+ (40% reduction)
```

### Phase 3: Deduplication & Quality Filter
```python
# Skip similar/blurry frames
- Perceptual hashing (pHash)
- Motion detection (skip if <5% change)
- Blur detection (skip unfocused frames)
- Face detection (prioritize frames with people)

Result: ~4M high-quality, unique frames (33% further reduction)
```

### Phase 4: Tiered Analysis

**Tier 1 - Quick Pass (Week 1)**
- Analyze ~100k most important frames
- LLaVA 13B @ 5 sec/frame
- 100k × 5 sec = 139 hours = **6 days continuous**
- Focus: Events, special moments, people

**Tier 2 - Deep Enrichment (Ongoing)**
- Background processing remaining 3.9M frames
- LLaVA 34B @ 10 sec/frame
- 3.9M × 10 sec = 10,833 hours = **452 days**
- BUT: Run 24/7 in background, done in ~1.5 years
- OR: Use faster models (LLaVA 13B) = **225 days**

**Optimization**: Parallel processing if multiple Macs available

---

## 🚀 Incremental Processing Plan

### Week 1: Foundation
1. **Catalog** - Scan all 3TB (2 hours)
2. **Extract** - Pull frames from top 100 videos (20 hours)
3. **Analyze** - LLaVA on 10k frames (14 hours)
4. **Search** - Test search on first batch

**Result**: Searchable sample of your archive

### Month 1: Core Content
- Process 1,000 most important videos
- ~500k frames analyzed
- All major events indexed
- Search works across key memories

### Month 3: Bulk Processing
- Continuous background processing
- Process 5k-10k frames/day
- ~1M frames completed
- System learns your content patterns

### Year 1: Complete Archive
- All 4M frames processed
- Full decade searchable
- Continuous updates for new content

---

## 🔍 Search Capabilities

### Query Examples
```
"beach vacations" → All beach videos across 10 years
"birthdays with grandma" → Family celebrations
"hiking in mountains" → Outdoor adventures
"christmas mornings" → Holiday memories
"first steps" → Baby milestones
"graduation" → Life events
```

### Advanced Features
1. **Timeline View**: Visual timeline of your life
2. **Face Clustering**: Group by people (optional)
3. **Location Search**: If GPS metadata available
4. **Date Range**: "summer 2018" queries
5. **Semantic**: "happy moments" "celebrations" "emotional"

---

## 🛠️ Technology Stack

### Local Vision Models
**Primary**: LLaVA 13B (Ollama)
- Good quality/speed balance
- 7GB model size
- ~5 sec/frame on M1/M2

**Optional**: LLaVA 34B or Qwen-VL
- Better quality for important content
- 20GB model size
- ~10 sec/frame

### Local Embedding Models
**Recommended**: BGE-Large-EN-v1.5
```bash
pip install sentence-transformers
```
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
embeddings = model.encode(descriptions)
```

**Specs**:
- 335M parameters
- 1024-dim embeddings
- ~1000 embeddings/second on Mac
- Better quality than OpenAI for semantic search

**Alternative**: all-MiniLM-L6-v2
- Smaller, faster (384-dim)
- ~5000 embeddings/second
- Good enough for most searches

### Vector Database
**Current**: Qdrant
- Handles millions of vectors
- Good performance
- Memory efficient

**Alternative**: FAISS (Facebook AI)
- Faster search
- Better for huge datasets
- File-based (no Docker needed)

### Metadata Database
**Current**: SQLite
- Good for millions of records
- Single file
- No setup

**Upgrade Option**: PostgreSQL
- Better for 10M+ records
- Full-text search
- Better concurrency

---

## 💰 Cost Analysis

### Hardware Costs
- **Current Mac**: $0 (you have it)
- **External 4TB SSD**: ~$200 (for processed data backup)
- **Total**: $200

### Operational Costs
- **Electricity**: ~$20/month @ 24/7 processing
- **API Costs**: $0 (all local)
- **Total Year 1**: ~$240

### Cloud Comparison
If using Claude for 4M frames:
- 4M frames × $0.049/50 frames = **$3,920**
- Plus OpenAI embeddings: **~$520**
- **Total**: $4,440

**Savings with local**: $4,200+ 🎉

---

## 📈 Performance Optimizations

### 1. Parallel Processing
```python
# Use all CPU cores
import multiprocessing
pool = multiprocessing.Pool(processes=8)

# Process 8 frames simultaneously
# 8× speedup on M2 Pro
```

### 2. GPU Acceleration
```python
# Use Metal Performance Shaders (MPS) on Mac
import torch
device = torch.device("mps")

# 2-3× faster inference
```

### 3. Batch Processing
```python
# Process frames in batches of 10
# Better GPU utilization
# 30% faster overall
```

### 4. Smart Caching
```python
# Cache model outputs
# Skip reprocessing on restarts
# Resume from last frame
```

### 5. Priority Queue
```python
# Process important videos first
# User sees results faster
# Can search while processing continues
```

---

## 🗺️ Implementation Roadmap

### Phase 0: Proof of Concept (This Week)
- [ ] Install LLaVA 13B
- [ ] Process 1 video end-to-end
- [ ] Test search quality
- [ ] Verify performance metrics

### Phase 1: MVP (Week 1-2)
- [ ] Build video cataloger
- [ ] Implement smart frame extraction
- [ ] Create local vision analyzer
- [ ] Setup background worker
- [ ] Process top 10 videos
- [ ] Test search on sample

### Phase 2: Scale Up (Week 3-4)
- [ ] Add deduplication
- [ ] Implement priority queue
- [ ] Setup local embeddings
- [ ] Process 100 videos
- [ ] UI improvements for large library

### Phase 3: Production (Month 2-3)
- [ ] Optimize for 24/7 processing
- [ ] Add progress monitoring
- [ ] Implement pause/resume
- [ ] Setup backup strategy
- [ ] Process 1000+ videos

### Phase 4: Complete Archive (Month 4-12)
- [ ] Continuous background processing
- [ ] Monitor and optimize
- [ ] Handle edge cases
- [ ] Complete all 3TB

---

## 🎯 Quick Start (Next Session)

```bash
# 1. Pull LLaVA model (7GB download)
ollama pull llava:13b

# 2. Install local embeddings
pip install sentence-transformers faiss-cpu

# 3. Test on one video
python3 test_local_analysis.py --video /path/to/wedding.mp4

# 4. If good → Start building full system
```

---

## 🤔 Key Decisions Needed

1. **Processing Speed vs Quality**
   - Fast (LLaVA 13B): 6 months to complete
   - Quality (LLaVA 34B): 1.5 years to complete
   - Hybrid: Fast for bulk, quality for key videos

2. **Storage Location**
   - All on Mac (need to free space)
   - External SSD (recommended)
   - NAS (if available)

3. **Processing Schedule**
   - 24/7 continuous (fastest)
   - Nights only (quieter)
   - When not using Mac (balanced)

4. **Content Priority**
   - Newest first (recent memories)
   - Chronological (tell story from start)
   - Smart (detect events, prioritize)

---

## 🎉 The Vision

Imagine typing:
- **"beach trips"** → See every beach vacation, organized by year
- **"family dinners with dad"** → Every moment together
- **"my 20s"** → Visual timeline of a decade
- **"learning to ski"** → Watch your progression
- **"road trips"** → All your adventures

**Your life, completely searchable. Every moment findable. 10 years in one database.**

This is totally achievable! Let's build it! 🚀
