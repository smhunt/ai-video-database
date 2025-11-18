#!/bin/bash

# Video Chat Server Startup Script

echo "🎬 Starting Video Chat Server..."
echo ""

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ Error: ffmpeg not found!"
    echo "Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)"
    exit 1
fi

# Check for API keys
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: ANTHROPIC_API_KEY not set in environment"
    echo "Checking .env file..."
    if [ -f .env ]; then
        # Load .env more carefully (handle spaces and special chars)
        set -a
        source .env
        set +a
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "❌ Error: ANTHROPIC_API_KEY not found in .env"
            echo "Please add it to your .env file"
            exit 1
        fi
    else
        echo "❌ Error: .env file not found"
        exit 1
    fi
fi

# Check if Qdrant is running
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
echo "Checking Qdrant at $QDRANT_URL..."
if curl -s "$QDRANT_URL/health" > /dev/null 2>&1; then
    echo "✅ Qdrant is running"
else
    echo "⚠️  Qdrant not detected at $QDRANT_URL"
    echo "Starting Qdrant with Docker..."
    if command -v docker &> /dev/null; then
        docker run -d -p 6333:6333 --name qdrant-video-chat qdrant/qdrant
        echo "⏳ Waiting for Qdrant to start..."
        sleep 3
        if curl -s "$QDRANT_URL/health" > /dev/null 2>&1; then
            echo "✅ Qdrant started successfully"
        else
            echo "❌ Failed to start Qdrant. Please start it manually:"
            echo "   docker run -p 6333:6333 qdrant/qdrant"
            exit 1
        fi
    else
        echo "❌ Docker not found. Please start Qdrant manually:"
        echo "   docker run -p 6333:6333 qdrant/qdrant"
        exit 1
    fi
fi

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python not found"
    exit 1
fi

# Check Python dependencies
echo "Checking Python dependencies..."
$PYTHON_CMD -c "import fastapi, uvicorn, anthropic, instructor, qdrant_client" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing dependencies. Installing..."
    pip3 install -r requirements_video_chat.txt || pip install -r requirements_video_chat.txt
fi

echo ""
echo "✅ All checks passed!"
echo ""
echo "🚀 Starting server..."
echo "   Web UI: http://localhost:8000"
echo "   API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
$PYTHON_CMD video_chat_server.py
