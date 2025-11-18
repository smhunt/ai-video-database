"""Create a Highlight Reel - combines multiple clips with text overlays"""
from src.client import DiffusionClient
from src.tools import VideoEditorTool, VisualFeedbackTool
from loguru import logger

client = DiffusionClient()
video_tool = VideoEditorTool(client=client)
feedback_tool = VisualFeedbackTool(client=client)

logger.info("🎬 Creating Highlight Reel...")

# Define highlight moments (in frames at 30fps)
# You can customize these timestamps to your actual highlights
highlights = [
    {"start": 0, "end": 90, "label": "Opening Scene"},      # 0-3 seconds
    {"start": 300, "end": 390, "label": "Action Moment"},   # 10-13 seconds
    {"start": 600, "end": 690, "label": "Epic Sequence"},   # 20-23 seconds
    {"start": 900, "end": 990, "label": "Grand Finale"},    # 30-33 seconds
]

# Build the JavaScript for creating highlight clips
result = video_tool.forward(
    assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
    javascript=f"""
    const [videoFile] = assets();

    // Highlight 1: Opening Scene (0-3 seconds)
    const clip1 = new core.VideoClip(videoFile, {{
        position: 'center',
        height: '100%'
    }}).subclip(0, 90);

    const text1 = new core.TextClip({{
        text: 'OPENING SCENE',
        position: {{ x: 'center', y: '10%' }},
        fontSize: 60,
        fontWeight: 'bold',
        fillColor: '#FF0000',
        stroke: {{ color: '#000000', width: 3 }}
    }});
    text1.duration = 90;
    text1.delay = 0;

    // Highlight 2: Action Moment (10-13 seconds) -> starts at 3s in timeline
    const clip2 = new core.VideoClip(videoFile, {{
        position: 'center',
        height: '100%'
    }}).subclip(300, 390);
    clip2.delay = 90;  // Start after first clip

    const text2 = new core.TextClip({{
        text: 'ACTION MOMENT',
        position: {{ x: 'center', y: '10%' }},
        fontSize: 60,
        fontWeight: 'bold',
        fillColor: '#FFD700',
        stroke: {{ color: '#000000', width: 3 }}
    }});
    text2.duration = 90;
    text2.delay = 90;

    // Highlight 3: Epic Sequence (20-23 seconds) -> starts at 6s in timeline
    const clip3 = new core.VideoClip(videoFile, {{
        position: 'center',
        height: '100%'
    }}).subclip(600, 690);
    clip3.delay = 180;

    const text3 = new core.TextClip({{
        text: 'EPIC SEQUENCE',
        position: {{ x: 'center', y: '10%' }},
        fontSize: 60,
        fontWeight: 'bold',
        fillColor: '#00FF00',
        stroke: {{ color: '#000000', width: 3 }}
    }});
    text3.duration = 90;
    text3.delay = 180;

    // Highlight 4: Grand Finale (30-33 seconds) -> starts at 9s in timeline
    const clip4 = new core.VideoClip(videoFile, {{
        position: 'center',
        height: '100%'
    }}).subclip(900, 990);
    clip4.delay = 270;

    const text4 = new core.TextClip({{
        text: 'GRAND FINALE',
        position: {{ x: 'center', y: '10%' }},
        fontSize: 60,
        fontWeight: 'bold',
        fillColor: '#FF00FF',
        stroke: {{ color: '#000000', width: 3 }}
    }});
    text4.duration = 90;
    text4.delay = 270;

    // Main title overlay
    const mainTitle = new core.TextClip({{
        text: 'HIGHLIGHT REEL',
        position: {{ x: 'center', y: '90%' }},
        fontSize: 40,
        fillColor: '#FFFFFF',
        fontWeight: 'bold',
        stroke: {{ color: '#000000', width: 2 }}
    }});
    mainTitle.duration = 360;  // Stays for entire video

    // Add all clips in order
    await composition.add(clip1);
    await composition.add(clip2);
    await composition.add(clip3);
    await composition.add(clip4);

    // Add text overlays
    await composition.add(text1);
    await composition.add(text2);
    await composition.add(text3);
    await composition.add(text4);
    await composition.add(mainTitle);

    await sample();
    """,
    output="output/highlight_reel.mp4"
)

logger.info(f"✅ Highlight reel composed: {result}")

# Get feedback
feedback = feedback_tool.forward(
    final_goal="Video should show 4 different highlight clips with colored labels and a main title"
)
logger.info(f"Feedback: {feedback}")

# Render
if "Render decision: True" in feedback:
    logger.info("🎬 Rendering highlight reel...")
    video_tool.forward(
        assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
        javascript="await render();",
        output="output/highlight_reel.mp4"
    )
    logger.info("🎉 Highlight reel complete! Saved to output/highlight_reel.mp4")
    logger.info("📊 Final video: 12 seconds (4 clips × 3 seconds each)")
else:
    logger.warning("❌ Feedback failed")

client.close()
