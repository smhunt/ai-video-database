"""Test: No feedback - just compose and render"""
from src.client import DiffusionClient
from src.tools import VideoEditorTool
from loguru import logger

client = DiffusionClient()
video_tool = VideoEditorTool(client=client)

logger.info("🧪 TEST: No feedback version...")

# Step 1: Create and sample
result = video_tool.forward(
    assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
    javascript="""
    const [videoFile] = assets();

    const video = new core.VideoClip(videoFile, {
        position: 'center',
        height: '100%'
    }).subclip(90, 240);  // 3-8 seconds = 5 second clip

    const text = new core.TextClip({
        text: 'NO FEEDBACK TEST',
        position: 'center',
        fontSize: 80,
        fillColor: '#FF00FF',
        stroke: { color: '#FFFFFF', width: 3 }
    });
    text.duration = 150;

    await composition.add(video);
    await composition.add(text);
    await sample();
    """,
    output="output/test_no_feedback.mp4"
)
logger.info(f"Composition created: {result}")

# Step 2: Render immediately (skip feedback)
result = video_tool.forward(
    assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
    javascript="await render();",
    output="output/test_no_feedback.mp4"
)
logger.info(f"✅ Rendered: {result}")

client.close()
