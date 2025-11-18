"""Test: Simple 5 second clip to verify the pattern"""
from src.client import DiffusionClient
from src.tools import VideoEditorTool, VisualFeedbackTool
from loguru import logger

client = DiffusionClient()
video_tool = VideoEditorTool(client=client)
feedback_tool = VisualFeedbackTool(client=client)

logger.info("🧪 TEST: Creating simple 5 second clip...")

# Step 1: Create composition with sample
result = video_tool.forward(
    assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
    javascript="""
    const [videoFile] = assets();

    const video = new core.VideoClip(videoFile, {
        position: 'center',
        height: '100%'
    }).subclip(60, 210);  // 2-7 seconds of source video

    const text = new core.TextClip({
        text: 'TEST VIDEO',
        position: 'center',
        fontSize: 100,
        fillColor: '#00FF00',
        stroke: { color: '#000000', width: 4 }
    });
    text.duration = 150;

    await composition.add(video);
    await composition.add(text);
    await sample();
    """,
    output="output/test_simple.mp4"
)
logger.info(f"Step 1 result: {result}")

# Step 2: Get feedback
feedback = feedback_tool.forward(
    final_goal="5 second video with green 'TEST VIDEO' text"
)
logger.info(f"Step 2 feedback: {feedback}")

# Step 3: Render
if "Render decision: True" in feedback:
    result = video_tool.forward(
        assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
        javascript="await render();",
        output="output/test_simple.mp4"
    )
    logger.info(f"✅ TEST PASSED - Video created")
else:
    logger.error("❌ TEST FAILED")

client.close()
