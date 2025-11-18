"""Simple direct execution - creates a 10 second video with 'Hello World' text"""
from src.client import DiffusionClient
from src.tools import VideoEditorTool, VisualFeedbackTool
from loguru import logger

# Initialize client
client = DiffusionClient()

# Create tools
video_tool = VideoEditorTool(client=client)
feedback_tool = VisualFeedbackTool(client=client)

logger.info("🎬 Starting video editing...")

# Task 1: Trim video to 10 seconds and add text 'Hello World'
logger.info("✂️  Task 1: Trimming video to 10 seconds and adding text")
result = video_tool.forward(
    assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
    javascript="""
    const [videoFile] = assets();

    // Create video clip trimmed to 10 seconds (300 frames at 30fps)
    const video = new core.VideoClip(videoFile, {
        position: 'center',
        height: '100%'
    }).subclip(0, 300);

    // Add text overlay centered
    const text = new core.TextClip({
        text: 'Hello World',
        position: 'center',
        fontSize: 80,
        fillColor: '#ffffff',
        stroke: {
            color: '#000000',
            width: 3
        }
    });
    text.duration = 300;

    // Add clips to composition
    await composition.add(video);
    await composition.add(text);

    // Generate preview samples
    await sample();
    """,
    output="output/video.mp4"
)
logger.info(f"✅ Result: {result}")

# Task 2: Visual feedback
logger.info("👁️  Task 2: Checking composition quality")
feedback = feedback_tool.forward(
    final_goal="Video should be 10 seconds long with 'Hello World' text centered over the video"
)
logger.info(f"Feedback: {feedback}")

# Task 3: Render if approved
if "Render decision: True" in feedback:
    logger.info("🎬 Task 3: Rendering final video...")
    result = video_tool.forward(
        assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
        javascript="await render();",
        output="output/video.mp4"
    )
    logger.info(f"✅ Render result: {result}")
    logger.info("🎉 SUCCESS! Video saved to output/video.mp4")
else:
    logger.warning("❌ Composition failed feedback check, not rendering")

logger.info("✨ All tasks completed!")
client.close()
