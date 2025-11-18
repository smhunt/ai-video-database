"""Edit 2: Split Screen Effect with Colored Overlays"""
from src.client import DiffusionClient
from src.tools import VideoEditorTool, VisualFeedbackTool
from loguru import logger

client = DiffusionClient()
video_tool = VideoEditorTool(client=client)
feedback_tool = VisualFeedbackTool(client=client)

logger.info("🎬 Creating Split Screen Video...")

# Create split screen with colored rectangles and text
result = video_tool.forward(
    assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
    javascript="""
    const [videoFile] = assets();

    // Left side video - 8 seconds
    const videoLeft = new core.VideoClip(videoFile, {
        position: { x: '25%', y: 'center' },
        width: '45%'
    }).subclip(0, 240);

    // Right side video (offset by 2 seconds) - 8 seconds
    const videoRight = new core.VideoClip(videoFile, {
        position: { x: '75%', y: 'center' },
        width: '45%'
    }).subclip(60, 300);

    // Colored divider in the middle
    const divider = new core.RectangleClip({
        width: 20,
        height: '100%',
        x: 'center',
        y: 'center',
        fillColor: '#FF6B6B'
    });
    divider.duration = 240;

    // Label for left side
    const leftLabel = new core.TextClip({
        text: 'SCENE 1',
        position: { x: '25%', y: '10%' },
        fontSize: 40,
        fillColor: '#00D9FF',
        fontWeight: 'bold',
        stroke: {
            color: '#000000',
            width: 2
        }
    });
    leftLabel.duration = 240;

    // Label for right side
    const rightLabel = new core.TextClip({
        text: 'SCENE 2',
        position: { x: '75%', y: '10%' },
        fontSize: 40,
        fillColor: '#FFD700',
        fontWeight: 'bold',
        stroke: {
            color: '#000000',
            width: 2
        }
    });
    rightLabel.duration = 240;

    // Add all clips
    await composition.add(videoLeft);
    await composition.add(videoRight);
    await composition.add(divider);
    await composition.add(leftLabel);
    await composition.add(rightLabel);

    await sample();
    """,
    output="output/edit2_split_screen.mp4"
)
logger.info(f"✅ Composition created: {result}")

# Get feedback
feedback = feedback_tool.forward(
    final_goal="Video should show two video clips side by side with labels and a colored divider"
)
logger.info(f"Feedback: {feedback}")

# Render
if "Render decision: True" in feedback:
    logger.info("🎬 Rendering split screen...")
    video_tool.forward(
        assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
        javascript="await render();",
        output="output/edit2_split_screen.mp4"
    )
    logger.info("🎉 Edit 2 complete! Saved to output/edit2_split_screen.mp4")
else:
    logger.warning("❌ Feedback failed")

client.close()
