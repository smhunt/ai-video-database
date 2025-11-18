"""Edit 1: Title Sequence with Multiple Text Animations"""
from src.client import DiffusionClient
from src.tools import VideoEditorTool, VisualFeedbackTool
from loguru import logger

client = DiffusionClient()
video_tool = VideoEditorTool(client=client)
feedback_tool = VisualFeedbackTool(client=client)

logger.info("🎬 Creating Title Sequence Video...")

# Create a title sequence with multiple text overlays
result = video_tool.forward(
    assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
    javascript="""
    const [videoFile] = assets();

    // Video clip - 15 seconds
    const video = new core.VideoClip(videoFile, {
        position: 'center',
        height: '100%'
    }).subclip(0, 450);

    // Main title (appears first 5 seconds)
    const mainTitle = new core.TextClip({
        text: 'BIG BUCK BUNNY',
        position: 'center',
        fontSize: 100,
        fontWeight: 'bold',
        fillColor: '#FFD700',
        stroke: {
            color: '#000000',
            width: 4
        }
    });
    mainTitle.duration = 150;
    mainTitle.delay = 0;

    // Subtitle (appears 5-10 seconds)
    const subtitle = new core.TextClip({
        text: 'An Epic Adventure',
        position: { x: 'center', y: '70%' },
        fontSize: 50,
        fillColor: '#FFFFFF',
        fontStyle: 'italic',
        stroke: {
            color: '#000000',
            width: 2
        }
    });
    subtitle.duration = 150;
    subtitle.delay = 150;

    // Credits (appears 10-15 seconds)
    const credits = new core.TextClip({
        text: 'Created with AI Video Database',
        position: { x: 'center', y: '80%' },
        fontSize: 30,
        fillColor: '#CCCCCC'
    });
    credits.duration = 150;
    credits.delay = 300;

    // Add all clips
    await composition.add(video);
    await composition.add(mainTitle);
    await composition.add(subtitle);
    await composition.add(credits);

    await sample();
    """,
    output="output/edit1_title_sequence.mp4"
)
logger.info(f"✅ Composition created: {result}")

# Get feedback
feedback = feedback_tool.forward(
    final_goal="Video should show three different text overlays appearing at different times"
)
logger.info(f"Feedback: {feedback}")

# Render
if "Render decision: True" in feedback:
    logger.info("🎬 Rendering title sequence...")
    video_tool.forward(
        assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
        javascript="await render();",
        output="output/edit1_title_sequence.mp4"
    )
    logger.info("🎉 Edit 1 complete! Saved to output/edit1_title_sequence.mp4")
else:
    logger.warning("❌ Feedback failed")

client.close()
