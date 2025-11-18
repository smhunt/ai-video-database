"""Generate Random Highlight Compilations - Avoids blank sections"""
from src.client import DiffusionClient
from src.tools import VideoEditorTool
from loguru import logger
import random
import sys

def generate_random_highlights(
    video_duration_seconds=30,  # Total duration of source video
    clip_length_seconds=3,      # Length of each highlight clip
    num_clips=10,               # Number of clips in compilation
    avoid_start_seconds=2,      # Skip first N seconds (fade in/blank)
    avoid_end_seconds=2,        # Skip last N seconds (fade out/blank)
    min_gap_seconds=2,          # Minimum gap between clips to avoid repetition
):
    """Generate random non-overlapping highlight timestamps"""

    # Convert to frames (30fps)
    total_frames = video_duration_seconds * 30
    clip_frames = clip_length_seconds * 30
    avoid_start = avoid_start_seconds * 30
    avoid_end = avoid_end_seconds * 30
    min_gap = min_gap_seconds * 30

    # Available range for clips
    usable_start = avoid_start
    usable_end = total_frames - avoid_end - clip_frames

    if usable_end <= usable_start:
        raise ValueError("Video too short for the given parameters")

    # Generate random start points
    attempts = 0
    max_attempts = 1000
    selected_clips = []

    while len(selected_clips) < num_clips and attempts < max_attempts:
        attempts += 1
        start_frame = random.randint(usable_start, usable_end)
        end_frame = start_frame + clip_frames

        # Check if this clip overlaps or is too close to existing clips
        too_close = False
        for existing_start, existing_end in selected_clips:
            if (start_frame - min_gap < existing_end and
                end_frame + min_gap > existing_start):
                too_close = True
                break

        if not too_close:
            selected_clips.append((start_frame, end_frame))

    # Sort clips chronologically
    selected_clips.sort()

    logger.info(f"Generated {len(selected_clips)} random highlight clips")
    for i, (start, end) in enumerate(selected_clips, 1):
        logger.info(f"  Clip {i}: {start/30:.1f}s - {end/30:.1f}s")

    return selected_clips


def create_highlight_compilation(
    output_name,
    num_clips=10,
    clip_length=3,
    add_labels=True,
    skip_feedback=True  # Skip AI feedback to save costs
):
    """Create a random highlight compilation"""

    client = DiffusionClient()
    video_tool = VideoEditorTool(client=client)

    logger.info(f"🎬 Creating random highlight compilation: {output_name}")

    # Generate random timestamps (Big Buck Bunny is 596 seconds long)
    clips = generate_random_highlights(
        video_duration_seconds=596,
        clip_length_seconds=clip_length,
        num_clips=num_clips,
        avoid_start_seconds=3,   # Skip opening fade
        avoid_end_seconds=3,     # Skip ending fade
        min_gap_seconds=5        # Space clips apart
    )

    # Build JavaScript code
    js_clips = []
    js_texts = []

    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFD700', '#FF00FF',
              '#00FFFF', '#FFA500', '#FF1493', '#7FFF00', '#FF4500']

    timeline_offset = 0

    for i, (start_frame, end_frame) in enumerate(clips, 1):
        clip_duration = end_frame - start_frame
        color = colors[i % len(colors)]

        # Video clip
        js_clips.append(f"""
    const clip{i} = new core.VideoClip(videoFile, {{
        position: 'center',
        height: '100%'
    }}).subclip({start_frame}, {end_frame});
    clip{i}.delay = {timeline_offset};
        """)

        # Optional label
        if add_labels:
            js_texts.append(f"""
    const text{i} = new core.TextClip({{
        text: 'CLIP #{i}',
        position: {{ x: 'center', y: '10%' }},
        fontSize: 50,
        fontWeight: 'bold',
        fillColor: '{color}',
        stroke: {{ color: '#000000', width: 3 }}
    }});
    text{i}.duration = {clip_duration};
    text{i}.delay = {timeline_offset};
            """)

        timeline_offset += clip_duration

    # Main title
    main_title = """
    const mainTitle = new core.TextClip({
        text: 'RANDOM HIGHLIGHTS',
        position: { x: 'center', y: '90%' },
        fontSize: 35,
        fillColor: '#FFFFFF',
        fontWeight: 'bold',
        stroke: { color: '#000000', width: 2 }
    });
    mainTitle.duration = """ + str(timeline_offset) + """;
    """

    # Combine all clips
    add_clips = "\n    ".join([f"await composition.add(clip{i});" for i in range(1, len(clips) + 1)])
    if add_labels:
        add_texts = "\n    ".join([f"await composition.add(text{i});" for i in range(1, len(clips) + 1)])
    else:
        add_texts = ""

    javascript = f"""
    const [videoFile] = assets();

    {"".join(js_clips)}
    {"".join(js_texts)}
    {main_title}

    // Add all clips
    {add_clips}
    {add_texts}
    await composition.add(mainTitle);

    // Render immediately (skip sampling to save time)
    await render();
    """

    # Execute
    result = video_tool.forward(
        assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
        javascript=javascript,
        output=f"output/{output_name}.mp4"
    )

    total_duration = timeline_offset / 30
    logger.info(f"✅ Result: {result}")
    logger.info(f"📊 Total duration: {total_duration:.1f} seconds ({num_clips} clips × {clip_length}s)")
    logger.info(f"🎉 Saved to output/{output_name}.mp4")

    client.close()
    return result


if __name__ == "__main__":
    # Generate multiple random compilations
    num_variations = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    logger.info(f"🚀 Generating {num_variations} random highlight compilations...")

    for i in range(1, num_variations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating compilation {i}/{num_variations}")
        logger.info(f"{'='*60}\n")

        create_highlight_compilation(
            output_name=f"random_highlights_{i:02d}",
            num_clips=10,           # 10 clips per video
            clip_length=3,          # 3 seconds each
            add_labels=True,        # Show clip numbers
            skip_feedback=True      # Skip AI to save costs
        )

    logger.info(f"\n🎉 All {num_variations} compilations complete!")
    logger.info(f"📁 Files saved in output/random_highlights_*.mp4")
