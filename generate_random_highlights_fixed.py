"""Generate Random Highlight Compilations - FIXED VERSION"""
from src.client import DiffusionClient
from src.tools import VideoEditorTool
from loguru import logger
import random
import sys


def create_highlight_compilation_fixed(
    output_name,
    num_clips=10,
    clip_length=3,
):
    """Create a random highlight compilation that actually works"""

    client = DiffusionClient()
    video_tool = VideoEditorTool(client=client)

    logger.info(f"🎬 Creating random highlight: {output_name}")

    # Pick random timestamps from Big Buck Bunny (596 seconds total)
    # Avoid first and last 5 seconds
    available_start = 5 * 30  # 5 seconds in frames
    available_end = (596 - 5 - clip_length) * 30  # Leave room for clip

    clips = []
    used_ranges = []

    attempts = 0
    while len(clips) < num_clips and attempts < 100:
        attempts += 1
        start_frame = random.randint(available_start, available_end)
        end_frame = start_frame + (clip_length * 30)

        # Check if overlaps with existing clips
        overlaps = False
        for used_start, used_end in used_ranges:
            if start_frame < used_end + 150 and end_frame > used_start - 150:  # 5 second gap
                overlaps = True
                break

        if not overlaps:
            clips.append((start_frame, end_frame))
            used_ranges.append((start_frame, end_frame))
            logger.info(f"  Clip {len(clips)}: {start_frame/30:.1f}s - {end_frame/30:.1f}s")

    clips.sort()  # Sort chronologically

    # Build composition with proper delays
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFD700', '#FF00FF',
              '#00FFFF', '#FFA500', '#FF1493', '#7FFF00', '#FF4500']

    # Create JavaScript
    js_code = "const [videoFile] = assets();\n\n"

    timeline_pos = 0
    for i, (start, end) in enumerate(clips, 1):
        duration = end - start
        color = colors[i % len(colors)]

        # Add video clip
        js_code += f"""
    const clip{i} = new core.VideoClip(videoFile, {{
        position: 'center',
        height: '100%'
    }}).subclip({start}, {end});
    clip{i}.delay = {timeline_pos};

    const text{i} = new core.TextClip({{
        text: 'CLIP #{i}',
        position: {{ x: 'center', y: '10%' }},
        fontSize: 50,
        fontWeight: 'bold',
        fillColor: '{color}',
        stroke: {{ color: '#000000', width: 3 }}
    }});
    text{i}.duration = {duration};
    text{i}.delay = {timeline_pos};

    await composition.add(clip{i});
    await composition.add(text{i});
"""
        timeline_pos += duration

    # Add main title
    js_code += f"""
    const title = new core.TextClip({{
        text: 'RANDOM HIGHLIGHTS',
        position: {{ x: 'center', y: '90%' }},
        fontSize: 35,
        fillColor: '#FFFFFF',
        fontWeight: 'bold',
        stroke: {{ color: '#000000', width: 2 }}
    }});
    title.duration = {timeline_pos};
    await composition.add(title);

    // Preview first, then render
    await sample();
    await render();
"""

    # Execute
    result = video_tool.forward(
        assets=["assets/big_buck_bunny_1080p_30fps.mp4"],
        javascript=js_code,
        output=f"output/{output_name}.mp4"
    )

    logger.info(f"✅ {output_name}.mp4 - Duration: {timeline_pos/30:.1f}s")
    client.close()
    return result


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    logger.info(f"🚀 Generating {num} random highlight compilation(s)...")

    for i in range(1, num + 1):
        create_highlight_compilation_fixed(
            output_name=f"highlights_fixed_{i:02d}",
            num_clips=10,
            clip_length=3
        )

    logger.info(f"✅ Done! Check output/highlights_fixed_*.mp4")
