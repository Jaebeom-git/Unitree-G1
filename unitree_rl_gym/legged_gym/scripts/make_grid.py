from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# video 모드일 때 필요 (image 모드만 쓸 거면 moviepy 없이도 동작)
from moviepy.editor import (
    VideoFileClip,
    ColorClip,
    CompositeVideoClip,
    clips_array,
    ImageClip,
)

# --------------------------------------------------------
# 모드 선택: "image" 또는 "video"
# --------------------------------------------------------
MODE = "image"  # "image" or "video"
MODE = "video"  # "image" or "video"

# --------------------------------------------------------
# 공통 설정
# --------------------------------------------------------
ROOT_DIR = Path("unitree_rl_gym/logs/analysis/Feedback_Walk")
ROOT_DIR = Path("unitree_rl_gym/logs/eval/g1/Feedback_Walk")

GRID_ROWS = 3
GRID_COLS = 5

# 캔버스 크기: 고정으로 쓰려면 숫자, 자동이면 None
OUT_W = None
OUT_H = None

# 자동일 때 전체 스케일 (1.0 = 원본 그리드 크기, 0.5 = 절반)
SCALE_RATIO = 1.0

# 배경 & 캡션 공통
BACKGROUND_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
CAPTION_HEIGHT_RATIO = 0.15
GRID_MARGIN_RATIO = 1.0

# --------------------------------------------------------
# 이미지 전용 설정
# --------------------------------------------------------
# IMAGE_FILENAME = "joint_pos_action_group.png"
# IMAGE_FILENAME = "summary_group.png"
IMAGE_FILENAME = "torque_vs_vel_group.png"

# --------------------------------------------------------
# 비디오 전용 설정
# --------------------------------------------------------
VIDEO_NAME = "record.mp4"  # None이면 ROOT_DIR 아래 모든 *.mp4 사용
VIDEO_EXT = ".mp4"

OUTPUT_FORMAT = "gif"      # "mp4" or "gif"
OUTPUT_FPS = 30
# --------------------------------------------------------


# =========================
# 공통 유틸
# =========================

def extract_tag_from_folder(folder: Path) -> str:
    """
    폴더 이름 패턴 예시
      - walk_Nov27_11-17-42_default        -> default
      - walk_Nov27_12-02-31_no_lin_vel_z   -> no_lin_vel_z
      - Nov27_11-17-42_default             -> default
      - Nov27_12-02-31_no_lin_vel_z        -> no_lin_vel_z
    """
    name = folder.name
    parts = name.split("_")

    # 패턴 1: walk_날짜_시간_태그...
    if name.startswith("walk_") and len(parts) >= 4:
        return "_".join(parts[3:])

    # 패턴 2: 날짜_시간_태그...
    if len(parts) >= 3:
        return "_".join(parts[2:])

    # 그 외: 마지막 토큰이나 전체 이름
    if len(parts) >= 1:
        return parts[-1]
    return name


def choose_font_pil(height: int) -> ImageFont.FreeTypeFont:
    """PIL 폰트 선택 (캡션용)."""
    font_size = max(10, int(height * 0.6))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def get_text_color(bg_color):
    """배경색이 어두우면 흰 글씨, 밝으면 검정 글씨."""
    r, g, b = bg_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)


def find_media_paths(
    root: Path,
    mode: str,
    image_filename: str,
    video_name: Optional[str],
    video_ext: str,
):
    """이미지/비디오 공통 파일 검색."""
    if mode == "image":
        return sorted(root.rglob(image_filename))
    else:
        if video_name is None:
            return sorted(root.rglob("*" + video_ext))
        else:
            return sorted(root.rglob(video_name))


# =========================
# 이미지 그리드
# =========================

def run_image_grid(paths):
    global OUT_W, OUT_H

    if not paths:
        print(f"{IMAGE_FILENAME} 파일을 찾지 못했습니다.")
        return

    max_tiles = GRID_ROWS * GRID_COLS
    if len(paths) < max_tiles:
        print(f"경고: 이미지가 {len(paths)}장뿐입니다. "
              f"앞에서부터 차례로만 채우고 나머지는 빈칸으로 둡니다.")
    paths = paths[:max_tiles]

    # 캔버스 크기 자동 설정
    if OUT_W is None or OUT_H is None:
        sample_img = Image.open(paths[0]).convert("RGB")
        w0, h0 = sample_img.size
        sample_img.close()

        base_out_w = w0 * GRID_COLS
        base_out_h = h0 * GRID_ROWS

        OUT_W = max(1, int(base_out_w * SCALE_RATIO))
        OUT_H = max(1, int(base_out_h * SCALE_RATIO))

        print("[IMAGE] 자동 캔버스 크기 (scale={}): {}x{}".format(
            SCALE_RATIO, OUT_W, OUT_H
        ))

    tile_w = OUT_W // GRID_COLS
    tile_h = OUT_H // GRID_ROWS

    caption_px = int(tile_h * CAPTION_HEIGHT_RATIO)
    font = choose_font_pil(caption_px)
    text_color = get_text_color(BACKGROUND_COLOR)

    canvas = Image.new("RGB", (OUT_W, OUT_H), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    for idx, path in enumerate(paths):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        x0 = col * tile_w
        y0 = row * tile_h

        caption_h = int(tile_h * CAPTION_HEIGHT_RATIO)
        img_area_h = tile_h - caption_h

        img_area_w = int(tile_w * GRID_MARGIN_RATIO)
        img_area_h2 = int(img_area_h * GRID_MARGIN_RATIO)

        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(img_area_w / float(w), img_area_h2 / float(h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        img_x = x0 + (tile_w - new_w) // 2
        img_y = y0 + (img_area_h - new_h) // 2
        canvas.paste(img_resized, (img_x, img_y))

        tag = extract_tag_from_folder(path.parent)
        text_w, text_h = draw.textsize(tag, font=font)
        text_x = x0 + (tile_w - text_w) // 2
        text_y = y0 + tile_h - caption_h + (caption_h - text_h) // 2
        draw.text((text_x, text_y), tag, fill=text_color, font=font)

    out_name = "grid_{}_{}x{}.png".format(
        IMAGE_FILENAME.replace(".png", ""), GRID_ROWS, GRID_COLS
    )
    output_path = ROOT_DIR / out_name
    canvas.save(output_path, dpi=(96, 96))
    print("[IMAGE] 저장 완료: {}".format(output_path))


# =========================
# 비디오 그리드
# =========================

def create_caption_clip(text, width, height, duration, bg_color):
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    font = choose_font_pil(height)
    text_color = get_text_color(bg_color)

    tw, th = draw.textsize(text, font=font)
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=text_color, font=font)

    return ImageClip(np.array(img)).set_duration(duration)


def build_tile_clip(video_clip, tag, tile_w, tile_h):
    duration = video_clip.duration

    caption_h = int(tile_h * CAPTION_HEIGHT_RATIO)
    video_area_h = tile_h - caption_h

    bg = ColorClip(size=(tile_w, tile_h),
                   color=BACKGROUND_COLOR, duration=duration)

    max_video_w = int(tile_w * GRID_MARGIN_RATIO)
    max_video_h = int(video_area_h * GRID_MARGIN_RATIO)

    w0, h0 = video_clip.size
    scale = min(max_video_w / float(w0), max_video_h / float(h0))
    video_resized = video_clip.resize(scale)

    vw, vh = video_resized.size
    vx = (tile_w - vw) // 2
    vy = (video_area_h - vh) // 2

    caption_clip = create_caption_clip(
        tag, tile_w, caption_h, duration, BACKGROUND_COLOR
    ).set_position((0, video_area_h))

    tile = CompositeVideoClip(
        [
            bg,
            video_resized.set_position((vx, vy)),
            caption_clip,
        ],
        size=(tile_w, tile_h),
    )
    return tile


def run_video_grid(paths):
    global OUT_W, OUT_H

    if not paths:
        print("비디오 파일을 찾지 못했습니다.")
        return

    max_tiles = GRID_ROWS * GRID_COLS
    if len(paths) < max_tiles:
        print("경고: 비디오가 {}개뿐입니다. "
              "앞에서부터 차례로만 채우고 나머지는 빈칸으로 둡니다.".format(len(paths)))
    paths = paths[:max_tiles]

    clips = [VideoFileClip(str(p)) for p in paths]
    min_duration = min(c.duration for c in clips)

    # 캔버스 크기 자동 설정
    if OUT_W is None or OUT_H is None:
        w0, h0 = clips[0].size
        base_out_w = w0 * GRID_COLS
        base_out_h = h0 * GRID_ROWS
        OUT_W = max(1, int(base_out_w * SCALE_RATIO))
        OUT_H = max(1, int(base_out_h * SCALE_RATIO))
        print("[VIDEO] 자동 캔버스 크기 (scale={}): {}x{}".format(
            SCALE_RATIO, OUT_W, OUT_H
        ))

    tile_w = OUT_W // GRID_COLS
    tile_h = OUT_H // GRID_ROWS

    clipped = [c.subclip(0, min_duration).without_audio() for c in clips]

    tile_clips = []
    for clip, path in zip(clipped, paths):
        tag = extract_tag_from_folder(path.parent)
        tile = build_tile_clip(clip, tag, tile_w, tile_h)
        tile_clips.append(tile)

    while len(tile_clips) < max_tiles:
        empty = ColorClip(
            size=(tile_w, tile_h),
            color=BACKGROUND_COLOR,
            duration=min_duration,
        )
        tile_clips.append(empty)

    grid = []
    for r in range(GRID_ROWS):
        row_clips = tile_clips[r * GRID_COLS:(r + 1) * GRID_COLS]
        grid.append(row_clips)

    final = clips_array(grid).set_duration(min_duration).set_audio(None)

    base_path = ROOT_DIR / "grid_video_{}x{}".format(GRID_ROWS, GRID_COLS)
    mp4_path = base_path.with_suffix(".mp4")

    final.write_videofile(
        str(mp4_path),
        fps=OUTPUT_FPS,
        codec="libx264",
        audio=False,
    )
    print("[VIDEO] MP4 저장 완료: {}".format(mp4_path))

    # 필요하면 MP4 → GIF 변환
    if OUTPUT_FORMAT.lower() == "gif":
        gif_path = base_path.with_suffix(".gif")
        tmp_clip = VideoFileClip(str(mp4_path))
        tmp_clip.write_gif(str(gif_path), fps=OUTPUT_FPS)
        tmp_clip.close()
        print("[VIDEO] GIF 저장 완료: {}".format(gif_path))

    final.close()
    for c in clips:
        c.close()


# =========================
# 엔트리 포인트
# =========================

def main():
    paths = find_media_paths(ROOT_DIR, MODE, IMAGE_FILENAME, VIDEO_NAME, VIDEO_EXT)

    if MODE == "image":
        run_image_grid(paths)
    elif MODE == "video":
        run_video_grid(paths)
    else:
        print("MODE는 'image' 또는 'video'만 가능합니다.")


if __name__ == "__main__":
    main()
