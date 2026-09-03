from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd


# ============================================================
# TYPES
# ============================================================


ProgressCallback = Callable[
    [float, str],
    None,
]


# ============================================================
# FESTIVAL SPECIFICATION
# ============================================================


@dataclass(frozen=True)
class FestivalSpec:
    """
    Exercise 3 festival video-delivery specification.

    Mandatory requirements:

        Container:
            MP4

        Video codec:
            H.265 / HEVC

        Audio codec:
            AAC

        Frame rate:
            25 FPS

        Display aspect ratio:
            16:9

        Resolution:
            854 x 480

        Video bitrate:
            2-5 Mb/s
            == 2000-5000 kb/s

        Audio bitrate:
            up to 256 kb/s

        Audio channels:
            stereo == 2 channels
    """

    container: str = "mp4"

    video_codec: str = "hevc"

    audio_codec: str = "aac"

    frame_rate: float = 25.0

    aspect_ratio_width: int = 16

    aspect_ratio_height: int = 9

    width: int = 854

    height: int = 480

    minimum_video_bitrate_kbps: int = 2000

    maximum_video_bitrate_kbps: int = 5000

    maximum_audio_bitrate_kbps: int = 256

    audio_channels: int = 2

    audio_sample_rate_hz: int = 48000

    # Central value inside the mandatory 2-5 Mb/s range.
    fixed_video_bitrate_kbps: int = 3000

    # Safely below the mandatory maximum of 256 kb/s.
    fixed_audio_bitrate_kbps: int = 192

    # H.265 VBV buffer.
    ffmpeg_buffer_kbps: int = 6000

    # Compliance tolerances.
    frame_rate_tolerance: float = 0.01

    aspect_ratio_tolerance: float = 0.005

    @property
    def resolution(
        self,
    ) -> tuple[int, int]:

        return (
            self.width,
            self.height,
        )

    @property
    def target_aspect_ratio(
        self,
    ) -> float:

        return (
            self.aspect_ratio_width
            / self.aspect_ratio_height
        )

    @property
    def target_sample_aspect_ratio(
        self,
    ) -> Fraction:
        """
        854x480 is not mathematically exactly 16:9 when using
        square pixels.

        Required SAR:

            (16/9) / (854/480)
            = 1280/1281

        This allows both:

            Resolution = 854x480
            Display aspect ratio = 16:9
        """

        return Fraction(
            self.aspect_ratio_width
            * self.height,
            self.aspect_ratio_height
            * self.width,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return asdict(
            self
        )


DEFAULT_SPEC = FestivalSpec()


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
}


# ============================================================
# TOOLCHAIN
# ============================================================


@dataclass
class ToolchainInfo:

    ffmpeg: str | None

    ffprobe: str | None

    ffmpeg_version: str = ""

    ffprobe_version: str = ""

    libx265_available: bool = False

    @property
    def available(
        self,
    ) -> bool:

        return (
            self.ffmpeg is not None
            and
            self.ffprobe is not None
        )

    @property
    def ready_for_conversion(
        self,
    ) -> bool:

        return (
            self.available
            and
            self.libx265_available
        )


def _candidate_tool_paths(
    executable_name: str,
) -> list[Path]:

    suffix = (
        ".exe"
        if os.name == "nt"
        else ""
    )

    filename = (
        executable_name
        + suffix
    )

    candidates: list[Path] = []

    prefix = Path(
        sys.prefix
    )

    candidates.extend(
        [
            prefix
            / "Library"
            / "bin"
            / filename,

            prefix
            / "bin"
            / filename,
        ]
    )

    conda_prefix = os.environ.get(
        "CONDA_PREFIX"
    )

    if conda_prefix:

        conda_path = Path(
            conda_prefix
        )

        candidates.extend(
            [
                conda_path
                / "Library"
                / "bin"
                / filename,

                conda_path
                / "bin"
                / filename,
            ]
        )

    return candidates


def locate_executable(
    name: str,
    environment_variable: str,
) -> str | None:

    explicit = os.environ.get(
        environment_variable
    )

    if explicit:

        explicit_path = Path(
            explicit
        )

        if explicit_path.is_file():

            return str(
                explicit_path
            )

    discovered = shutil.which(
        name
    )

    if discovered:

        return discovered

    for candidate in _candidate_tool_paths(
        name
    ):

        if candidate.is_file():

            return str(
                candidate
            )

    return None


def _first_version_line(
    executable: str,
) -> str:

    try:

        result = subprocess.run(
            [
                executable,
                "-version",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            check=True,
        )

        lines = (
            result.stdout
            .strip()
            .splitlines()
        )

        if lines:

            return lines[0]

    except Exception:

        pass

    return ""


def discover_toolchain(
) -> ToolchainInfo:

    ffmpeg = locate_executable(
        "ffmpeg",
        "FFMPEG_BINARY",
    )

    ffprobe = locate_executable(
        "ffprobe",
        "FFPROBE_BINARY",
    )

    tools = ToolchainInfo(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )

    if ffmpeg:

        tools.ffmpeg_version = (
            _first_version_line(
                ffmpeg
            )
        )

        try:

            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-encoders",
                ],
                capture_output=True,
                text=True,
                errors="replace",
                check=True,
            )

            text = (
                result.stdout
                + result.stderr
            )

            tools.libx265_available = (
                "libx265"
                in text
            )

        except Exception:

            tools.libx265_available = False

    if ffprobe:

        tools.ffprobe_version = (
            _first_version_line(
                ffprobe
            )
        )

    return tools


def require_probe(
    tools: ToolchainInfo,
) -> None:

    if not tools.ffprobe:

        raise FileNotFoundError(
            "ffprobe was not found. Install FFmpeg and "
            "ensure ffprobe is available on PATH."
        )


def require_conversion_tools(
    tools: ToolchainInfo,
) -> None:

    if not tools.ffmpeg:

        raise FileNotFoundError(
            "ffmpeg was not found."
        )

    if not tools.ffprobe:

        raise FileNotFoundError(
            "ffprobe was not found."
        )

    if not tools.libx265_available:

        raise RuntimeError(
            "The installed FFmpeg build does not expose "
            "the libx265 H.265 encoder."
        )


# ============================================================
# SUBPROCESS
# ============================================================


def run_subprocess(
    command: list[str],
) -> subprocess.CompletedProcess:

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Command failed:\n\n"
            + " ".join(
                str(item)
                for item
                in command
            )
            + "\n\nSTDERR:\n"
            + result.stderr
        )

    return result


# ============================================================
# FFPROBE
# ============================================================


def get_video_info(
    video_path: Path | str,
    tools: ToolchainInfo,
) -> dict[str, Any]:

    require_probe(
        tools
    )

    video_path = Path(
        video_path
    )

    command = [
        str(
            tools.ffprobe
        ),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-count_frames",
        str(
            video_path
        ),
    ]

    result = run_subprocess(
        command
    )

    if not result.stdout.strip():

        raise RuntimeError(
            "ffprobe returned no metadata for "
            f"{video_path}"
        )

    info = json.loads(
        result.stdout
    )

    # Preserve the examined source path so that MOV versus MP4
    # can be distinguished reliably.
    info["_source_path"] = str(
        video_path
    )

    return info


def find_stream(
    info: dict[str, Any],
    stream_type: str,
) -> dict[str, Any] | None:

    return next(
        (
            stream
            for stream
            in info.get(
                "streams",
                [],
            )
            if (
                stream.get(
                    "codec_type"
                )
                == stream_type
            )
        ),
        None,
    )


# ============================================================
# SAFE VALUE CONVERSION
# ============================================================


def safe_int(
    value,
    default=None,
):

    try:

        if (
            value is None
            or
            value == ""
            or
            value == "N/A"
        ):

            return default

        return int(
            float(
                value
            )
        )

    except Exception:

        return default


def safe_float(
    value,
    default=None,
):

    try:

        if (
            value is None
            or
            value == ""
            or
            value == "N/A"
        ):

            return default

        return float(
            value
        )

    except Exception:

        return default


def rational_to_float(
    value,
) -> float | None:

    try:

        if (
            value is None
            or
            value == ""
            or
            value == "N/A"
            or
            value == "0/0"
        ):

            return None

        return float(
            Fraction(
                str(
                    value
                )
            )
        )

    except Exception:

        return None


def display_ratio_to_float(
    value,
) -> float | None:

    if (
        value is None
        or
        value == ""
        or
        value == "N/A"
        or
        value == "0:1"
        or
        value == "0/1"
    ):

        return None

    text = str(
        value
    )

    try:

        if ":" in text:

            numerator, denominator = (
                text.split(
                    ":",
                    1,
                )
            )

            denominator_value = float(
                denominator
            )

            if denominator_value == 0:

                return None

            return (
                float(
                    numerator
                )
                / denominator_value
            )

        return rational_to_float(
            text
        )

    except Exception:

        return None


def bitrate_to_kbps(
    value,
) -> int:

    try:

        if (
            value is None
            or
            value == ""
            or
            value == "N/A"
        ):

            return 0

        return int(
            round(
                float(
                    value
                )
                / 1000.0
            )
        )

    except Exception:

        return 0


# ============================================================
# CONTAINER / CODEC HELPERS
# ============================================================


def is_mp4_container(
    format_name: str,
) -> bool:
    """
    Basic FFprobe format-family test.

    FFprobe often reports MP4 and MOV using the shared family:

        mov,mp4,m4a,3gp,3g2,mj2

    Use is_mp4_format() for final coursework compliance.
    """

    tokens = {
        token.strip().lower()
        for token
        in str(
            format_name
            or ""
        ).split(",")
    }

    return (
        "mp4"
        in tokens
    )


def is_mp4_format(
    info: dict[str, Any],
) -> bool:
    """
    Coursework-level MP4 test.

    A .mov file must not accidentally PASS simply because FFprobe
    exposes MOV and MP4 through the same demuxer family.
    """

    fmt = info.get(
        "format",
        {},
    )

    format_name = str(
        fmt.get(
            "format_name",
            "",
        )
    )

    if not is_mp4_container(
        format_name
    ):

        return False

    source_path = info.get(
        "_source_path"
    )

    if source_path:

        suffix = Path(
            source_path
        ).suffix.lower()

        if suffix:

            return (
                suffix == ".mp4"
            )

    major_brand = str(
        fmt.get(
            "tags",
            {},
        ).get(
            "major_brand",
            "",
        )
    ).strip().lower()

    if major_brand.startswith(
        "qt"
    ):

        return False

    return True


def normalised_container_name(
    format_name: str,
) -> str:

    if is_mp4_container(
        format_name
    ):

        return "MP4"

    return (
        format_name
        or "Unknown"
    )


def get_container_display_name(
    info: dict[str, Any],
    video_path: Path | str | None = None,
) -> str:

    if is_mp4_format(
        info
    ):

        return "MP4"

    if video_path is not None:

        suffix = Path(
            video_path
        ).suffix.lower()

        if suffix:

            return suffix.lstrip(
                "."
            ).upper()

    return normalised_container_name(
        info.get(
            "format",
            {},
        ).get(
            "format_name",
            "",
        )
    )


def is_hevc_codec(
    codec_name,
) -> bool:

    codec = str(
        codec_name
        or ""
    ).strip().lower()

    return (
        codec
        in {
            "hevc",
            "h265",
            "h.265",
        }
    )


# ============================================================
# VIDEO METADATA HELPERS
# ============================================================


def get_video_fps(
    video: dict[str, Any],
) -> float | None:

    average = rational_to_float(
        video.get(
            "avg_frame_rate"
        )
    )

    if (
        average is not None
        and
        average > 0
    ):

        return average

    real = rational_to_float(
        video.get(
            "r_frame_rate"
        )
    )

    if (
        real is not None
        and
        real > 0
    ):

        return real

    return None


def get_video_display_aspect_ratio(
    video: dict[str, Any],
) -> float | None:

    display_ratio = display_ratio_to_float(
        video.get(
            "display_aspect_ratio"
        )
    )

    if display_ratio is not None:

        return display_ratio

    width = safe_int(
        video.get(
            "width"
        ),
        0,
    )

    height = safe_int(
        video.get(
            "height"
        ),
        0,
    )

    if (
        width <= 0
        or
        height <= 0
    ):

        return None

    sample_ratio = display_ratio_to_float(
        video.get(
            "sample_aspect_ratio"
        )
    )

    if sample_ratio is None:

        sample_ratio = 1.0

    return (
        width
        / height
        * sample_ratio
    )


# ============================================================
# BITRATE EXTRACTION
# ============================================================


def estimate_audio_bitrate_kbps(
    info: dict[str, Any],
) -> int:

    audio = find_stream(
        info,
        "audio",
    )

    if audio is None:

        return 0

    possible_values = [
        audio.get(
            "bit_rate"
        ),
        audio.get(
            "tags",
            {},
        ).get(
            "BPS"
        ),
        audio.get(
            "tags",
            {},
        ).get(
            "BPS-eng"
        ),
    ]

    for value in possible_values:

        bitrate = bitrate_to_kbps(
            value
        )

        if bitrate > 0:

            return bitrate

    return 0


def estimate_video_bitrate_kbps(
    info: dict[str, Any],
) -> int:
    """
    Estimate VIDEO bitrate rather than blindly using the entire
    container bitrate.

    Priority:

        1. video stream bit_rate
        2. video BPS tag
        3. total format bitrate - audio bitrate
        4. file-size/duration estimate - audio bitrate
    """

    video = find_stream(
        info,
        "video",
    )

    if video is None:

        return 0

    possible_values = [
        video.get(
            "bit_rate"
        ),
        video.get(
            "tags",
            {},
        ).get(
            "BPS"
        ),
        video.get(
            "tags",
            {},
        ).get(
            "BPS-eng"
        ),
    ]

    for value in possible_values:

        bitrate = bitrate_to_kbps(
            value
        )

        if bitrate > 0:

            return bitrate

    fmt = info.get(
        "format",
        {},
    )

    format_bitrate = bitrate_to_kbps(
        fmt.get(
            "bit_rate"
        )
    )

    audio_bitrate = (
        estimate_audio_bitrate_kbps(
            info
        )
    )

    if format_bitrate > 0:

        estimated = (
            format_bitrate
            - audio_bitrate
        )

        if estimated > 0:

            return estimated

    duration = safe_float(
        fmt.get(
            "duration"
        ),
        None,
    )

    file_size = safe_float(
        fmt.get(
            "size"
        ),
        None,
    )

    if (
        duration is not None
        and
        duration > 0
        and
        file_size is not None
        and
        file_size > 0
    ):

        approximate_total = (
            file_size
            * 8.0
            / duration
            / 1000.0
        )

        approximate_video = (
            approximate_total
            - audio_bitrate
        )

        return max(
            0,
            int(
                round(
                    approximate_video
                )
            ),
        )

    return 0


# ============================================================
# METADATA EXTRACTION
# ============================================================


def extract_media_metrics_from_info(
    video_path: Path | str,
    info: dict[str, Any],
) -> dict[str, Any]:

    video_path = Path(
        video_path
    )

    fmt = info.get(
        "format",
        {},
    )

    video = find_stream(
        info,
        "video",
    )

    audio = find_stream(
        info,
        "audio",
    )

    if video is None:

        raise RuntimeError(
            f"No video stream found in {video_path.name}."
        )

    duration = safe_float(
        (
            fmt.get(
                "duration"
            )
            or
            video.get(
                "duration"
            )
        ),
        None,
    )

    fps = get_video_fps(
        video
    )

    frame_count = safe_int(
        (
            video.get(
                "nb_read_frames"
            )
            or
            video.get(
                "nb_frames"
            )
        ),
        None,
    )

    if (
        frame_count is None
        and
        duration is not None
        and
        fps is not None
    ):

        frame_count = int(
            round(
                duration
                * fps
            )
        )

    width = safe_int(
        video.get(
            "width"
        ),
        0,
    )

    height = safe_int(
        video.get(
            "height"
        ),
        0,
    )

    return {
        "file":
            video_path.name,

        "path":
            str(
                video_path
            ),

        "file_size_bytes":
            (
                video_path.stat().st_size
                if video_path.exists()
                else None
            ),

        "container":
            fmt.get(
                "format_name",
                "",
            ),

        "container_display":
            get_container_display_name(
                info,
                video_path,
            ),

        "duration_sec":
            duration,

        "width":
            width,

        "height":
            height,

        "aspect_ratio":
            get_video_display_aspect_ratio(
                video
            ),

        "display_aspect_ratio":
            video.get(
                "display_aspect_ratio"
            ),

        "sample_aspect_ratio":
            video.get(
                "sample_aspect_ratio"
            ),

        "fps":
            fps,

        "frame_count":
            frame_count,

        "video_codec":
            video.get(
                "codec_name"
            ),

        "video_profile":
            video.get(
                "profile"
            ),

        "pixel_format":
            video.get(
                "pix_fmt"
            ),

        "video_bitrate_kbps":
            estimate_video_bitrate_kbps(
                info
            ),

        "audio_codec":
            (
                audio.get(
                    "codec_name"
                )
                if audio is not None
                else None
            ),

        "audio_bitrate_kbps":
            estimate_audio_bitrate_kbps(
                info
            ),

        "audio_channels":
            (
                safe_int(
                    audio.get(
                        "channels"
                    ),
                    None,
                )
                if audio is not None
                else None
            ),

        "audio_sample_rate":
            (
                safe_int(
                    audio.get(
                        "sample_rate"
                    ),
                    None,
                )
                if audio is not None
                else None
            ),

        "audio_present":
            audio is not None,
    }


def extract_media_metrics(
    video_path: Path | str,
    tools: ToolchainInfo,
) -> dict[str, Any]:

    info = get_video_info(
        video_path,
        tools,
    )

    return extract_media_metrics_from_info(
        video_path,
        info,
    )


# ============================================================
# COMPLIANCE CHECK
# ============================================================


def check_compliance(
    info: dict[str, Any],
    spec: FestivalSpec = DEFAULT_SPEC,
) -> list[str]:
    """
    Return every problematic field.

    An empty list means all nine mandatory coursework fields pass.
    """

    errors: list[str] = []

    video = find_stream(
        info,
        "video",
    )

    audio = find_stream(
        info,
        "audio",
    )

    # --------------------------------------------------------
    # 1. CONTAINER
    # --------------------------------------------------------

    if not is_mp4_format(
        info
    ):

        source_path = info.get(
            "_source_path"
        )

        if source_path:

            container_text = (
                Path(
                    source_path
                ).suffix
                .lstrip(
                    "."
                )
                .lower()
            )

        else:

            container_text = (
                info.get(
                    "format",
                    {},
                ).get(
                    "format_name",
                    "",
                )
                or "unreadable"
            )

        errors.append(
            (
                "container: "
                f"{container_text}"
            )
        )

    # --------------------------------------------------------
    # VIDEO STREAM
    # --------------------------------------------------------

    if video is None:

        errors.extend(
            [
                "video stream missing",
                "video codec: missing",
                "resolution: missing",
                "aspect ratio: missing",
                "frame rate: missing",
                "video bitrate: missing",
            ]
        )

        return errors

    # --------------------------------------------------------
    # 2. VIDEO CODEC
    # --------------------------------------------------------

    video_codec = video.get(
        "codec_name"
    )

    if not is_hevc_codec(
        video_codec
    ):

        errors.append(
            (
                "video codec: "
                f"{video_codec or 'unreadable'}"
            )
        )

    # --------------------------------------------------------
    # 3. RESOLUTION
    # --------------------------------------------------------

    width = safe_int(
        video.get(
            "width"
        ),
        0,
    )

    height = safe_int(
        video.get(
            "height"
        ),
        0,
    )

    if (
        width,
        height,
    ) != spec.resolution:

        errors.append(
            (
                "resolution: "
                f"{width}x{height}"
            )
        )

    # --------------------------------------------------------
    # 4. ASPECT RATIO
    # --------------------------------------------------------

    aspect_ratio = (
        get_video_display_aspect_ratio(
            video
        )
    )

    if aspect_ratio is None:

        errors.append(
            "aspect ratio: unreadable"
        )

    elif (
        abs(
            aspect_ratio
            - spec.target_aspect_ratio
        )
        > spec.aspect_ratio_tolerance
    ):

        errors.append(
            (
                "aspect ratio: "
                f"{aspect_ratio:.6f}"
            )
        )

    # --------------------------------------------------------
    # 5. FRAME RATE
    # --------------------------------------------------------

    fps = get_video_fps(
        video
    )

    if fps is None:

        errors.append(
            "frame rate: unreadable"
        )

    elif (
        abs(
            fps
            - spec.frame_rate
        )
        > spec.frame_rate_tolerance
    ):

        errors.append(
            (
                "frame rate: "
                f"{fps:.3f} fps"
            )
        )

    # --------------------------------------------------------
    # 6. VIDEO BITRATE
    # --------------------------------------------------------

    video_bitrate = (
        estimate_video_bitrate_kbps(
            info
        )
    )

    if video_bitrate <= 0:

        errors.append(
            "video bitrate: unreadable"
        )

    elif not (
        spec.minimum_video_bitrate_kbps
        <= video_bitrate
        <= spec.maximum_video_bitrate_kbps
    ):

        errors.append(
            (
                "video bitrate: "
                f"{video_bitrate} kb/s"
            )
        )

    # --------------------------------------------------------
    # AUDIO STREAM
    # --------------------------------------------------------

    if audio is None:

        errors.extend(
            [
                "audio stream missing",
                "audio codec: missing",
                "audio bitrate: missing",
                "audio channels: missing",
            ]
        )

        return errors

    # --------------------------------------------------------
    # 7. AUDIO CODEC
    # --------------------------------------------------------

    audio_codec = str(
        audio.get(
            "codec_name"
        )
        or ""
    ).lower()

    if (
        audio_codec
        != spec.audio_codec
    ):

        errors.append(
            (
                "audio codec: "
                f"{audio_codec or 'unreadable'}"
            )
        )

    # --------------------------------------------------------
    # 8. AUDIO BITRATE
    # --------------------------------------------------------

    audio_bitrate = (
        estimate_audio_bitrate_kbps(
            info
        )
    )

    if audio_bitrate <= 0:

        errors.append(
            "audio bitrate: unreadable"
        )

    elif (
        audio_bitrate
        > spec.maximum_audio_bitrate_kbps
    ):

        errors.append(
            (
                "audio bitrate: "
                f"{audio_bitrate} kb/s"
            )
        )

    # --------------------------------------------------------
    # 9. AUDIO CHANNELS
    # --------------------------------------------------------

    channels = safe_int(
        audio.get(
            "channels"
        ),
        0,
    )

    if (
        channels
        != spec.audio_channels
    ):

        errors.append(
            (
                "audio channels: "
                f"{channels}"
            )
        )

    return errors


# ============================================================
# FIELD-BY-FIELD TABLE
# ============================================================


def compliance_field_rows(
    input_file: str,
    info: dict[str, Any],
    spec: FestivalSpec = DEFAULT_SPEC,
    stage: str = "Original",
) -> list[dict[str, Any]]:
    """
    Produce the exact nine coursework verification rows.
    """

    video = find_stream(
        info,
        "video",
    )

    audio = find_stream(
        info,
        "audio",
    )

    rows: list[
        dict[str, Any]
    ] = []

    def add_row(
        field_name: str,
        expected,
        actual,
        passed: bool,
    ) -> None:

        rows.append(
            {
                "input_file":
                    input_file,

                "stage":
                    stage,

                "field":
                    field_name,

                "expected":
                    expected,

                "actual":
                    actual,

                "status":
                    (
                        "PASS"
                        if passed
                        else "FAIL"
                    ),
            }
        )

    # --------------------------------------------------------
    # CONTAINER
    # --------------------------------------------------------

    container_ok = is_mp4_format(
        info
    )

    if container_ok:

        actual_container = "MP4"

    else:

        source_path = info.get(
            "_source_path"
        )

        if source_path:

            actual_container = (
                Path(
                    source_path
                ).suffix
                .lstrip(
                    "."
                )
                .upper()
                or "Unreadable"
            )

        else:

            actual_container = (
                info.get(
                    "format",
                    {},
                ).get(
                    "format_name",
                    "",
                )
                or "Unreadable"
            )

    add_row(
        "Container",
        "MP4",
        actual_container,
        container_ok,
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    if video is None:

        add_row(
            "Video codec",
            "H.265 / HEVC",
            "Missing video stream",
            False,
        )

        add_row(
            "Resolution",
            "854x480",
            "Missing video stream",
            False,
        )

        add_row(
            "Aspect ratio",
            "16:9",
            "Missing video stream",
            False,
        )

        add_row(
            "Frame rate",
            "25 fps",
            "Missing video stream",
            False,
        )

        add_row(
            "Video bitrate",
            "2000-5000 kb/s",
            "Missing video stream",
            False,
        )

    else:

        video_codec = video.get(
            "codec_name"
        )

        codec_ok = is_hevc_codec(
            video_codec
        )

        add_row(
            "Video codec",
            "H.265 / HEVC",
            (
                "H.265 / HEVC"
                if codec_ok
                else (
                    video_codec
                    or "Unreadable"
                )
            ),
            codec_ok,
        )

        width = safe_int(
            video.get(
                "width"
            ),
            0,
        )

        height = safe_int(
            video.get(
                "height"
            ),
            0,
        )

        resolution_ok = (
            (
                width,
                height,
            )
            == spec.resolution
        )

        add_row(
            "Resolution",
            (
                f"{spec.width}"
                "x"
                f"{spec.height}"
            ),
            (
                f"{width}"
                "x"
                f"{height}"
            ),
            resolution_ok,
        )

        aspect_ratio = (
            get_video_display_aspect_ratio(
                video
            )
        )

        aspect_ok = (
            aspect_ratio is not None
            and
            abs(
                aspect_ratio
                - spec.target_aspect_ratio
            )
            <= spec.aspect_ratio_tolerance
        )

        add_row(
            "Aspect ratio",
            "16:9",
            (
                "16:9"
                if aspect_ok
                else (
                    f"{aspect_ratio:.4f}"
                    if aspect_ratio is not None
                    else "Unreadable"
                )
            ),
            aspect_ok,
        )

        fps = get_video_fps(
            video
        )

        fps_ok = (
            fps is not None
            and
            abs(
                fps
                - spec.frame_rate
            )
            <= spec.frame_rate_tolerance
        )

        add_row(
            "Frame rate",
            "25 fps",
            (
                f"{fps:.3f} fps"
                if fps is not None
                else "Unreadable"
            ),
            fps_ok,
        )

        video_bitrate = (
            estimate_video_bitrate_kbps(
                info
            )
        )

        video_bitrate_ok = (
            spec.minimum_video_bitrate_kbps
            <= video_bitrate
            <= spec.maximum_video_bitrate_kbps
        )

        add_row(
            "Video bitrate",
            (
                f"{spec.minimum_video_bitrate_kbps}"
                "-"
                f"{spec.maximum_video_bitrate_kbps}"
                " kb/s"
            ),
            (
                f"{video_bitrate} kb/s"
                if video_bitrate > 0
                else "Unreadable"
            ),
            video_bitrate_ok,
        )

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    if audio is None:

        add_row(
            "Audio codec",
            "AAC",
            "Missing audio stream",
            False,
        )

        add_row(
            "Audio bitrate",
            "≤256 kb/s",
            "Missing audio stream",
            False,
        )

        add_row(
            "Audio channels",
            "2 / stereo",
            "Missing audio stream",
            False,
        )

    else:

        audio_codec = str(
            audio.get(
                "codec_name"
            )
            or ""
        ).lower()

        audio_codec_ok = (
            audio_codec
            == spec.audio_codec
        )

        add_row(
            "Audio codec",
            "AAC",
            (
                "AAC"
                if audio_codec_ok
                else (
                    audio_codec
                    or "Unreadable"
                )
            ),
            audio_codec_ok,
        )

        audio_bitrate = (
            estimate_audio_bitrate_kbps(
                info
            )
        )

        audio_bitrate_ok = (
            0
            < audio_bitrate
            <= spec.maximum_audio_bitrate_kbps
        )

        add_row(
            "Audio bitrate",
            (
                "≤"
                f"{spec.maximum_audio_bitrate_kbps}"
                " kb/s"
            ),
            (
                f"{audio_bitrate} kb/s"
                if audio_bitrate > 0
                else "Unreadable"
            ),
            audio_bitrate_ok,
        )

        channels = safe_int(
            audio.get(
                "channels"
            ),
            0,
        )

        stereo_ok = (
            channels
            == spec.audio_channels
        )

        add_row(
            "Audio channels",
            "2 / stereo",
            (
                "2 / stereo"
                if stereo_ok
                else str(
                    channels
                )
            ),
            stereo_ok,
        )

    return rows


# ============================================================
# OUTPUT FILENAMES
# ============================================================


def converted_ok_path(
    input_path: Path | str,
    output_directory: Path | str | None = None,
) -> Path:
    """
    Exercise 3 body specification:

        ORIGINALNAME_convertedOK.mp4
    """

    input_path = Path(
        input_path
    )

    if output_directory is None:

        output_directory = (
            input_path.parent
        )

    output_directory = Path(
        output_directory
    )

    return (
        output_directory
        / f"{input_path.stem}_convertedOK.mp4"
    )


def format_ok_path(
    input_path: Path | str,
    output_directory: Path | str | None = None,
) -> Path:
    """
    Marking-rubric-compatible alias:

        ORIGINALNAME_formatOK.mp4
    """

    input_path = Path(
        input_path
    )

    if output_directory is None:

        output_directory = (
            input_path.parent
        )

    output_directory = Path(
        output_directory
    )

    return (
        output_directory
        / f"{input_path.stem}_formatOK.mp4"
    )


def coursework_converted_path(
    input_path: Path | str,
    output_directory: Path | str | None = None,
) -> Path:
    """
    Backwards-compatible helper.

    The primary coursework filename is *_convertedOK.mp4.
    """

    return converted_ok_path(
        input_path,
        output_directory,
    )


# ============================================================
# FFMPEG PROGRESS
# ============================================================


def _parse_ffmpeg_time(
    value: str,
) -> float | None:

    try:

        hours, minutes, seconds = (
            value.split(
                ":"
            )
        )

        return (
            float(
                hours
            )
            * 3600
            +
            float(
                minutes
            )
            * 60
            +
            float(
                seconds
            )
        )

    except Exception:

        return None


# ============================================================
# BITRATE NORMALISATION
# ============================================================


def normalise_requested_video_bitrate(
    selected_video_bitrate_kbps: int,
    spec: FestivalSpec,
) -> int:
    """
    Keep the actual target safely inside the required
    2000-5000 kb/s range.
    """

    requested = int(
        selected_video_bitrate_kbps
    )

    if not (
        spec.minimum_video_bitrate_kbps
        <= requested
        <= spec.maximum_video_bitrate_kbps
    ):

        raise ValueError(
            "Video bitrate must be between "
            "2000 and 5000 kb/s."
        )

    safe_minimum = (
        spec.minimum_video_bitrate_kbps
        + 200
    )

    safe_maximum = (
        spec.maximum_video_bitrate_kbps
        - 200
    )

    return max(
        safe_minimum,
        min(
            requested,
            safe_maximum,
        ),
    )


# ============================================================
# FFMPEG COMMAND
# ============================================================


def build_conversion_command(
    input_path: Path | str,
    output_path: Path | str,
    tools: ToolchainInfo,
    selected_video_bitrate_kbps: int,
    has_audio: bool,
    spec: FestivalSpec = DEFAULT_SPEC,
) -> list[str]:
    """
    Build a strict Exercise 3 conversion command.

    Enforced output:

        MP4
        H.265 / HEVC
        AAC
        25 FPS CFR
        16:9 display aspect
        854x480
        2000-5000 kb/s video
        <=256 kb/s audio
        stereo audio
    """

    require_conversion_tools(
        tools
    )

    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    bitrate = (
        normalise_requested_video_bitrate(
            selected_video_bitrate_kbps,
            spec,
        )
    )

    sar = (
        spec.target_sample_aspect_ratio
    )

    sar_text = (
        f"{sar.numerator}/"
        f"{sar.denominator}"
    )

    buffer_kbps = max(
        spec.ffmpeg_buffer_kbps,
        bitrate * 2,
    )

    command = [
        str(
            tools.ffmpeg
        ),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(
            input_path
        ),
    ]

    # --------------------------------------------------------
    # CREATE SILENT STEREO AUDIO WHEN SOURCE HAS NO AUDIO
    # --------------------------------------------------------

    if not has_audio:

        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                (
                    "anullsrc="
                    "channel_layout=stereo:"
                    f"sample_rate={spec.audio_sample_rate_hz}"
                ),
            ]
        )

    # --------------------------------------------------------
    # STREAM MAPPING
    # --------------------------------------------------------

    if has_audio:

        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
            ]
        )

    else:

        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        )

    # --------------------------------------------------------
    # VIDEO FILTER
    # --------------------------------------------------------

    video_filter = (
        "scale="
        f"{spec.width}:"
        f"{spec.height}:"
        "force_original_aspect_ratio=decrease:"
        "flags=lanczos,"
        "pad="
        f"{spec.width}:"
        f"{spec.height}:"
        "(ow-iw)/2:"
        "(oh-ih)/2:"
        "color=black,"
        f"fps={int(spec.frame_rate)},"
        f"setsar={sar_text},"
        "setdar=16/9"
    )

    # --------------------------------------------------------
    # X265 RATE CONTROL
    # --------------------------------------------------------

    x265_parameters = (
        f"vbv-maxrate={bitrate}:"
        f"vbv-bufsize={buffer_kbps}:"
        "vbv-init=1:"
        "nal-hrd=cbr:"
        "strict-cbr=1:"
        "filler=1:"
        "force-cfr=1:"
        "log-level=error"
    )

    command.extend(
        [
            # VIDEO
            "-vf",
            video_filter,

            "-c:v",
            "libx265",

            "-tag:v",
            "hvc1",

            "-preset",
            "medium",

            "-b:v",
            f"{bitrate}k",

            "-minrate",
            f"{bitrate}k",

            "-maxrate",
            f"{bitrate}k",

            "-bufsize",
            f"{buffer_kbps}k",

            "-x265-params",
            x265_parameters,

            "-pix_fmt",
            "yuv420p",

            "-r",
            str(
                int(
                    spec.frame_rate
                )
            ),

            "-fps_mode",
            "cfr",

            "-video_track_timescale",
            "90000",

            "-aspect",
            "16:9",

            # AUDIO
            "-c:a",
            "aac",

            "-b:a",
            f"{spec.fixed_audio_bitrate_kbps}k",

            "-ac",
            str(
                spec.audio_channels
            ),

            "-ar",
            str(
                spec.audio_sample_rate_hz
            ),

            # CONTAINER / METADATA
            "-map_metadata",
            "-1",

            "-metadata:s:v:0",
            "rotate=0",

            "-movflags",
            "+faststart",

            "-f",
            "mp4",
        ]
    )

    if not has_audio:

        command.append(
            "-shortest"
        )

    command.extend(
        [
            "-progress",
            "pipe:1",

            "-nostats",

            str(
                output_path
            ),
        ]
    )

    return command


# ============================================================
# LOW-LEVEL FFMPEG EXECUTION
# ============================================================


def _run_ffmpeg_conversion(
    command: list[str],
    input_name: str,
    duration: float,
    progress_callback: ProgressCallback | None,
) -> tuple[
    float,
    str,
]:

    started = time.perf_counter()

    with tempfile.TemporaryFile(
        mode="w+",
        encoding="utf-8",
    ) as stderr_file:

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        if process.stdout is None:

            raise RuntimeError(
                "Unable to read FFmpeg progress."
            )

        for raw_line in process.stdout:

            line = raw_line.strip()

            if "=" not in line:

                continue

            key, value = line.split(
                "=",
                1,
            )

            current_time = None

            if key == "out_time":

                current_time = (
                    _parse_ffmpeg_time(
                        value
                    )
                )

            elif (
                key
                in {
                    "out_time_us",
                    "out_time_ms",
                }
            ):

                raw_time = safe_float(
                    value,
                    None,
                )

                if raw_time is not None:

                    current_time = (
                        raw_time
                        / 1_000_000.0
                    )

            if (
                current_time is not None
                and
                duration > 0
                and
                progress_callback is not None
            ):

                progress_callback(
                    min(
                        0.99,
                        current_time
                        / duration,
                    ),
                    (
                        "Transcoding "
                        f"{input_name}"
                    ),
                )

            if (
                key == "progress"
                and
                value == "end"
                and
                progress_callback is not None
            ):

                progress_callback(
                    1.0,
                    (
                        "Transcoding complete: "
                        f"{input_name}"
                    ),
                )

        return_code = (
            process.wait()
        )

        stderr_file.seek(
            0
        )

        stderr_text = (
            stderr_file.read()
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    if return_code != 0:

        raise RuntimeError(
            "FFmpeg conversion failed.\n\n"
            + stderr_text
        )

    return (
        elapsed,
        stderr_text,
    )


# ============================================================
# POST-CONVERSION VALIDATION
# ============================================================


def validate_converted_video(
    output_path: Path | str,
    tools: ToolchainInfo,
    spec: FestivalSpec = DEFAULT_SPEC,
) -> dict[str, Any]:
    """
    Run ffprobe again after conversion.

    A converted file is accepted only when all nine mandatory
    Exercise 3 fields pass.
    """

    output_path = Path(
        output_path
    )

    info = get_video_info(
        output_path,
        tools,
    )

    errors = check_compliance(
        info,
        spec,
    )

    rows = compliance_field_rows(
        output_path.name,
        info,
        spec,
        stage="Converted",
    )

    all_nine_pass = (
        len(
            rows
        )
        == 9
        and
        all(
            row.get(
                "status"
            )
            == "PASS"
            for row
            in rows
        )
    )

    return {
        "info":
            info,

        "errors":
            errors,

        "rows":
            rows,

        "all_nine_fields_pass":
            all_nine_pass,

        "compliant":
            (
                len(
                    errors
                )
                == 0
                and
                all_nine_pass
            ),
    }


# ============================================================
# COMPLETE ORIGINAL / CONVERTED VERIFICATION
# ============================================================


def build_verification_result(
    original_path: Path | str,
    converted_path: Path | str | None,
    tools: ToolchainInfo,
    spec: FestivalSpec = DEFAULT_SPEC,
) -> dict[str, Any]:
    """
    Build separate original and converted verification evidence.

    This prevents the Streamlit UI from accidentally displaying
    the original FAIL rows as though they were converted results.
    """

    original_path = Path(
        original_path
    )

    original_info = get_video_info(
        original_path,
        tools,
    )

    original_errors = check_compliance(
        original_info,
        spec,
    )

    original_rows = compliance_field_rows(
        input_file=(
            original_path.name
        ),
        info=(
            original_info
        ),
        spec=(
            spec
        ),
        stage="Original",
    )

    original_metrics = (
        extract_media_metrics_from_info(
            original_path,
            original_info,
        )
    )

    result: dict[str, Any] = {
        "original_path":
            str(
                original_path
            ),

        "original_file":
            original_path.name,

        "original_info":
            original_info,

        "original_metrics":
            original_metrics,

        "original_errors":
            original_errors,

        "original_compliant":
            (
                len(
                    original_errors
                )
                == 0
            ),

        "original_field_rows":
            original_rows,

        "converted_path":
            None,

        "converted_file":
            None,

        "converted_info":
            None,

        "converted_metrics":
            None,

        "converted_errors":
            [],

        "converted_compliant":
            None,

        "converted_field_rows":
            [],

        "all_nine_fields_pass":
            None,
    }

    if converted_path is None:

        return result

    converted_path = Path(
        converted_path
    )

    if not converted_path.is_file():

        return result

    validation = validate_converted_video(
        converted_path,
        tools,
        spec,
    )

    converted_info = validation[
        "info"
    ]

    converted_metrics = (
        extract_media_metrics_from_info(
            converted_path,
            converted_info,
        )
    )

    result.update(
        {
            "converted_path":
                str(
                    converted_path
                ),

            "converted_file":
                converted_path.name,

            "converted_info":
                converted_info,

            "converted_metrics":
                converted_metrics,

            "converted_errors":
                validation[
                    "errors"
                ],

            "converted_compliant":
                validation[
                    "compliant"
                ],

            "converted_field_rows":
                validation[
                    "rows"
                ],

            "all_nine_fields_pass":
                validation[
                    "all_nine_fields_pass"
                ],
        }
    )

    return result


# ============================================================
# CONVERSION
# ============================================================


def convert_video(
    input_path: Path | str,
    output_path: Path | str,
    tools: ToolchainInfo,
    selected_video_bitrate_kbps: int = 3000,
    spec: FestivalSpec = DEFAULT_SPEC,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Convert one film into the mandatory Exercise 3 format.

    Success is reported only when all nine converted fields pass.

    Both filename conventions are created:

        *_convertedOK.mp4
        *_formatOK.mp4
    """

    require_conversion_tools(
        tools
    )

    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # ORIGINAL ANALYSIS
    # --------------------------------------------------------

    source_info = get_video_info(
        input_path,
        tools,
    )

    source_errors = check_compliance(
        source_info,
        spec,
    )

    source_rows = compliance_field_rows(
        input_file=(
            input_path.name
        ),
        info=(
            source_info
        ),
        spec=(
            spec
        ),
        stage="Original",
    )

    source_metrics = (
        extract_media_metrics_from_info(
            input_path,
            source_info,
        )
    )

    duration = float(
        source_metrics.get(
            "duration_sec"
        )
        or 0.0
    )

    has_audio = bool(
        source_metrics.get(
            "audio_present"
        )
    )

    actual_target_bitrate = (
        normalise_requested_video_bitrate(
            selected_video_bitrate_kbps,
            spec,
        )
    )

    # --------------------------------------------------------
    # BUILD COMMAND
    # --------------------------------------------------------

    command = build_conversion_command(
        input_path=input_path,
        output_path=output_path,
        tools=tools,
        selected_video_bitrate_kbps=(
            actual_target_bitrate
        ),
        has_audio=has_audio,
        spec=spec,
    )

    if output_path.exists():

        try:

            output_path.unlink()

        except OSError:

            pass

    # --------------------------------------------------------
    # TRANSCODE
    # --------------------------------------------------------

    try:

        (
            elapsed,
            stderr_text,
        ) = _run_ffmpeg_conversion(
            command=command,
            input_name=input_path.name,
            duration=duration,
            progress_callback=(
                progress_callback
            ),
        )

    except Exception:

        if output_path.exists():

            try:

                output_path.unlink()

            except OSError:

                pass

        raise

    # --------------------------------------------------------
    # BASIC OUTPUT CHECK
    # --------------------------------------------------------

    if (
        not output_path.is_file()
        or
        output_path.stat().st_size <= 0
    ):

        raise RuntimeError(
            "FFmpeg completed but did not create "
            "a usable converted video."
        )

    # --------------------------------------------------------
    # MANDATORY POST-CONVERSION VERIFICATION
    # --------------------------------------------------------

    validation = validate_converted_video(
        output_path,
        tools,
        spec,
    )

    converted_info = validation[
        "info"
    ]

    converted_errors = validation[
        "errors"
    ]

    converted_rows = validation[
        "rows"
    ]

    if not validation[
        "compliant"
    ]:

        failure_text = "\n".join(
            (
                "• "
                + error
            )
            for error
            in converted_errors
        )

        if not failure_text:

            failure_text = (
                "Converted output did not produce "
                "exactly nine PASS rows."
            )

        raise RuntimeError(
            (
                "Converted video failed mandatory "
                "Exercise 3 verification.\n\n"
                "Failed fields:\n"
                f"{failure_text}"
            )
        )

    converted_metrics = (
        extract_media_metrics_from_info(
            output_path,
            converted_info,
        )
    )

    # --------------------------------------------------------
    # CREATE BOTH COURSEWORK-COMPATIBLE FILENAMES
    # --------------------------------------------------------

    converted_alias = converted_ok_path(
        input_path,
        output_path.parent,
    )

    format_alias = format_ok_path(
        input_path,
        output_path.parent,
    )

    aliases = [
        converted_alias,
        format_alias,
    ]

    for alias_path in aliases:

        try:

            same_file = (
                output_path.resolve()
                ==
                alias_path.resolve()
            )

        except OSError:

            same_file = (
                output_path
                ==
                alias_path
            )

        if not same_file:

            shutil.copy2(
                output_path,
                alias_path,
            )

        alias_validation = (
            validate_converted_video(
                alias_path,
                tools,
                spec,
            )
        )

        if not alias_validation[
            "compliant"
        ]:

            raise RuntimeError(
                (
                    f"{alias_path.name} was created "
                    "but failed post-copy verification."
                )
            )

    # --------------------------------------------------------
    # FINAL NINE-FIELD ASSERTION
    # --------------------------------------------------------

    all_pass = all(
        row.get(
            "status"
        )
        == "PASS"
        for row
        in converted_rows
    )

    if (
        len(
            converted_rows
        )
        != 9
        or
        not all_pass
    ):

        raise RuntimeError(
            (
                "Internal verification inconsistency: "
                "the converted output did not produce "
                "exactly nine PASS rows."
            )
        )

    # --------------------------------------------------------
    # COMPLETE RESULT
    # --------------------------------------------------------

    return {
        "conversion_performed":
            True,

        "conversion_reason":
            (
                "Original file was non-compliant."
                if source_errors
                else (
                    "Conversion was explicitly requested "
                    "for an already compliant file."
                )
            ),

        "original_path":
            str(
                input_path
            ),

        "original_file":
            input_path.name,

        "original_compliant":
            (
                len(
                    source_errors
                )
                == 0
            ),

        "original_errors":
            source_errors,

        "original_info":
            source_info,

        "original_field_rows":
            source_rows,

        "output_path":
            str(
                output_path
            ),

        "converted_path":
            str(
                converted_alias
            ),

        "converted_file":
            converted_alias.name,

        "convertedOK_path":
            str(
                converted_alias
            ),

        "formatOK_path":
            str(
                format_alias
            ),

        "coursework_output_path":
            str(
                converted_alias
            ),

        "processing_time_seconds":
            elapsed,

        "requested_video_bitrate_kbps":
            int(
                selected_video_bitrate_kbps
            ),

        "actual_target_video_bitrate_kbps":
            actual_target_bitrate,

        "command":
            command,

        "ffmpeg_stderr":
            stderr_text,

        "source_metrics":
            source_metrics,

        "converted_metrics":
            converted_metrics,

        "converted_info":
            converted_info,

        "converted_errors":
            [],

        "converted_field_rows":
            converted_rows,

        # Backwards compatibility with existing app.py code.
        "field_rows":
            converted_rows,

        "compliance_errors":
            [],

        "converted_compliant":
            True,

        "all_nine_fields_pass":
            True,

        "compliant":
            True,
    }


# ============================================================
# COURSEWORK CHECK + CONDITIONAL CONVERSION
# ============================================================


def process_video_for_coursework(
    input_path: Path | str,
    output_directory: Path | str,
    tools: ToolchainInfo,
    selected_video_bitrate_kbps: int = 3000,
    spec: FestivalSpec = DEFAULT_SPEC,
    convert_already_compliant: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Official Exercise 3 workflow.

    Default behaviour:

        compliant input
            -> inspect only
            -> do NOT re-encode

        non-compliant input
            -> inspect
            -> automatically convert
            -> ffprobe converted copy
            -> require 9 / 9 PASS

    Therefore:

        convert_already_compliant=False

    is the correct normal coursework configuration.
    """

    input_path = Path(
        input_path
    )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_info = get_video_info(
        input_path,
        tools,
    )

    original_errors = check_compliance(
        original_info,
        spec,
    )

    original_rows = compliance_field_rows(
        input_file=(
            input_path.name
        ),
        info=(
            original_info
        ),
        spec=(
            spec
        ),
        stage="Original",
    )

    original_metrics = (
        extract_media_metrics_from_info(
            input_path,
            original_info,
        )
    )

    original_compliant = (
        len(
            original_errors
        )
        == 0
    )

    # --------------------------------------------------------
    # ALREADY COMPLIANT -> NO CONVERSION BY DEFAULT
    # --------------------------------------------------------

    if (
        original_compliant
        and
        not convert_already_compliant
    ):

        if progress_callback is not None:

            progress_callback(
                1.0,
                (
                    f"{input_path.name} is already compliant; "
                    "conversion skipped."
                ),
            )

        return {
            "conversion_performed":
                False,

            "conversion_reason":
                (
                    "Already compliant; "
                    "conversion not required."
                ),

            "original_path":
                str(
                    input_path
                ),

            "original_file":
                input_path.name,

            "original_compliant":
                True,

            "original_errors":
                [],

            "original_info":
                original_info,

            "original_metrics":
                original_metrics,

            "original_field_rows":
                original_rows,

            "converted_path":
                None,

            "converted_file":
                None,

            "converted_compliant":
                None,

            "converted_field_rows":
                [],

            "all_nine_fields_pass":
                None,

            "output_path":
                None,

            "convertedOK_path":
                None,

            "formatOK_path":
                None,

            "coursework_output_path":
                None,

            # Backwards-compatible field.
            "field_rows":
                original_rows,

            "compliance_errors":
                [],

            # The source itself is compliant.
            "compliant":
                True,
        }

    # --------------------------------------------------------
    # NON-COMPLIANT OR FORCED CONVERSION
    # --------------------------------------------------------

    primary_output = converted_ok_path(
        input_path,
        output_directory,
    )

    conversion_result = convert_video(
        input_path=input_path,
        output_path=primary_output,
        tools=tools,
        selected_video_bitrate_kbps=(
            selected_video_bitrate_kbps
        ),
        spec=spec,
        progress_callback=(
            progress_callback
        ),
    )

    conversion_result[
        "conversion_reason"
    ] = (
        "Original file was non-compliant."
        if not original_compliant
        else (
            "Already compliant, but conversion "
            "was explicitly requested."
        )
    )

    return conversion_result


# ============================================================
# THUMBNAIL
# ============================================================


def extract_thumbnail(
    video_path: Path | str,
    maximum_width: int = 900,
) -> np.ndarray | None:

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():

        return None

    try:

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames > 0:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                max(
                    0,
                    total_frames // 3,
                ),
            )

        success, frame = (
            capture.read()
        )

        if not success:

            return None

        height, width = (
            frame.shape[:2]
        )

        if width > maximum_width:

            scale = (
                maximum_width
                / width
            )

            frame = cv2.resize(
                frame,
                (
                    maximum_width,
                    int(
                        round(
                            height
                            * scale
                        )
                    ),
                ),
                interpolation=cv2.INTER_AREA,
            )

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

    finally:

        capture.release()


# ============================================================
# TEMPORAL COMPLEXITY
# ============================================================


def estimate_temporal_variation(
    video_path: Path | str,
    max_samples: int = 60,
    resize_width: int = 320,
) -> dict[str, Any]:

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():

        return {
            "temporal_variation_mean":
                0.0,

            "temporal_variation_std":
                0.0,

            "sampled_frames":
                0,
        }

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if frame_count > 0:

        stride = max(
            1,
            frame_count
            // max(
                max_samples,
                1,
            ),
        )

    else:

        stride = 1

    previous_gray = None

    differences: list[
        float
    ] = []

    sampled = 0

    frame_index = 0

    try:

        while sampled < max_samples:

            success, frame = (
                capture.read()
            )

            if not success:

                break

            if (
                frame_index
                % stride
                != 0
            ):

                frame_index += 1

                continue

            frame_index += 1

            _, width = (
                frame.shape[:2]
            )

            if width > resize_width:

                scale = (
                    resize_width
                    / width
                )

                frame = cv2.resize(
                    frame,
                    (
                        resize_width,
                        int(
                            round(
                                frame.shape[0]
                                * scale
                            )
                        ),
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            gray = cv2.GaussianBlur(
                gray,
                (
                    5,
                    5,
                ),
                0,
            )

            if previous_gray is not None:

                difference = cv2.absdiff(
                    previous_gray,
                    gray,
                )

                differences.append(
                    float(
                        np.mean(
                            difference
                        )
                    )
                )

            previous_gray = gray

            sampled += 1

    finally:

        capture.release()

    if not differences:

        return {
            "temporal_variation_mean":
                0.0,

            "temporal_variation_std":
                0.0,

            "sampled_frames":
                sampled,
        }

    return {
        "temporal_variation_mean":
            round(
                float(
                    np.mean(
                        differences
                    )
                ),
                3,
            ),

        "temporal_variation_std":
            round(
                float(
                    np.std(
                        differences
                    )
                ),
                3,
            ),

        "sampled_frames":
            sampled,
    }


# ============================================================
# ADAPTIVE BITRATE EXTENSION
# ============================================================


def choose_adaptive_bitrate(
    video_path: Path | str,
    tools: ToolchainInfo,
    spec: FestivalSpec = DEFAULT_SPEC,
    temporal_samples: int = 60,
) -> dict[str, Any]:

    metrics = extract_media_metrics(
        video_path,
        tools,
    )

    variation = estimate_temporal_variation(
        video_path,
        max_samples=(
            temporal_samples
        ),
        resize_width=320,
    )

    width = int(
        metrics.get(
            "width"
        )
        or 0
    )

    height = int(
        metrics.get(
            "height"
        )
        or 0
    )

    fps = float(
        metrics.get(
            "fps"
        )
        or 0.0
    )

    source_bitrate = int(
        metrics.get(
            "video_bitrate_kbps"
        )
        or 0
    )

    temporal_variation = float(
        variation[
            "temporal_variation_mean"
        ]
    )

    source_area = (
        width
        * height
    )

    target_area = (
        spec.width
        * spec.height
    )

    score = 0

    reasons: list[str] = []

    if (
        source_area
        >= target_area
        * 3
    ):

        score += 2

        reasons.append(
            "high source resolution"
        )

    elif (
        source_area
        >= target_area
        * 1.5
    ):

        score += 1

        reasons.append(
            "moderate source resolution"
        )

    if fps >= 29:

        score += 1

        reasons.append(
            "higher source frame rate"
        )

    if source_bitrate >= 8000:

        score += 2

        reasons.append(
            "very high source bitrate"
        )

    elif source_bitrate >= 5000:

        score += 1

        reasons.append(
            "high source bitrate"
        )

    if temporal_variation >= 18:

        score += 2

        reasons.append(
            "high temporal variation"
        )

    elif temporal_variation >= 8:

        score += 1

        reasons.append(
            "moderate temporal variation"
        )

    if score <= 1:

        selected = 2400

        complexity = "Low"

    elif score <= 3:

        selected = 3000

        complexity = "Medium"

    elif score <= 5:

        selected = 4000

        complexity = "High"

    else:

        selected = 4600

        complexity = "Very High"

    selected = max(
        2200,
        min(
            selected,
            4800,
        ),
    )

    return {
        "input_file":
            Path(
                video_path
            ).name,

        "complexity_score":
            score,

        "complexity_class":
            complexity,

        "temporal_variation_mean":
            temporal_variation,

        "temporal_variation_std":
            variation[
                "temporal_variation_std"
            ],

        "selected_bitrate_kbps":
            selected,

        "selection_reason":
            (
                ", ".join(
                    reasons
                )
                if reasons
                else "low source complexity"
            ),
    }


# ============================================================
# SSIM
# ============================================================


def _ssim_grayscale(
    first: np.ndarray,
    second: np.ndarray,
) -> float:

    first = first.astype(
        np.float64
    )

    second = second.astype(
        np.float64
    )

    c1 = (
        0.01
        * 255
    ) ** 2

    c2 = (
        0.03
        * 255
    ) ** 2

    mu1 = cv2.GaussianBlur(
        first,
        (
            11,
            11,
        ),
        1.5,
    )

    mu2 = cv2.GaussianBlur(
        second,
        (
            11,
            11,
        ),
        1.5,
    )

    mu1_sq = (
        mu1
        * mu1
    )

    mu2_sq = (
        mu2
        * mu2
    )

    mu12 = (
        mu1
        * mu2
    )

    sigma1_sq = (
        cv2.GaussianBlur(
            first * first,
            (
                11,
                11,
            ),
            1.5,
        )
        - mu1_sq
    )

    sigma2_sq = (
        cv2.GaussianBlur(
            second * second,
            (
                11,
                11,
            ),
            1.5,
        )
        - mu2_sq
    )

    sigma12 = (
        cv2.GaussianBlur(
            first * second,
            (
                11,
                11,
            ),
            1.5,
        )
        - mu12
    )

    numerator = (
        (
            2
            * mu12
            + c1
        )
        *
        (
            2
            * sigma12
            + c2
        )
    )

    denominator = (
        (
            mu1_sq
            + mu2_sq
            + c1
        )
        *
        (
            sigma1_sq
            + sigma2_sq
            + c2
        )
    )

    score = np.mean(
        numerator
        /
        np.maximum(
            denominator,
            1e-12,
        )
    )

    return float(
        score
    )


# ============================================================
# PSNR / SSIM VIDEO COMPARISON
# ============================================================


def compare_video_quality(
    original_path: Path | str,
    converted_path: Path | str,
    sample_count: int = 12,
) -> dict[str, Any]:

    original_capture = cv2.VideoCapture(
        str(
            original_path
        )
    )

    converted_capture = cv2.VideoCapture(
        str(
            converted_path
        )
    )

    if (
        not original_capture.isOpened()
        or
        not converted_capture.isOpened()
    ):

        original_capture.release()

        converted_capture.release()

        return {
            "frames_compared":
                0,

            "mean_psnr_db":
                None,

            "mean_ssim":
                None,
        }

    original_fps = max(
        original_capture.get(
            cv2.CAP_PROP_FPS
        ),
        1e-9,
    )

    converted_fps = max(
        converted_capture.get(
            cv2.CAP_PROP_FPS
        ),
        1e-9,
    )

    original_duration_ms = (
        original_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        /
        original_fps
        * 1000.0
    )

    converted_duration_ms = (
        converted_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        /
        converted_fps
        * 1000.0
    )

    usable_duration = min(
        original_duration_ms,
        converted_duration_ms,
    )

    psnr_values: list[
        float
    ] = []

    ssim_values: list[
        float
    ] = []

    try:

        for index in range(
            sample_count
        ):

            fraction = (
                (
                    index
                    + 1
                )
                /
                (
                    sample_count
                    + 1
                )
            )

            timestamp = (
                usable_duration
                * fraction
            )

            original_capture.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp,
            )

            converted_capture.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp,
            )

            (
                original_ok,
                original_frame,
            ) = original_capture.read()

            (
                converted_ok,
                converted_frame,
            ) = converted_capture.read()

            if (
                not original_ok
                or
                not converted_ok
            ):

                continue

            converted_frame = cv2.resize(
                converted_frame,
                (
                    original_frame.shape[1],
                    original_frame.shape[0],
                ),
                interpolation=cv2.INTER_LINEAR,
            )

            psnr_values.append(
                float(
                    cv2.PSNR(
                        original_frame,
                        converted_frame,
                    )
                )
            )

            original_gray = cv2.cvtColor(
                original_frame,
                cv2.COLOR_BGR2GRAY,
            )

            converted_gray = cv2.cvtColor(
                converted_frame,
                cv2.COLOR_BGR2GRAY,
            )

            ssim_values.append(
                _ssim_grayscale(
                    original_gray,
                    converted_gray,
                )
            )

    finally:

        original_capture.release()

        converted_capture.release()

    return {
        "frames_compared":
            len(
                psnr_values
            ),

        "mean_psnr_db":
            (
                round(
                    float(
                        np.mean(
                            psnr_values
                        )
                    ),
                    3,
                )
                if psnr_values
                else None
            ),

        "mean_ssim":
            (
                round(
                    float(
                        np.mean(
                            ssim_values
                        )
                    ),
                    6,
                )
                if ssim_values
                else None
            ),
    }


# ============================================================
# QUALITY EVALUATION
# ============================================================


def evaluate_transcoding_quality(
    original_path: Path | str,
    converted_path: Path | str,
    tools: ToolchainInfo,
    conversion_type: str,
    selected_bitrate_kbps: int,
    sample_count: int = 12,
    spec: FestivalSpec = DEFAULT_SPEC,
) -> dict[str, Any]:

    original_metrics = extract_media_metrics(
        original_path,
        tools,
    )

    converted_info = get_video_info(
        converted_path,
        tools,
    )

    converted_metrics = (
        extract_media_metrics_from_info(
            converted_path,
            converted_info,
        )
    )

    post_errors = check_compliance(
        converted_info,
        spec,
    )

    quality = compare_video_quality(
        original_path,
        converted_path,
        sample_count=(
            sample_count
        ),
    )

    original_size = (
        Path(
            original_path
        ).stat().st_size
    )

    converted_size = (
        Path(
            converted_path
        ).stat().st_size
    )

    if original_size > 0:

        storage_reduction = (
            (
                original_size
                - converted_size
            )
            / original_size
            * 100.0
        )

    else:

        storage_reduction = None

    return {
        "input_file":
            Path(
                original_path
            ).name,

        "converted_file":
            Path(
                converted_path
            ).name,

        "conversion_type":
            conversion_type,

        "selected_bitrate_kbps":
            selected_bitrate_kbps,

        "original_size_bytes":
            original_size,

        "converted_size_bytes":
            converted_size,

        "storage_reduction_percent":
            (
                round(
                    storage_reduction,
                    2,
                )
                if storage_reduction is not None
                else None
            ),

        "original_fps":
            original_metrics.get(
                "fps"
            ),

        "converted_fps":
            converted_metrics.get(
                "fps"
            ),

        "converted_video_bitrate_kbps":
            converted_metrics.get(
                "video_bitrate_kbps"
            ),

        "converted_audio_bitrate_kbps":
            converted_metrics.get(
                "audio_bitrate_kbps"
            ),

        "frames_compared":
            quality[
                "frames_compared"
            ],

        "mean_psnr_db":
            quality[
                "mean_psnr_db"
            ],

        "mean_ssim":
            quality[
                "mean_ssim"
            ],

        "post_conversion_status":
            (
                "COMPLIANT"
                if not post_errors
                else "NON-COMPLIANT"
            ),

        "post_conversion_errors":
            "; ".join(
                post_errors
            ),
    }


# ============================================================
# FIXED VS ADAPTIVE COMPARISON
# ============================================================


def build_fixed_vs_adaptive_comparison(
    fixed_quality_df: pd.DataFrame,
    adaptive_quality_df: pd.DataFrame,
    adaptive_decisions_df: pd.DataFrame,
) -> pd.DataFrame:

    if (
        fixed_quality_df.empty
        and
        adaptive_quality_df.empty
    ):

        return pd.DataFrame()

    fixed = fixed_quality_df.copy()

    adaptive = (
        adaptive_quality_df.copy()
    )

    if not fixed.empty:

        fixed = fixed.rename(
            columns={
                column:
                    (
                        column
                        if column == "input_file"
                        else (
                            "fixed_"
                            + column
                        )
                    )
                for column
                in fixed.columns
            }
        )

    if not adaptive.empty:

        adaptive = adaptive.rename(
            columns={
                column:
                    (
                        column
                        if column == "input_file"
                        else (
                            "adaptive_"
                            + column
                        )
                    )
                for column
                in adaptive.columns
            }
        )

    if (
        not fixed.empty
        and
        not adaptive.empty
    ):

        result = fixed.merge(
            adaptive,
            on="input_file",
            how="outer",
        )

    elif not fixed.empty:

        result = fixed

    else:

        result = adaptive

    if not adaptive_decisions_df.empty:

        result = result.merge(
            adaptive_decisions_df,
            on="input_file",
            how="left",
        )

    return result


# ============================================================
# APPENDIX TABLES
# ============================================================


def build_appendix_tables(
    comparison_df: pd.DataFrame,
):

    if comparison_df.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    summary = comparison_df.copy()

    decision_columns = [
        column
        for column
        in [
            "input_file",
            "complexity_score",
            "complexity_class",
            "temporal_variation_mean",
            "selected_bitrate_kbps",
            "selection_reason",
        ]
        if column
        in summary.columns
    ]

    if decision_columns:

        decision = summary[
            decision_columns
        ].copy()

    else:

        decision = pd.DataFrame()

    quality_columns = [
        column
        for column
        in summary.columns
        if (
            column == "input_file"
            or
            "psnr"
            in column.lower()
            or
            "ssim"
            in column.lower()
            or
            "status"
            in column.lower()
            or
            "size"
            in column.lower()
        )
    ]

    if quality_columns:

        quality = summary[
            quality_columns
        ].copy()

    else:

        quality = pd.DataFrame()

    return (
        summary,
        decision,
        quality,
    )


# ============================================================
# FORMAT REPORT
# ============================================================


def generate_format_report(
    results_df: pd.DataFrame,
) -> str:

    lines: list[str] = [
        (
            "EXERCISE 3 - "
            "FESTIVAL VIDEO FORMAT REPORT"
        ),
        "=" * 78,
        "",
        "Required festival format:",
        (
            "Container: MP4 | "
            "Video: H.265/HEVC | "
            "Audio: AAC | "
            "Frame rate: 25 FPS | "
            "Aspect ratio: 16:9 | "
            "Resolution: 854x480 | "
            "Video bitrate: 2000-5000 kb/s | "
            "Audio bitrate: <=256 kb/s | "
            "Audio channels: stereo"
        ),
        "",
    ]

    for _, row in results_df.iterrows():

        input_file = str(
            row.get(
                "input_file",
                "",
            )
        )

        original_status = str(
            row.get(
                "original_status",
                "",
            )
        )

        lines.append(
            (
                f"{input_file}: "
                f"{original_status}"
            )
        )

        original_errors = str(
            row.get(
                "original_errors",
                "",
            )
            or ""
        )

        if original_errors:

            lines.append(
                " Problematic fields:"
            )

            for error in original_errors.split(
                "; "
            ):

                if error:

                    lines.append(
                        (
                            "  - "
                            + error
                        )
                    )

        converted_file = str(
            row.get(
                "converted_file",
                "",
            )
            or ""
        )

        if converted_file:

            lines.append(
                (
                    " Converted output: "
                    + converted_file
                )
            )

        converted_ok = str(
            (
                row.get(
                    "convertedOK_alias",
                    "",
                )
                or
                row.get(
                    "coursework_output_file",
                    "",
                )
                or
                row.get(
                    "coursework_output_path",
                    "",
                )
                or ""
            )
        )

        if converted_ok:

            lines.append(
                (
                    " Coursework output: "
                    + converted_ok
                )
            )

        post_status = str(
            row.get(
                "post_conversion_status",
                "",
            )
            or ""
        )

        if post_status:

            lines.append(
                (
                    " Post-conversion status: "
                    + post_status
                )
            )

        post_errors = str(
            row.get(
                "post_conversion_errors",
                "",
            )
            or ""
        )

        if post_errors:

            lines.append(
                " Post-conversion failures:"
            )

            for error in post_errors.split(
                "; "
            ):

                if error:

                    lines.append(
                        (
                            "  - "
                            + error
                        )
                    )

        lines.append(
            ""
        )

    return "\n".join(
        lines
    )


# ============================================================
# EXTENSION REPORT
# ============================================================


def generate_extension_report(
    comparison_df: pd.DataFrame,
) -> str:

    lines = [
        "EXERCISE 3 EXTENSION REPORT",
        "=" * 78,
        "",
    ]

    if comparison_df.empty:

        lines.append(
            (
                "No extension results "
                "were generated."
            )
        )

        return "\n".join(
            lines
        )

    for _, row in comparison_df.iterrows():

        lines.append(
            (
                "Input: "
                + str(
                    row.get(
                        "input_file",
                        "",
                    )
                )
            )
        )

        lines.append(
            (
                "Complexity: "
                + str(
                    row.get(
                        "complexity_class",
                        "",
                    )
                )
            )
        )

        lines.append(
            (
                "Adaptive bitrate: "
                + str(
                    row.get(
                        "selected_bitrate_kbps",
                        "",
                    )
                )
                + " kb/s"
            )
        )

        lines.append(
            ""
        )

    return "\n".join(
        lines
    )


# ============================================================
# DISPLAY SIZE
# ============================================================


def human_bytes(
    value: int | float | None,
) -> str:

    if value is None:

        return "—"

    size = float(
        value
    )

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    unit_index = 0

    while (
        size >= 1024
        and
        unit_index
        < len(
            units
        ) - 1
    ):

        size /= 1024

        unit_index += 1

    if unit_index == 0:

        return (
            f"{int(size):,} B"
        )

    return (
        f"{size:.2f} "
        f"{units[unit_index]}"
    )
