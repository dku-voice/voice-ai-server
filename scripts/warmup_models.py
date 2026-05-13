"""
scripts/warmup_models.py - STT / DeepFace 모델 warmup 스크립트

PR이나 시연 전에 모델 weight 다운로드와 로딩이 되는지 확인할 때 쓴다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import configure_console_encoding
from app.services.stt_service import get_model_status, load_model
from app.services.vision_service import _run_deepface_age, warm_up_deepface


def warmup_stt() -> None:
    """faster-whisper 모델 weight를 받고 로딩한다."""
    print("[Warmup] STT 모델 로딩 시작")
    load_model()

    status = get_model_status()
    if not status["ready"]:
        raise RuntimeError(f"STT 모델이 준비되지 않았습니다: {status['error']}")

    print("[Warmup] STT 모델 로딩 완료")


def warmup_vision(image_path: Path | None) -> None:
    """DeepFace age 모델 weight를 받고 로딩한다."""
    if image_path is None:
        print("[Warmup] DeepFace age 모델 로딩 시작")
        warm_up_deepface()
        print("[Warmup] DeepFace age 모델 로딩 완료")
        return

    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    print(f"[Warmup] DeepFace 실제 이미지 추론 시작: {image_path}")
    result = _run_deepface_age(image_path.read_bytes())
    print("[Warmup] DeepFace 실제 이미지 추론 완료")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STT와 DeepFace 모델 weight 다운로드/로딩을 확인합니다.",
    )
    parser.add_argument("--stt-only", action="store_true", help="STT 모델만 warmup합니다.")
    parser.add_argument("--vision-only", action="store_true", help="DeepFace 모델만 warmup합니다.")
    parser.add_argument(
        "--vision-image",
        type=Path,
        default=None,
        help="DeepFace warmup에 사용할 이미지 경로입니다. 없으면 더미 이미지로 로딩만 확인합니다.",
    )
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()

    if args.stt_only and args.vision_only:
        print("[Warmup] --stt-only와 --vision-only는 동시에 사용할 수 없습니다.", file=sys.stderr)
        return 2

    try:
        if not args.vision_only:
            warmup_stt()
        if not args.stt_only:
            warmup_vision(args.vision_image)
    except Exception as exc:
        print(f"[Warmup] 실패: {exc}", file=sys.stderr)
        return 1

    print("[Warmup] 전체 모델 warmup 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
