"""Shared model settings and repository-relative paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "roneneldan/TinyStories-3M"
MODEL_REVISION = "cfaf26ec85ecdfc1bd7c2638104cce55cb67f894"
SEQUENCE_LENGTH = 32
GENERATED_LENGTH = 20
PROMPT = 'Once upon a time, a curious little girl named Lily lived in a quiet village beside a green forest. Every morning, she carried fresh bread to her grandmother and watched the birds in the trees.'

MODEL_DIR = ROOT / "artifacts/models"
EXPORT_DIR = ROOT / "artifacts/exports"
RESULT_DIR = ROOT / "results"
C_DIR = ROOT / "src/c/generated"
MODEL_PATH = MODEL_DIR / "block0_w8a8_baseline.tflite"
