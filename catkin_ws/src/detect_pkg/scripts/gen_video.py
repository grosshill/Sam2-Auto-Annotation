#!/usr/bin/env python3

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def list_pngs(folder: str) -> List[str]:
	files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
	files.sort()
	return [os.path.join(folder, f) for f in files]


def sample_indices(total_available: int, total_needed: int) -> np.ndarray:
	if total_available <= 0:
		return np.array([], dtype=int)
	# Evenly sample across the available range. Repeats if total_needed > total_available.
	return np.linspace(0, total_available - 1, total_needed).astype(int)


def prepare_writer(first_frame_path: str, out_path: str, fps: int) -> Tuple[cv2.VideoWriter, Tuple[int, int]]:
	frame = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
	if frame is None:
		raise RuntimeError(f"Failed to read first frame: {first_frame_path}")
	height, width = frame.shape[:2]
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
	if not writer.isOpened():
		raise RuntimeError(f"Failed to open video writer: {out_path}")
	return writer, (width, height)


def write_video_from_folder(folder: str, out_path: str, fps: int, duration_s: int) -> None:
	frames = list_pngs(folder)
	if not frames:
		raise RuntimeError(f"No PNG frames found in {folder}")

	total_needed = fps * duration_s
	indices = sample_indices(len(frames), total_needed)

	writer, (width, height) = prepare_writer(frames[indices[0]], out_path, fps)
	try:
		for idx in indices:
			frame = cv2.imread(frames[idx], cv2.IMREAD_COLOR)
			if frame is None:
				raise RuntimeError(f"Failed to read frame: {frames[idx]}")
			if frame.shape[1] != width or frame.shape[0] != height:
				frame = cv2.resize(frame, (width, height))
			writer.write(frame)
	finally:
		writer.release()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Build 30 Hz videos from track1/track2 PNG frames with frame sampling."
	)
	parser.add_argument("--root", default=".", help="Root directory containing track1 and track2")
	parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
	parser.add_argument("--duration", type=int, default=107, help="Video duration in seconds")
	parser.add_argument("--track1", default="track1", help="Track1 folder name")
	parser.add_argument("--track2", default="track2", help="Track2 folder name")
	parser.add_argument("--out1", default="track1.mp4", help="Output video for track1")
	parser.add_argument("--out2", default="track2.mp4", help="Output video for track2")
	args = parser.parse_args()

	track1_dir = os.path.join(args.root, args.track1)
	track2_dir = os.path.join(args.root, args.track2)

	write_video_from_folder(track1_dir, args.out1, args.fps, args.duration)
	write_video_from_folder(track2_dir, args.out2, args.fps, args.duration)


if __name__ == "__main__":
	main()
