# src/app_softcsrnet_local.py
import argparse
import os
import sys
import cv2
import numpy as np

from src.detectors.soft_csrnet import SoftCSRNetCounter

def parse_args():
    p = argparse.ArgumentParser(description="Soft-CSRNet crowd density from local video")
    p.add_argument("--video", required=True, help="Path to input video (e.g., data/crowd.mp4)")
    p.add_argument("--save", default=None, help="Optional output path (e.g., out/softcsrnet.mp4)")
    p.add_argument("--weights", default=None, help="Path to Soft-CSRNet weights (.pth).")
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor to calibrate counts.")
    p.add_argument("--blur", type=int, default=9, help="Heatmap blur sigma.")
    p.add_argument("--no-tile", action="store_true", help="Disable tiling (frames are small).")
    p.add_argument("--ema", type=float, default=0.6, help="EMA decay (0 disables smoothing).")
    p.add_argument("--roi", default=None, help="ROI polygon as x1,y1;x2,y2;... (screen px)")
    p.add_argument("--heat_alpha", type=float, default=0.55, help="Heatmap overlay alpha [0..1]")
    p.add_argument("--homography", default=None, help="(optional) 3x3 .npy for m^2 density (compute in app if needed)")
    return p.parse_args()

def parse_roi(roi_str):
    if not roi_str:
        return None
    pts = []
    for pair in roi_str.split(";"):
        x, y = pair.split(",")
        pts.append((int(float(x)), int(float(y))))
    if len(pts) < 3:
        print("ROI must have >=3 points; ignoring.", file=sys.stderr)
        return None
    return [pts]

def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"Input video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    counter = SoftCSRNetCounter(
        weights_path=args.weights,
        tile=not args.no_tile,
        scale_factor=args.scale,
        blur_sigma=args.blur,
        ema_decay=args.ema
    )

    rois = parse_roi(args.roi)
    H = None
    if args.homography:
        try:
            H = np.load(args.homography)
            if H.shape != (3, 3):
                print("Homography must be 3x3; ignoring.", file=sys.stderr)
                H = None
        except Exception as e:
            print(f"Could not load homography ({e}); continuing without m^2 density.", file=sys.stderr)

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"Failed to open writer at {args.save}; continuing without saving.", file=sys.stderr)
            writer = None

    win = "Soft-CSRNet Crowd Density"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = counter.process(frame, rois=rois, H=H, return_visualized=True, heat_alpha=args.heat_alpha)
        view = res.annotated if res.annotated is not None else frame

        cv2.imshow(win, view)
        if writer is not None:
            writer.write(view)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
