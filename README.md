
## DWPose-Yolo11

Whole-body pose estimation toolkit that pairs a YOLO11m detector with a DW-LL UCoco pose head, both in ONNX format. The pipeline predicts 133 keypoints (body, feet, face, and hands) and can run on CPU or GPU. Triton-ready configs and lightweight Python clients are included for serving.

### Key Features
- Whole-body coverage (18 body, 6 foot, 68 face, 42 hand keypoints).
- Pure ONNX inference.
- Detector is swappable: use any Ultralytics YOLO checkpoint.
- Triton ensemble configs plus HTTP/gRPC/SHM clients for production serving.
- Triton ensemble runs fully on GPU (pre/post included), avoiding CPU↔GPU copies.
- Optional visualization utilities for quick demos and debugging.


#### Yolo onnx Model Export
If you need to regenerate the YOLO detector ONNX:
```
yolo export model=yolo11m.pt format=onnx simplify=True dynamic=True imgsz=640 half=True device=0
```


### Testing
- `python test_dwpose.py` — local ONNX inference on images or video; set paths and device inside the file.
- `python test_triton_client.py --input <img_or_video> --url localhost:8000` — Triton HTTP smoke test with optional visualization/save.
- `python test_triton_client_grpc.py --input <img_or_video> --url localhost:8001` — Triton gRPC smoke test with optional visualization/save.
Prefer the `test_*.py` scripts for validation; `ensemble_client*.py` are lower-level client helpers.
- `python triton_benchmark.py --input <img> --url localhost:8001 --method all --runs 10` — benchmark Triton end-to-end latency across HTTP/gRPC/SHM; auto maps ports, runs warmup, reports avg/min/max/FPS, optional `--show`/`--output` for visualization.

### Acknowledgement
- Built on the open-source DWPose work: https://github.com/IDEA-Research/DWPose
- Thanks to Ultralytics YOLO ecosystem: https://github.com/ultralytics/ultralytics