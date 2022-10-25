python -m torch.distributed.launch --nproc_per_node 2 train.py --data dock-coco.yaml --cfg yolov5n-dock.yaml --weights yolov5n.pt --batch 128 --imgsz 224

