# import some common libraries
import os
import cv2
import random
import numpy as np
import json
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

def get_tomato_dicts(img_dir):
    json_file = os.path.join(img_dir, "_annotations.coco.json")
    dataset_dicts = []
    with open(json_file) as f:
        imgs_anns = json.load(f)
    imgs = imgs_anns['images']
    annotations = imgs_anns['annotations']
    for i in range(626):
        record = {}
        img = imgs[i]
        filename = os.path.join(img_dir, img["file_name"])
        record["file_name"] = filename
        record["image_id"] = img["id"]
        record["height"] = img["height"]
        record["width"] = img["width"]
        obj = []
        for j in range(3428):
            if(annotations[j]["image_id"] == i):
                temp_anno = dict(annotations[j])
                temp_anno["bbox_mode"] = BoxMode.XYXY_ABS
                temp_anno.pop("id",None)
                temp_anno.pop("image_id",None)
                temp_anno.pop("area",None)                
                obj.append(temp_anno)
                
        record["annotations"] = obj
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "valid"]:
    DatasetCatalog.register("tomato_" + d, lambda d=d: get_tomato_dicts("tomato/" + d))
    MetadataCatalog.get("tomato_" + d).set(thing_classes=["tomato"])
tomato_metadata = MetadataCatalog.get("tomato_train")


# visualise training dataset
dataset_dicts = get_tomato_dicts("tomato/train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d['file_name'])
    visualizer = Visualizer(img[:, :, ::-1], metadata=tomato_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("dataset",vis.get_image()[:, :, ::-1])
    cv2.waitKey(500)


# cfg = get_cfg()

# # this is the part where i can change the model. atm it uses mask rcnn. 
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("tomato_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (balloon)

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=False)
# trainer.train()

# # set up inference
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# cfg.DATASETS.TEST = ("tomato_val", )
# predictor = DefaultPredictor(cfg)

# # run & visualise inference
# dataset_dicts = get_tomato_dicts("tomato/valid")
# for d in random.sample(dataset_dicts, 3):    
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=tomato_metadata, 
#                    scale=0.8, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#     )
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow("inference",v.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

# # run evaluation
# evaluator = COCOEvaluator("tomato_val", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "tomato_val")
# inference_on_dataset(trainer.model, val_loader, evaluator) 