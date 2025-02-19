from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.data import make_data_loader

import os
import json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm

import argparse

from utils import load_images_and_targets, save_query_bank, print_class_distribution

class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.device = torch.device(cfg.MODEL.DEVICE)
        
    def set_model(self, model):
        if self.cfg.num_gpus > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.cfg.local_rank], output_device=self.cfg.local_rank,
                broadcast_buffers=self.cfg.MODEL.BACKBONE.USE_BN,
                find_unused_parameters=self.cfg.SOLVER.FIND_UNUSED_PARAMETERS
            )
        
        self.model = model
    
    def extract_queries(self, images, targets, query_images):        
        if self.cfg.num_gpus > 1:
            query_images = self.model.module.extract_query(images.to(self.device), targets, query_images)
        else:
            query_images = self.model.extract_query(images.to(self.device), targets, query_images)
        return query_images
    
    def set_features(self, queries, targets, save_features=False):
        query_images=defaultdict(list)
        self.features = self.extract_queries(queries, targets, query_images)
        if save_features:
            save_query_bank(self.features)

        print("✅ Query extraction completed!")
        # query_images : {class_num : Tensor(num_shots, 1, 256)}
        # So query_images is a dictionary with class_num as key and a tensor of shape (num_shots, 1, 256) as value
        print_class_distribution(query_images)
    
    def predict(self, text_prompt, image_prompt, visualize=False):
        # # Predict the image
        # with torch.no_grad():
        #     output = self.model.predict(text_prompt, image_input, self.features)
        
        # if visualize:
        #     # Visualize the output
        #     pass
        
        # return output
        pass

def main():
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser.add_argument("--config-file", 
                        default="configs/pretrain/mq-glip-t.yaml", 
                        metavar="FILE", 
                        help="Path to config file")
    
    parser.add_argument("--weight", 
                        default="MODEL/mq-glip-t", 
                        metavar="FILE", 
                        help="Path to the weight file")
    
    
    parser.add_argument("--data_root", 
                        default="DATASET/milcivil_fewshot/",
                        help="Path to COCO dataset root")
    
    
    parser.add_argument("--save-features",
                        action="store_true",
                        help="Save extracted features to disk")
    parser.add_argument("--visualize",
                        action="store_true",
                        help="Visualize the output")
    
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    
    args = parser.parse_args()
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    
    cfg.num_gpus = num_gpus
    cfg.DATA.data_root = args.data_root
    
    support_annotation_path = os.path.join(args.data_root, "support/annotations.json")
    support_image_dir = os.path.join(args.data_root, "support/images")
    query_annotation_path = os.path.join(args.data_root, "query/annotations.json")
    query_image_dir = os.path.join(args.data_root, "query/images")
    
    cfg.DATA.support_annotation_path = support_annotation_path
    cfg.DATA.support_image_dir = support_image_dir
    cfg.DATA.query_annotation_path = query_annotation_path
    cfg.DATA.query_image_dir = query_image_dir
    
    # 1. Load configuration file
    cfg.merge_from_file(args.config_file)
    
    # 2. Initialize model and load weights
    predictor = Predictor(cfg)
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    
    checkpointer = DetectronCheckpointer(cfg, model)
    _ = checkpointer.load(args.weight)
    model.eval()
    
    predictor.set_model(model)
    
    
    # 3. Load COCO dataset
    with open (support_annotation_path, "r") as f:
        data = json.load(f)
    
    # 4. Map image IDs to paths
    image_id_to_path = {img["id"]: os.path.join(support_image_dir, img["file_name"]) for img in data["images"]}
    image_id_to_annotations = defaultdict(list)

    for ann in data["annotations"]:
        image_id_to_annotations[ann["image_id"]].append(ann)
    
    category_ids = [cat["id"] for cat in data["categories"]]
    category_mapping = {cat: i for i, cat in enumerate(category_ids)}
    
    # 5. Extract vision queries
    image_ids = list(image_id_to_path.keys())

    # Convert images and targets to tensors
    images, targets = load_images_and_targets(image_ids, image_id_to_path, image_id_to_annotations, device="cuda", category_mapping=category_mapping)

    predictor.set_features(images, targets, save_features=args.save_features)
    
    # 6. Build DataLoader from Query Images
    query_data_loader = make_data_loader(cfg, is_train=False, is_distributed=False, inference_mode=True)
    
    # 7. Detect Objects in Query Images
    #TODO: Implement This
    # for query_images, query_targets in query_data_loader:
    #    predictor.predict(query_images, query_targets, visualize=args.visualize)
    

if __name__ == "__main__":
    main()