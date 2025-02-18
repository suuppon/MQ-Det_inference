import torch
from PIL import Image
from torchvision import transforms
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.comm import get_rank  # Multi-GPU handling
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

def load_images_and_targets(image_ids, image_id_to_path, image_id_to_annotations, category_mapping, device="cuda", transform=None):
    """
    Load images and corresponding annotations from COCO format.
    Convert them into ImageList and annotations into Mask R-CNN-compatible format.
    """
    images = []
    image_sizes = []
    targets = []
    image_paths = []

    # 기본 이미지 변환 설정 (ToTensor만 적용)
    transform = transform if transform else transforms.Compose([
        transforms.ToTensor(),  # Normalize image to [0,1]
    ])

    for idx, image_id in enumerate(image_ids):
        # Load image
        img_path = image_id_to_path[image_id]
        img = Image.open(img_path).convert("RGB")  # 원본 유지
        img_tensor = transform(img)  # (C, H, W)
        
        images.append(img_tensor)  # 이미지 리스트에 추가
        image_sizes.append(img.size)  # 원본 이미지 크기 저장
        image_paths.append(img_path)  # 이미지 경로 저장

        # Load annotations
        ann_list = image_id_to_annotations[image_id]
        ann_list = [ann for ann in ann_list if ann.get("iscrowd", 0) == 0]  # ignore_crowd 적용

        boxes = []
        labels = []
        segmentations = []
        keypoints = []
        cboxes = []

        for ann in ann_list:
            x_min, y_min, width, height = ann["bbox"]
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(category_mapping.get(ann["category_id"], ann["category_id"]))  # 변환된 category_id 사용

            if "segmentation" in ann:
                segmentations.append(ann["segmentation"])

            if "keypoints" in ann:
                keypoints.append(ann["keypoints"])

            if "cbox" in ann:
                cboxes.append(ann["cbox"])

        if boxes:
            target = BoxList(
                torch.tensor(boxes, dtype=torch.float32, device=device),
                img.size, mode="xyxy"
            )

            labels = torch.tensor(labels, dtype=torch.int64, device=device)
            target.add_field("labels", labels)

            if segmentations:
                masks = SegmentationMask(segmentations, img.size, mode='poly')
                target.add_field("masks", masks)

            if keypoints:
                keypoints = PersonKeypoints(keypoints, img.size)
                target.add_field("keypoints", keypoints)

            if cboxes:
                cboxes = torch.tensor(cboxes, dtype=torch.float32, device=device).reshape(-1, 4)
                cboxes = BoxList(cboxes, img.size, mode="xywh").convert("xyxy")
                target.add_field("cbox", cboxes)

            # 이미지 크기에 맞게 클리핑
            target = target.clip_to_image(remove_empty=True)
        else:
            target = None  # 어노테이션 없는 경우

        targets.append(target)

    # 🟢 `ImageList` 객체로 변환
    images_list = ImageList(torch.stack(images).to(device), image_sizes)

    return images_list, targets

def print_class_distribution(images):
    """
    Print the number of stored queries for each class.
    """
    print("\n📊 Query Class Distribution:")
    sorted_classes = sorted(images.keys())
    for class_id in sorted_classes:
        num_queries = len(images[class_id])
        print(f" - Class {class_id}: {num_queries} queries")

def save_query_bank(query_images):
    """
    Save the extracted query images based on the configuration.
    Handles both single-GPU and multi-GPU scenarios.
    """
    if cfg.num_gpus > 1:
        global_rank = get_rank()
        save_name = 'MODEL/{}_query_{}_pool{}_{}{}_rank{}.pth'.format(
            cfg.VISION_QUERY.DATASET_NAME if cfg.VISION_QUERY.DATASET_NAME else cfg.DATASETS.TRAIN[0].split('_')[0],
            cfg.VISION_QUERY.MAX_QUERY_NUMBER,
            cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            'sel' if cfg.VISION_QUERY.SELECT_FPN_LEVEL else 'all',
            cfg.VISION_QUERY.QUERY_ADDITION_NAME,
            global_rank
        )
    else:
        if cfg.VISION_QUERY.QUERY_BANK_SAVE_PATH != '':
            save_name = cfg.VISION_QUERY.QUERY_BANK_SAVE_PATH
        else:
            save_name = 'MODEL/{}_query_{}_pool{}_{}{}.pth'.format(
                cfg.VISION_QUERY.DATASET_NAME if cfg.VISION_QUERY.DATASET_NAME else cfg.DATASETS.TRAIN[0].split('_')[0],
                cfg.VISION_QUERY.MAX_QUERY_NUMBER,
                cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
                'sel' if cfg.VISION_QUERY.SELECT_FPN_LEVEL else 'all',
                cfg.VISION_QUERY.QUERY_ADDITION_NAME
            )

    print(f"💾 Saving extracted queries to: {save_name}")
    torch.save(query_images, save_name)
    print("✅ Query Bank saved successfully!")