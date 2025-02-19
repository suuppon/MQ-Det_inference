import json
import os
import random
import shutil
import argparse
from collections import defaultdict

def select_support_bboxes(images, image_to_annotations, shot, categories):
    """
    Selects exactly `shot` bounding boxes per category.
    
    Parameters:
    - images: Dictionary mapping image_id to image information.
    - image_to_annotations: Dictionary mapping image_id to list of annotations.
    - shot: The required number of bounding boxes per class.
    - categories: List of category dictionaries (each should have an "id" key).
    
    Returns:
    - selected_images_set: Set of image ids chosen for the support set.
    - selected_annotations: List of selected bounding box annotations.
    - coverage: Dictionary tracking the number of bounding boxes per category.
    """
    category_bbox_count = {cat["id"]: 0 for cat in categories}  # Track bbox count per category
    selected_annotations = []  # Store selected bbox annotations
    selected_images = set()  # Track images that contribute to the support set

    # Sort images by ID to ensure consistency in selection
    remaining_images = sorted(images.keys())

    while any(category_bbox_count[c] < shot for c in category_bbox_count):
        best_image = None
        best_annotations = []
        
        for image_id in remaining_images:
            if image_id in selected_images:
                continue
            
            anns = image_to_annotations.get(image_id, [])
            new_annotations = []
            
            for ann in anns:
                cat_id = ann["category_id"]
                if category_bbox_count[cat_id] < shot:
                    new_annotations.append(ann)

            if len(new_annotations) > 0:
                best_image = image_id
                best_annotations = new_annotations
                break  # Prioritize first available image (ensuring fair selection)

        if best_image is None:
            # No more images to fulfill the shot requirement
            break

        selected_images.add(best_image)
        selected_annotations.extend(best_annotations)

        # Update the bbox count for each selected annotation
        for ann in best_annotations:
            category_bbox_count[ann["category_id"]] += 1
            
            # Stop adding if we hit the required `shot`
            if category_bbox_count[ann["category_id"]] >= shot:
                break

    return selected_images, selected_annotations, category_bbox_count

def split_few_shot_coco(annotations_path, images_dir, shot, output_dir):
    """
    Splits a COCO dataset into a few-shot dataset with exactly `shot` bounding boxes per class.
    
    Parameters:
    - annotations_path: Path to the COCO annotations JSON file.
    - images_dir: Directory containing the COCO images.
    - shot: The required number of bounding boxes per class.
    - output_dir: Directory to save the split dataset.
    """
    # Define support and query directories
    output_dir = output_dir + f"_{shot}shot"
    support_dir = os.path.join(output_dir, "support")
    query_dir = os.path.join(output_dir, "query")
    support_images_dir = os.path.join(support_dir, "images")
    query_images_dir = os.path.join(query_dir, "images")
    
    os.makedirs(support_images_dir, exist_ok=True)
    os.makedirs(query_images_dir, exist_ok=True)
    
    # Load annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)
    
    # Map image id to image information
    images = {img["id"]: img for img in coco_data["images"]}
    annotations = coco_data["annotations"]

    # Collect annotations for each image id
    image_to_annotations = defaultdict(list)
    for ann in annotations:
        image_to_annotations[ann["image_id"]].append(ann)

    # Use the new selection function that ensures `shot` bboxes per class
    support_image_ids, support_annotations, coverage = select_support_bboxes(
        images, image_to_annotations, shot, coco_data["categories"]
    )

    # Ensure exactly `num_classes * shot` bboxes in the support set
    expected_bboxes = len(coco_data["categories"]) * shot
    actual_bboxes = len(support_annotations)
    if actual_bboxes != expected_bboxes:
        print(f"Error: Expected {expected_bboxes} bounding boxes but found {actual_bboxes}.")
    
    # Separate annotations for the query set
    query_annotations = [ann for ann in annotations if ann not in support_annotations]
    query_image_ids = set(images.keys()) - support_image_ids

    # Copy images to support/query directories
    for image_id, img in images.items():
        img_path = os.path.join(images_dir, img["file_name"])
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} does not exist.")
            continue
        if image_id in support_image_ids:
            shutil.copy(img_path, os.path.join(support_images_dir, img["file_name"]))
        else:
            shutil.copy(img_path, os.path.join(query_images_dir, img["file_name"]))

    # Helper function to save annotations JSON file
    def save_annotations(anns, image_ids, save_path):
        filtered_images = [img for img in coco_data["images"] if img["id"] in image_ids]
        filtered_data = {
            "images": filtered_images,
            "annotations": anns,
            "categories": coco_data["categories"]
        }
        with open(save_path, "w") as f:
            json.dump(filtered_data, f, indent=4)

    save_annotations(support_annotations, support_image_ids, os.path.join(support_dir, "annotations.json"))
    save_annotations(query_annotations, query_image_ids, os.path.join(query_dir, "annotations.json"))

    print(f"Support set: {len(support_annotations)} bounding boxes across {len(support_image_ids)} images")
    print(f"Query set: {len(query_annotations)} bounding boxes across {len(query_image_ids)} images")


def main():
    parser = argparse.ArgumentParser(description="Split COCO dataset into Few-Shot dataset")
    
    parser.add_argument("--annotations", 
                        default="DATASET/milcivil/train/_annotations.coco.json", 
                        help="Path to COCO annotations JSON file")
    
    parser.add_argument("--images_dir", 
                        default="DATASET/milcivil/train",
                        help="Path to COCO images directory")
    
    parser.add_argument("--shot", 
                        default=3,
                        type=int,  
                        help="Number of images for support set per class")
    
    parser.add_argument("--output_dir", 
                        default="DATASET/milcivil_fewshot",
                        help="Output directory to save the split dataset")
    
    args = parser.parse_args()
    
    split_few_shot_coco(args.annotations, args.images_dir, args.shot, args.output_dir)
    
if __name__ == "__main__":
    main()