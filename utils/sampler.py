import json
import os
import random
import shutil

from pycocotools.coco import COCO


class COCOSampler:
    def __init__(self, original_annotation_file, output_directory):
        self.original_annotation_file = original_annotation_file
        self.output_directory = output_directory
        self.coco = COCO(original_annotation_file)

    def random_sample(self, target_size_mb):
        filtered_annotations = []
        filtered_images = []
        total_size_mb = 0
        num_sampled_images = 0

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)

        image_ids = self.coco.getImgIds()

        random.shuffle(image_ids)

        for image_id in image_ids:
            # Get annotations for the image
            annotations_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(annotations_ids)
            filtered_annotations.extend(annotations)

            # Get image information
            image_info = self.coco.loadImgs(image_id)[0]
            filtered_images.append(image_info)

            # Construct the original image path
            original_image_path = os.path.join(
                "data", "cadis", "training", image_info["file_name"]
            )

            image_size_mb = os.path.getsize(original_image_path) / (1024 * 1024)

            if total_size_mb + image_size_mb > target_size_mb:
                break

            # Construct the new image path in the output directory
            image_filename = os.path.basename(original_image_path)
            new_image_path = os.path.join(self.output_directory, image_filename)

            # Copy the image to the output directory
            shutil.copy(original_image_path, new_image_path)

            # Update the total size
            total_size_mb += image_size_mb
            num_sampled_images += 1

        # Create a new COCO-style JSON object containing only the filtered information
        filtered_coco_data = {
            "info": self.coco.dataset["info"],
            "licenses": self.coco.dataset["licenses"],
            "images": filtered_images,
            "annotations": filtered_annotations,
            "categories": self.coco.dataset["categories"],
        }

        # Save the filtered COCO-style JSON object to the output directory
        filtered_coco_annotation_file = os.path.join(
            self.output_directory, "filtered_coco_json_file.json"
        )
        with open(filtered_coco_annotation_file, "w") as f:
            json.dump(filtered_coco_data, f)

        # Print the total size of sampled images and the number of sampled images
        print("Total size of sampled images:", total_size_mb, "MB")
        print("Number of sampled images:", num_sampled_images)
