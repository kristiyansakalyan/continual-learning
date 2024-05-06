import json
import os
import random
import shutil

from pycocotools.coco import COCO


class COCOUtils:
    def __init__(self, original_annotation_file, output_directory):
        self.original_annotation_file = original_annotation_file
        self.output_directory = output_directory
        self.coco = COCO(original_annotation_file)

    def random_sample(self, target_size_mb, directory_path):
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
            original_image_path = os.path.join(directory_path, image_info["file_name"])

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

    @staticmethod
    def merge_coco_jsons(main_coco, sampled_coco, output_dir=None):
        if output_dir is None:
            output_dir = os.getcwd()

        coco1 = COCO(main_coco)
        coco2 = COCO(sampled_coco)

        # Find the maximum IDs used in the first JSON file
        max_image_id_1 = max(image["id"] for image in coco1.dataset["images"])
        max_annotation_id_1 = max(
            annotation["id"] for annotation in coco1.dataset["annotations"]
        )

        # Increment IDs in the second JSON file to avoid conflicts
        for image in coco2.dataset["images"]:
            image["id"] += max_image_id_1 + 1

        for annotation in coco2.dataset["annotations"]:
            annotation["id"] += max_annotation_id_1 + 1
            annotation["image_id"] += max_image_id_1 + 1

        merged_categories = coco1.dataset["categories"] + coco2.dataset["categories"]
        seen = set()
        unique_merged_categories = []

        for item in merged_categories:
            # Create a tuple of id and name to check for uniqueness
            key = (item["id"], item["name"])
            if key not in seen:
                unique_merged_categories.append(item)
                seen.add(key)

        # Merge annotations, images, categories, info, and licenses
        merged_annotations = coco1.dataset["annotations"] + coco2.dataset["annotations"]
        merged_images = coco1.dataset["images"] + coco2.dataset["images"]
        merged_info = coco1.dataset["info"]
        merged_licenses = coco1.dataset["licenses"]

        merged_coco_data = {
            "info": merged_info,
            "licenses": merged_licenses,
            "images": merged_images,
            "annotations": merged_annotations,
            "categories": unique_merged_categories,
        }

        # Save merged COCO JSON to output directory
        output_file = os.path.join(output_dir, "merged_coco.json")
        with open(output_file, "w") as f:
            json.dump(merged_coco_data, f, indent=4)

        return output_file
