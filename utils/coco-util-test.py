import json
import os
import shutil
import unittest

from coco_utils import COCOUtils


class TestMergeCOCOJSONs(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test output
        self.test_output_dir = "test_output"
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        # Remove temporary directory and its contents
        shutil.rmtree(self.test_output_dir)

    def test_merge_coco_jsons_new_categories_overlap_ids(self):
        # Sample JSON data for testing
        coco1_data = {
            "info": {},
            "licenses": [],
            "images": [{"id": 1, "file_name": "image1.jpg"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
            "categories": [
                {"id": 1, "name": "category1"},
                {"id": 5, "name": "category2"},
            ],
        }

        coco2_data = {
            "info": {},
            "licenses": [],
            "images": [{"id": 1, "file_name": "image2.jpg"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 2}],
            "categories": [
                {"id": 1, "name": "category1"},
                {"id": 2, "name": "new_category"},
            ],
        }

        # Write sample JSON data to temporary files
        with open("coco1.json", "w") as f:
            json.dump(coco1_data, f)
        with open("coco2.json", "w") as f:
            json.dump(coco2_data, f)

        # Merge the sample JSON files
        output_file = COCOUtils.merge_coco_jsons(
            "coco1.json", "coco2.json", self.test_output_dir
        )

        # Read the merged JSON file
        with open(output_file, "r") as f:
            merged_data = json.load(f)

        # Assert the correctness of the merged JSON data
        self.assertEqual(len(merged_data["images"]), 2)
        self.assertEqual(len(merged_data["annotations"]), 2)
        self.assertEqual(len(merged_data["categories"]), 3)  # One new category added

        print(json.dumps(merged_data, indent=4))

        # Clean up temporary files
        os.remove("coco1.json")
        os.remove("coco2.json")


if __name__ == "__main__":
    unittest.main()
