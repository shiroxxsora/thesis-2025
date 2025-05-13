import os
import shutil
from pathlib import Path
import json
import logging

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set higher level for libraries like shutil if they become too verbose
logging.getLogger("shutil").setLevel(logging.WARNING)


# --- Helper Functions (find_source_image_path, combine_coco_annotations - unchanged from previous version) ---
def find_source_image_path(source_base_dir, folder_name, original_filename):
    """Tries to find the source image file in common locations."""
    # Path directly in the folder
    path1 = os.path.join(source_base_dir, folder_name, original_filename)
    if os.path.exists(path1):
        return path1
    # Path inside an 'images' subdirectory
    path2 = os.path.join(source_base_dir, folder_name, 'images', original_filename)
    if os.path.exists(path2):
        return path2
    return None # Not found

def combine_coco_annotations(annotations_with_folders):
    """
    Combines multiple COCO annotation dictionaries, handling ID and category merging.

    Args:
        annotations_with_folders (list): A list of tuples, where each tuple is
                                         (annotation_data, folder_name).

    Returns:
        dict: The combined COCO annotation dictionary, or None if input is empty.
              Includes temporary 'folder_name' in images for later processing.
    """
    logging.info(f"Received {len(annotations_with_folders)} annotation sets to combine.")
    if not annotations_with_folders:
        logging.warning("Annotation list is empty!")
        return None

    # --- Initialization ---
    # Use info/licenses from the first file as a base
    base_info = annotations_with_folders[0][0].get("info", {})
    base_licenses = annotations_with_folders[0][0].get("licenses", [])

    combined = {
        "info": base_info,
        "licenses": base_licenses,
        "categories": [],
        "images": [],
        "annotations": []
    }

    # --- Category Merging ---
    # category_name -> {new_id, category_dict}
    merged_categories_map = {}
    # (folder_name, old_cat_id) -> new_cat_id
    category_id_remapping = {}
    new_category_id_counter = 1

    logging.info("Starting category merging...")
    for ann_data, folder_name in annotations_with_folders:
        for category in ann_data.get("categories", []):
            original_cat_id = category["id"]
            category_name = category["name"]
            mapping_key = (folder_name, original_cat_id)

            if category_name not in merged_categories_map:
                # New category encountered
                new_cat_info = category.copy()
                new_cat_info["id"] = new_category_id_counter
                merged_categories_map[category_name] = {
                    "new_id": new_category_id_counter,
                    "category_dict": new_cat_info
                }
                category_id_remapping[mapping_key] = new_category_id_counter
                logging.debug(f"  New category '{category_name}' added with ID {new_category_id_counter}")
                new_category_id_counter += 1
            else:
                # Existing category, just map the old ID from this file to the new global ID
                existing_new_id = merged_categories_map[category_name]["new_id"]
                category_id_remapping[mapping_key] = existing_new_id
                logging.debug(f"  Mapping category '{category_name}' (Folder: {folder_name}, Old ID: {original_cat_id}) to New ID: {existing_new_id}")

    # Populate the combined categories list
    combined["categories"] = sorted(
        [cat_info["category_dict"] for cat_info in merged_categories_map.values()],
        key=lambda x: x["id"]
    )
    logging.info(f"Finished category merging. Total unique categories: {len(combined['categories'])}")

    # --- Image and Annotation ID Remapping ---
    # (folder_name, old_image_id) -> new_image_id
    image_id_mapping = {}
    new_image_id_counter = 1
    new_annotation_id_counter = 1

    logging.info("Starting image and annotation merging...")
    for ann_data, folder_name in annotations_with_folders:
        image_map_for_current_file = {} # Track images processed from *this* file

        # Process Images
        for img in ann_data.get("images", []):
            original_img_id = img["id"]
            img_mapping_key = (folder_name, original_img_id)

            if img_mapping_key not in image_id_mapping:
                new_img = img.copy()
                new_img["id"] = new_image_id_counter
                # Store folder_name temporarily for image path resolution later
                new_img["folder_name"] = folder_name
                combined["images"].append(new_img)
                image_id_mapping[img_mapping_key] = new_image_id_counter
                image_map_for_current_file[original_img_id] = new_image_id_counter
                new_image_id_counter += 1
            else:
                 # Image might be duplicated if JSONs overlap, map it anyway for annotations
                 image_map_for_current_file[original_img_id] = image_id_mapping[img_mapping_key]


        # Process Annotations
        for ann_item in ann_data.get("annotations", []):
            original_img_id = ann_item["image_id"]
            original_cat_id = ann_item["category_id"]

            img_mapping_key = (folder_name, original_img_id)
            cat_mapping_key = (folder_name, original_cat_id)

            # Check if the image this annotation belongs to was processed from this file/folder context
            if original_img_id in image_map_for_current_file:
                new_image_id = image_map_for_current_file[original_img_id]

                # Check if the category mapping exists
                if cat_mapping_key in category_id_remapping:
                    new_cat_id = category_id_remapping[cat_mapping_key]

                    new_ann = ann_item.copy()
                    new_ann["id"] = new_annotation_id_counter
                    new_ann["image_id"] = new_image_id
                    new_ann["category_id"] = new_cat_id
                    combined["annotations"].append(new_ann)
                    new_annotation_id_counter += 1
                else:
                    logging.warning(f"  Skipping annotation ID {ann_item.get('id', 'N/A')} for image ID {original_img_id} (Folder: {folder_name}): Category ID {original_cat_id} not found in category map for this folder.")
            else:
                # This can happen if an annotation references an image not listed in its *own* JSON's "images" section
                 logging.warning(f"  Skipping annotation ID {ann_item.get('id', 'N/A')} (Folder: {folder_name}): Corresponding image ID {original_img_id} was not found or mapped within its own annotation file context.")


    logging.info(f"Combined images: {len(combined['images'])}")
    logging.info(f"Combined annotations: {len(combined['annotations'])}")
    return combined


# --- Main Combining Function (Corrected) ---
def combine_dataset(source_dir, target_dir, preserve_names=False):
    """
    Combines multiple COCO datasets from subfolders into a single dataset.

    Args:
        source_dir (str): Path to the directory containing dataset subfolders.
        target_dir (str): Path to the directory where the combined dataset will be saved.
        preserve_names (bool): If True, prefixes image filenames with their source
                               folder name. If False, renames images using their
                               new unique ID (e.g., image_1.jpg).
    """
    try:
        # --- Setup Directories ---
        target_images_dir = os.path.join(target_dir, 'images')
        os.makedirs(target_images_dir, exist_ok=True)
        logging.info(f"Target directory prepared: {target_dir}")
        logging.info(f"Target images directory prepared: {target_images_dir}")

        # --- Collect Annotations ---
        all_annotations_with_folders = []
        subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]
        logging.info(f"Found {len(subfolders)} potential dataset subfolders.")

        for subfolder in subfolders:
            folder_name = os.path.basename(subfolder)
            annotations_subdir = os.path.join(subfolder, 'annotations')
            if not (os.path.exists(annotations_subdir) and os.path.isdir(annotations_subdir)):
                 logging.debug(f"Skipping folder '{folder_name}': No 'annotations' subdirectory found.")
                 continue

            logging.info(f"Processing annotations in folder: {folder_name}")
            found_json = False
            for json_file in os.listdir(annotations_subdir):
                if json_file.lower().endswith('.json'):
                    json_path = os.path.join(annotations_subdir, json_file)
                    logging.info(f"  Reading annotation file: {json_file}")
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            annotations_data = json.load(f)
                        # Validate basic COCO structure (presence of images/annotations keys)
                        if "images" in annotations_data and "annotations" in annotations_data:
                             all_annotations_with_folders.append((annotations_data, folder_name))
                             found_json = True
                             logging.debug(f"    Successfully loaded and added {json_file}")
                        else:
                             logging.warning(f"    Skipping {json_file}: Missing 'images' or 'annotations' key.")
                    except json.JSONDecodeError as e:
                        logging.error(f"    Error reading JSON file {json_path}: {e}")
                    except Exception as e:
                        logging.error(f"    Unexpected error reading file {json_path}: {e}")
            if not found_json:
                 logging.warning(f"  No valid .json annotation files found in {annotations_subdir}")


        # --- Combine Annotations ---
        if not all_annotations_with_folders:
            logging.error("No valid annotation files found in any subfolder. Cannot proceed.")
            return

        combined_annotations = combine_coco_annotations(all_annotations_with_folders)

        if not combined_annotations:
            logging.error("Failed to combine annotations.")
            return

        # --- Process and Copy Images ---
        logging.info("Processing and copying images...")
        files_info = {} # Stores mapping from new filename to original info
        images_copied_count = 0
        images_missing_count = 0

        # Process the actual list that's part of combined_annotations
        images_to_process = combined_annotations.get("images", [])

        for img_data in images_to_process: # Iterate directly over the list in the dict
            original_file_name = img_data.get("file_name")
            # Get the temporary folder_name we stored
            folder_name = img_data.get("folder_name")
            new_image_id = img_data.get("id") # Already the unique combined ID

            if not original_file_name or not folder_name:
                logging.warning(f"  Skipping image entry with missing 'file_name' or 'folder_name': {img_data}")
                continue

            # Find the source image file
            source_path = find_source_image_path(source_dir, folder_name, original_file_name)

            if source_path:
                # Determine target filename
                file_extension = Path(original_file_name).suffix
                if preserve_names:
                    target_filename = f"{folder_name}_{original_file_name}"
                else:
                    # Use the new unique image ID for the filename
                    target_filename = f"image_{new_image_id}{file_extension}"

                target_path = os.path.join(target_images_dir, target_filename)

                try:
                    shutil.copy2(source_path, target_path)
                    logging.debug(f"  Copied: {source_path} -> {target_path}")
                    images_copied_count += 1

                    # --- !!! THIS IS THE FIX !!! ---
                    # Update the file_name within the combined_annotations dictionary
                    # to match the actual name of the copied file.
                    img_data["file_name"] = target_filename
                    # ---------------------------------

                    # Store mapping info
                    files_info[target_filename] = {
                        'original_full_path': source_path,
                        'original_name': original_file_name,
                        'source_folder': folder_name,
                        'new_image_id': new_image_id,
                    }

                except Exception as e:
                    logging.error(f"  Failed to copy {source_path} to {target_path}: {e}")
                    # If copy fails, file_name in img_data remains original,
                    # which is consistent since the target file wasn't created.
            else:
                logging.warning(f"  Source image not found for '{original_file_name}' in folder '{folder_name}'. Searched in '{os.path.join(source_dir, folder_name)}' and '{os.path.join(source_dir, folder_name, 'images')}'. This image entry will remain in annotations but the file is missing.")
                images_missing_count += 1

            # Clean up the temporary folder_name key from the image entry
            if "folder_name" in img_data:
                del img_data["folder_name"]

        logging.info(f"Finished image processing. Copied: {images_copied_count}, Missing source files: {images_missing_count}")

        # --- Final Cleanup and Save ---
        # Remove temporary folder_name from annotations as well (though it shouldn't be there)
        for ann_item in combined_annotations.get("annotations", []):
            if "folder_name" in ann_item:
                del ann_item["folder_name"] # Should not exist here, but just in case

        # Save combined annotations
        annotations_file = os.path.join(target_dir, 'annotations.json') # Standard name
        logging.info(f"Saving combined annotations to {annotations_file}...")
        try:
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(combined_annotations, f, ensure_ascii=False, indent=4)
            logging.info("Combined annotations saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save combined annotations: {e}")

        # Save file mapping info
        info_file = os.path.join(target_dir, 'files_info.json')
        logging.info(f"Saving file mapping info to {info_file}...")
        try:
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(files_info, f, ensure_ascii=False, indent=4)
            logging.info("File mapping info saved successfully.")
        except Exception as e:
             logging.error(f"Failed to save file mapping info: {e}")

        logging.info("\nDataset combination process finished!")
        logging.info(f"Total images combined: {len(combined_annotations.get('images',[]))}")
        logging.info(f"Total annotations combined: {len(combined_annotations.get('annotations',[]))}")
        logging.info(f"Total unique categories: {len(combined_annotations.get('categories',[]))}")

    except Exception as e:
        logging.exception(f"An critical error occurred during the dataset combination: {e}")


# --- Main Execution (Unchanged) ---
if __name__ == "__main__":
    # --- Configuration ---
    # !!! ADJUST THESE PATHS ACCORDING TO YOUR STRUCTURE !!!
    SOURCE_DATASET_DIR = "Dataset"  # Directory containing subfolders like Dataset/Set1, Dataset/Set2 etc.
    TARGET_COMBINED_DIR = "Combined_Dataset" # Directory where the combined dataset will be created
    PRESERVE_ORIGINAL_NAMES = True # Set to False to rename images like image_1.jpg, image_2.png etc.

    # --- Run ---
    if not os.path.isdir(SOURCE_DATASET_DIR):
         logging.error(f"Source directory '{SOURCE_DATASET_DIR}' not found. Please check the path.")
    else:
        combine_dataset(
             source_dir=SOURCE_DATASET_DIR,
             target_dir=TARGET_COMBINED_DIR,
             preserve_names=PRESERVE_ORIGINAL_NAMES
        )