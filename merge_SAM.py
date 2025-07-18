import os
import shutil
import xml.etree.ElementTree as ET
from glob import glob
from collections import defaultdict

# Paths
IMAGE_DIR = "/home/hpc/tovl/tovl104v/traco_ankilab/batch_processed_dataset/images_SAM_Neelima"
ANNOTATION_DIR = "/home/hpc/tovl/tovl104v/traco_ankilab/batch_processed_dataset/annotations_SAM_Neelima"
MERGED_IMAGE_DIR = "/home/hpc/tovl/tovl104v/traco_ankilab/batch_processed_dataset/merged_label_images"
MERGED_ANN_DIR = "/home/hpc/tovl/tovl104v/traco_ankilab/batch_processed_dataset/merged_label_annotations"

os.makedirs(MERGED_IMAGE_DIR, exist_ok=True)
os.makedirs(MERGED_ANN_DIR, exist_ok=True)

# Extract frame prefix (e.g., training098_frame0032)
def get_frame_prefix(filename):
    parts = filename.split("_")
    return "_".join(parts[:2])  # training098_frame0032

# Group XMLs by frame
xml_files = glob(os.path.join(ANNOTATION_DIR, "*.xml"))
frame_groups = defaultdict(list)

for xml_file in xml_files:
    basename = os.path.basename(xml_file)
    frame_id = get_frame_prefix(basename)
    frame_groups[frame_id].append(xml_file)

print(f"🔎 Found {len(frame_groups)} unique frames")

for frame_id, xml_list in frame_groups.items():
    merged_root = None
    merged_tree = None
    bug_objects = []

    for idx, xml_path in enumerate(sorted(xml_list)):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            obj = root.find("object")
            if obj is not None:
                name_tag = obj.find("name")
                if name_tag is not None:
                    name_tag.text = f"hexbug{idx}"  # Rename to hexbug0, hexbug1, ...
                bug_objects.append(obj)

            if merged_tree is None:
                merged_tree = tree
                merged_root = merged_tree.getroot()

                # Remove all objects to prepare for clean merge
                for obj in merged_root.findall("object"):
                    merged_root.remove(obj)

        except Exception as e:
            print(f"⚠️ Failed to parse {xml_path}: {e}")
            continue

    # Add all renamed objects
    for obj in bug_objects:
        merged_root.append(obj)

    # Fix filename in XML
    image_filename = f"{frame_id}.jpg"
    if merged_root.find("filename") is not None:
        merged_root.find("filename").text = image_filename

    # Save merged annotation
    merged_xml_path = os.path.join(MERGED_ANN_DIR, f"{frame_id}.xml")
    merged_tree.write(merged_xml_path)

    # Save one corresponding image
    image_candidates = glob(os.path.join(IMAGE_DIR, f"{frame_id}_hexbug*.jpg"))
    if image_candidates:
        shutil.copy(image_candidates[0], os.path.join(MERGED_IMAGE_DIR, image_filename))

print(f"\n✅ Merged annotations saved to: {MERGED_ANN_DIR}")
print(f"✅ Merged images saved to:      {MERGED_IMAGE_DIR}")
