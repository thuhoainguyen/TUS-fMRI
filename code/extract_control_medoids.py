#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract control session medoid positions for Left and Right hemispheres.

@author Hoai Thu Nguyen
"""

import os
import glob
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("extract_control_medoids")

def extract_medoids() -> None:
    # Define directories
    base_dir = Path(__file__).resolve().parent.parent
    actual_dir = base_dir / "data" / "gum" / "actual"
    output_dir = base_dir / "data" / "gum" / "medoid"
    
    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping of CON medoids for each subject
    # Format: subject_id -> { 'L': left_medoid_index, 'R': right_medoid_index }
    con_medoids = {
        "sub-03": {"L": 118, "R": 19},
        "sub-04": {"L": 20, "R": 159},
        "sub-05": {"L": 36, "R": 108},
        "sub-06": {"L": 28, "R": 155},
        "sub-11": {"L": 98, "R": 42},
    }
    
    for sub_id, indices in con_medoids.items():
        log.info(f"Processing {sub_id}...")
        
        # Locate the ses-con XML file in actual_dir/sub_id/
        pattern = str(actual_dir / sub_id / f"{sub_id}_ses-con_*_GUMMarkers*.xml")
        xml_files = glob.glob(pattern)
        if not xml_files:
            log.error(f"Could not find ses-con XML file for {sub_id} using pattern: {pattern}")
            continue
            
        xml_path = Path(xml_files[0])
        log.info(f"  Found source file: {xml_path.name}")
        
        # Extract the date suffix from the filename
        # Format of filename: sub-03_ses-con_RL_GUMMarkers20260507.xml
        # The date suffix is the numbers following 'GUMMarkers'
        filename_stem = xml_path.stem
        if "GUMMarkers" in filename_stem:
            parts = filename_stem.split("GUMMarkers")
            date_suffix = parts[-1]
        else:
            date_suffix = "20260602"  # fallback default
            
        l_idx = indices["L"]
        r_idx = indices["R"]
        
        # Parse XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            log.exception(f"  Failed to parse XML file {xml_path.name}: {e}")
            continue
            
        # Filter Element tags
        elements_to_remove = []
        for elem in root.findall("Element"):
            idx_str = elem.attrib.get("index")
            if idx_str is None:
                elements_to_remove.append(elem)
                continue
            try:
                idx_val = int(idx_str)
            except ValueError:
                elements_to_remove.append(elem)
                continue
                
            if idx_val != l_idx and idx_val != r_idx:
                elements_to_remove.append(elem)
                
        for elem in elements_to_remove:
            root.remove(elem)
            
        # Ensure we indent the tree to match the original formatting (2 spaces)
        if hasattr(ET, "indent"):
            ET.indent(root, space="  ")
            
        # Construct the output filename
        # convention: sub-ID_ses-con_pos-medoid-L<L_index>-R<R_index>_GUMMarkers<date>.xml
        out_filename = f"{sub_id}_ses-con_pos-medoid-L{l_idx}-R{r_idx}_GUMMarkers{date_suffix}.xml"
        out_path = output_dir / out_filename
        
        try:
            # Save the file with xml declaration and proper encoding
            tree.write(out_path, encoding="utf-8", xml_declaration=True)
            log.info(f"  Successfully saved filtered medoid file to {out_path.name}")
        except Exception as e:
            log.exception(f"  Failed to write XML file {out_path.name}: {e}")
            
if __name__ == "__main__":
    extract_medoids()
