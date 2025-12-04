import os
import pandas as pd
from clairvoyance.params import CLASSES

def convert_to_yolo(data_type='train'):
    """
    Convert CSV annotations to YOLO format (.txt files)
    """
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root_path, 'raw_data', data_type)
    csv_path = os.path.join(data_path, '_annotations.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"ℹ️ Columns found: {df.columns.tolist()}")
    
    # Class mapping
    class_to_int = {c: i for i, c in enumerate(CLASSES)}
    
    # Create labels directory if not exists (YOLO expects images and labels, usually in parallel folders, 
    # but for simplicity we can put .txt next to .jpg or in a 'labels' subfolder. 
    # Standard YOLO structure: datasets/images/train and datasets/labels/train.
    # We will try to adapt to the current structure or move files.
    # Let's keep it simple: create .txt files next to images first.)
    
    count = 0
    for index, row in df.iterrows():
        filename = row['filename']
        img_width = row['width']
        img_height = row['height']
        cls = row['class']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        
        # Normalize coordinates
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        class_id = class_to_int.get(cls)
        if class_id is None:
            continue
            
        # YOLO format: class_id x_center y_center width height
        yolo_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
        
        # Save to .txt file (same name as image)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(data_path, txt_filename)
        
        # Append if file exists (multiple objects per image)
        mode = 'a' if os.path.exists(txt_path) else 'w'
        with open(txt_path, mode) as f:
            f.write(yolo_line)
            
        count += 1
        
    print(f"✅ Converted {count} annotations for {data_type}")

if __name__ == '__main__':
    convert_to_yolo('train')
    convert_to_yolo('valid')
    convert_to_yolo('test')
