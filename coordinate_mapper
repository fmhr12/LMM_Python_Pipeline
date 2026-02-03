import cv2
import matplotlib.pyplot as plt
import re

def parse_ai_output(text):
    """
    Parses the raw output string from the AI.
    Handles the new format: [Confidence, ymin, xmin, ymax, xmax]
    """
    boxes = []
    # Regex to capture content inside square brackets
    matches = re.findall(r"\[([0-9\.,\s]+)\]", text)

    for match in matches:
        try:
            # Convert the string of numbers into a list of floats
            vals = [float(x.strip()) for x in match.split(',')]
            
            # --- NEW LOGIC FOR 5-ITEM LIST ---
            if len(vals) == 5:
                # Format: [Score, ymin, xmin, ymax, xmax]
                score = vals[0]
                # Extract coords and convert to int
                coords = [int(vals[1]), int(vals[2]), int(vals[3]), int(vals[4])]
                boxes.append({"score": score, "coords": coords})
                
            # --- FALLBACK FOR OLD 4-ITEM LIST ---
            elif len(vals) == 4:
                # Format: [ymin, xmin, ymax, xmax]
                coords = [int(x) for x in vals]
                boxes.append({"score": None, "coords": coords})
                
        except Exception as e:
            print(f"Skipping malformed line: {match}")
            continue
            
    return boxes

def draw_boxes_on_image(img, boxes_data):
    """
    Draws bounding boxes and confidence scores on the image.
    """
    img_copy = img.copy()
    h, w, _ = img_copy.shape
    
    for item in boxes_data:
        # Get coordinates [ymin, xmin, ymax, xmax] (Normalized 0-1000)
        ymin_n, xmin_n, ymax_n, xmax_n = item['coords']
        
        # Denormalize to actual pixel values
        xmin = int((xmin_n / 1000) * w)
        xmax = int((xmax_n / 1000) * w)
        ymin = int((ymin_n / 1000) * h)
        ymax = int((ymax_n / 1000) * h)

        # Draw Rectangle (Green, thickness 2)
        cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Draw Score Label (if score exists)
        if item['score'] is not None:
            label = f"{item['score']:.2f}"
            # Draw a small background for text readability (optional)
            cv2.rectangle(img_copy, (xmin, ymin - 20), (xmin + 40, ymin), (0, 255, 0), -1)
            cv2.putText(img_copy, label, (xmin + 2, ymin - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
    return img_copy

# 1. Load your image
# Make sure to update the path to your actual image file
img_path = 'Path'
img = cv2.imread(img_path)

# 2. PASTE THE RAW OUTPUT FROM THE MODEL
# You can paste the whole block like ">>> DETECTED: [0.95, ...]"
raw_ai_string = """
>>> DETECTED: [0.75, 150, 200, 190, 240] # Example
"""

if img is None:
    print("Error: Image not found at path:", img_path)
else:
    # 3. Parse
    formatted_data = parse_ai_output(raw_ai_string)
    print(f"Found {len(formatted_data)} boxes.")

    # 4. Draw
    result_image = draw_boxes_on_image(img, formatted_data)

    # 5. Display
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
