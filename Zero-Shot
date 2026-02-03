import os
import json
import mimetypes
import random
import re
import time
from pathlib import Path
from google import genai
from google.genai import types

# --- CONFIGURATION ---
PROJECT_ID  = "project_ID"        
LOCATION    = "global"                      

# 1. INPUT FOLDERS
RAW_DIR   = Path('path')
YOLO_DIR  = Path('path')

# 2. CHECKPOINT FILE
CHECKPOINT_PATH = Path("scan_progress_checkpoint.json")

# --- MODEL ---
MODEL_NAME = "gemini-3-pro-preview" 

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- PROMPT: OCCLUSAL (ZERO-SHOT REVISION) ---
PROMPT = """
One of the prompts from Appendix 1A or 2A
"""

# --- HELPER FUNCTIONS ---

def get_file_content(filepath, mime_type=None):
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type is None: mime_type = "application/octet-stream"
    return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

def call_gemini(prompt, image_path):
    # Construct content: Prompt -> Target Image (No References)
    contents = [prompt]
    contents.append(get_file_content(image_path))
    
    config = types.GenerateContentConfig(
        response_mime_type="text/plain", 
        safety_settings=[types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE")]
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
            return response.text if response.text else ""
        except Exception as e:
            time.sleep(5)
            if attempt == 2: return None
    return None

def parse_streamed_coordinates(text):
    if text is None: return []
    predictions = []
    # Match list containing decimals or integers
    matches = re.findall(r">>> DETECTED:\s*(\[[0-9.,\s]+\])", text)
    for match in matches:
        try:
            vals = json.loads(match)
            # CASE 1: New Format [score, y, x, y, x]
            if len(vals) == 5:
                predictions.append({
                    "score": float(vals[0]),       # First item is score
                    "box_2d": vals[1:]             # Remaining 4 are coords
                })
            # CASE 2: Fallback (Model ignored instructions and did old format)
            # We assume if the first number is > 1.0, it's a coordinate, not a score.
            elif len(vals) == 4:
                predictions.append({
                    "score": 1.0, # Default confidence if missing
                    "box_2d": vals
                })
        except: continue
    return predictions

def parse_yolo_file(yolo_path):
    gts = []
    try:
        with open(yolo_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x_c, y_c, w, h = map(float, parts[1:5])
                    xmin = int((x_c - w/2) * 1000); xmax = int((x_c + w/2) * 1000)
                    ymin = int((y_c - h/2) * 1000); ymax = int((y_c + h/2) * 1000)
                    gts.append({"box_2d": [max(0, ymin), max(0, xmin), min(1000, ymax), min(1000, xmax)]})
    except: pass
    return gts

# --- CHECKPOINT FUNCTIONS ---

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                data = json.load(f)
            if isinstance(data['processed_files'], list):
                temp_dict = {filename: {} for filename in data['processed_files']}
                data['processed_files'] = temp_dict
            
            # Ensure mAP/mIoU fields exist
            if 'all_predictions' not in data: data['all_predictions'] = []
            if 'iou_sum' not in data['stats']: data['stats']['iou_sum'] = 0.0

            print(f"✅ Checkpoint loaded! Resuming from {len(data['processed_files'])} processed images.")
            return data
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}. Starting fresh.")
    
    return {
        "processed_files": {}, 
        "stats": {
            "tooth_tp": 0, "tooth_fp": 0, "tooth_fn": 0,
            "img_tp": 0, "img_tn": 0, "img_fp": 0, "img_fn": 0,
            "iou_sum": 0.0
        },
        "all_predictions": [] 
    }

def save_checkpoint(data):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(data, f, indent=4)

# --- METRIC CALCULATION FUNCTIONS ---

def calculate_iou(box1, box2):
    y_top = max(box1[0], box2[0]); x_left = max(box1[1], box2[1])
    y_bottom = min(box1[2], box2[2]); x_right = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = ((box1[3]-box1[1])*(box1[2]-box1[0])) + ((box2[3]-box2[1])*(box2[2]-box2[0])) - intersection
    return intersection / union if union > 0 else 0.0

def compute_matches(preds, gts):
    # 1. Sort Predictions by Confidence (High -> Low)
    preds.sort(key=lambda x: x['score'], reverse=True)
    
    tp = []; fp = []; fn = []
    matched_gt = set()
    
    for pred in preds:
        best_iou = 0; best_idx = -1
        for i, gt in enumerate(gts):
            if i in matched_gt: continue
            iou = calculate_iou(pred['box_2d'], gt['box_2d'])
            if iou > best_iou: best_iou = iou; best_idx = i
            
        if best_iou >= 0.5: 
            tp.append((pred, best_iou)) # Store tuple (pred, iou)
            matched_gt.add(best_idx)
        else: 
            fp.append(pred)
            
    for i, gt in enumerate(gts):
        if i not in matched_gt: fn.append(gt)
    return tp, fp, fn

def calculate_map(all_preds, total_gt_count):
    """
    Calculates Mean Average Precision (mAP) using Standard Interpolation.
    """
    if total_gt_count == 0: return 0.0
    if not all_preds: return 0.0
    
    # 1. Sort all predictions by confidence (High -> Low)
    all_preds.sort(key=lambda x: x['score'], reverse=True)
    
    # 2. distinct TP/FP lists for cumulative summing
    tps = [1 if p['is_tp'] else 0 for p in all_preds]
    fps = [1 if not p['is_tp'] else 0 for p in all_preds]
    
    # 3. Calculate Cumulative Sums
    for i in range(1, len(tps)):
        tps[i] += tps[i-1]
        fps[i] += fps[i-1]
        
    # 4. Calculate Precision and Recall arrays
    precisions = [tp / (tp + fp) for tp, fp in zip(tps, fps)]
    recalls    = [tp / total_gt_count for tp in tps]

    # 5. Add "Sentinel" values to handle edge cases
    precisions = [0.0] + precisions + [0.0]
    recalls    = [0.0] + recalls + [1.0]

    # 6. SMOOTHING (The "Interpolation" Step)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 7. Calculate Area Under Curve (AUC)
    ap = 0.0
    for i in range(1, len(recalls)):
        recall_change = recalls[i] - recalls[i-1]
        if recall_change > 0:
            ap += recall_change * precisions[i]
            
    return ap

def print_performance_report(stats, count, all_preds=None):
    tooth_tp = stats['tooth_tp']; tooth_fp = stats['tooth_fp']; tooth_fn = stats['tooth_fn']
    img_tp = stats['img_tp']; img_tn = stats['img_tn']; img_fp = stats['img_fp']; img_fn = stats['img_fn']
    iou_sum = stats.get('iou_sum', 0.0)

    # Standard Metrics
    t_prec = tooth_tp/(tooth_tp+tooth_fp) if (tooth_tp+tooth_fp)>0 else 0
    t_rec = tooth_tp/(tooth_tp+tooth_fn) if (tooth_tp+tooth_fn)>0 else 0
    t_f1 = 2*(t_prec*t_rec)/(t_prec+t_rec) if (t_prec+t_rec)>0 else 0
    
    # NEW METRICS
    m_iou = iou_sum / tooth_tp if tooth_tp > 0 else 0.0
    
    map_50 = 0.0
    if all_preds:
        total_gt = tooth_tp + tooth_fn
        map_50 = calculate_map(all_preds, total_gt)

    # Image Level
    img_tot = img_tp+img_tn+img_fp+img_fn
    i_acc = (img_tp+img_tn)/img_tot if img_tot>0 else 0
    i_prec = img_tp/(img_tp+img_fp) if (img_tp+img_fp)>0 else 0
    i_rec = img_tp/(img_tp+img_fn) if (img_tp+img_fn)>0 else 0
    i_spec = img_tn/(img_tn+img_fp) if (img_tn+img_fp)>0 else 0
    i_f1 = 2*(i_prec*i_rec)/(i_prec+i_rec) if (i_prec+i_rec)>0 else 0
    
    print(f"\n" + "="*50)
    print(f"   PERFORMANCE REPORT (Total Processed: {count})")
    print("="*50)
    print(f"--- [A] TOOTH/OBJECT LEVEL ---")
    print(f"TP: {tooth_tp} | FP: {tooth_fp} | FN: {tooth_fn}")
    print(f"Precision: {t_prec:.4f}")
    print(f"Recall:    {t_rec:.4f}")
    print(f"F1 Score:  {t_f1:.4f}")
    print(f"mIoU:      {m_iou:.4f} (Mean IoU of True Positives)")
    print(f"mAP@50:    {map_50:.4f}")
    
    print(f"\n--- [B] IMAGE/DIAGNOSTIC LEVEL ---")
    print(f"TP: {img_tp} | TN: {img_tn} | FP: {img_fp} | FN: {img_fn}")
    print(f"Accuracy:    {i_acc:.4f}")
    print(f"Sensitivity: {i_rec:.4f}")
    print(f"Specificity: {i_spec:.4f}")
    print(f"Precision:   {i_prec:.4f}")
    print(f"F1 Score:    {i_f1:.4f}")
    print("="*50 + "\n")

# --- MAIN EXECUTION ---
if not RAW_DIR.exists(): print("Error: RAW_DIR not found"); exit()

# 1. LOAD CHECKPOINT
data = load_checkpoint()

# 2. PREPARE FILE LIST
all_files = [f for f in RAW_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg'] and not f.name.startswith('.')]
random.shuffle(all_files)

files_to_process = [f for f in all_files if f.name not in data['processed_files']]

print(f"--- STARTING OCCLUSAL ANALYSIS (ZERO-SHOT) ---")
print(f"Total found: {len(all_files)}")
print(f"Already done: {len(data['processed_files'])}")
print(f"Remaining: {len(files_to_process)}")

for i, raw_file in enumerate(files_to_process):
    print(f"[{i+1}/{len(files_to_process)}] Processing {raw_file.name}...")
    
    # PREDICT (Zero-Shot call)
    pred_text = call_gemini(PROMPT_OCCLUSAL, raw_file)
    preds = parse_streamed_coordinates(pred_text)
    
    if preds:
        print(f"   -> Model Predicted {len(preds)} boxes")
        for p in preds:
            print(f"      Score: {p['score']} | Coords: {p['box_2d']}")
    else:
        print("   -> Model Predicted: Clean")
    
    # GT
    yolo_file = YOLO_DIR / (raw_file.stem + ".txt")
    gts = parse_yolo_file(yolo_file) if yolo_file.exists() else []
    
    # UPDATE STATS
    tp_data, fp_boxes, fn_boxes = compute_matches(preds, gts)
    
    # Update Stats
    data['stats']['tooth_tp'] += len(tp_data)
    data['stats']['tooth_fp'] += len(fp_boxes)
    data['stats']['tooth_fn'] += len(fn_boxes)
    
    # Add IoU Sum
    data['stats']['iou_sum'] += sum([x[1] for x in tp_data])

    # Save Predictions for mAP Calculation
    for x in tp_data: 
        data['all_predictions'].append({'score': x[0]['score'], 'is_tp': True})
    for p in fp_boxes: 
        data['all_predictions'].append({'score': p['score'], 'is_tp': False})

    # --- REVISED IMAGE LEVEL LOGIC ---
    img_status = ""
    has_gt = len(gts) > 0
    
    # "Correct Prediction": At least one prediction matched a GT with IoU >= 0.5
    has_correct_pred = len(tp_data) > 0 
    
    # "Any Prediction": The model flagged something, even if location was wrong
    has_any_pred = len(preds) > 0

    if has_gt:
        if has_correct_pred:
            # 1. Image has caries AND Model found it in the right spot
            data['stats']['img_tp'] += 1
            img_status = "TP"
        elif has_any_pred:
            # 2. Image has caries, Model flagged caries, BUT in the wrong spot.
            # STRICT STANDARD: Count as False Negative because it missed the real lesion.
            data['stats']['img_fn'] += 1
            img_status = "FN (Wrong Location)"
        else:
            # 3. Image has caries, Model said "Clean"
            data['stats']['img_fn'] += 1
            img_status = "FN"
    else:
        if has_any_pred:
            # 4. Image is Healthy, Model flagged caries
            data['stats']['img_fp'] += 1
            img_status = "FP"
        else:
            # 5. Image is Healthy, Model said "Clean"
            data['stats']['img_tn'] += 1
            img_status = "TN"

    print(f"   -> Match Stats: TP={len(tp_data)} FP={len(fp_boxes)} FN={len(fn_boxes)} (Image: {img_status})")

    # MARK AS DONE & SAVE DETECTION DATA
    data['processed_files'][raw_file.name] = {
        "box_tp": len(tp_data),
        "box_fp": len(fp_boxes),
        "box_fn": len(fn_boxes),
        "image_classification": img_status, # Saves the detailed status
        "detections": preds
    }
    
    # SAVE CHECKPOINT & REPORT (every 5 images)
    if (i+1) % 5 == 0:
        save_checkpoint(data)
        print("   [Checkpoint Saved]")
        print_performance_report(data['stats'], len(data['processed_files']), data['all_predictions'])

# FINAL SAVE
save_checkpoint(data)
print("\n" + "#"*60)
print("PROCESSING COMPLETE. FINAL STATISTICS:")
print("#"*60)
print_performance_report(data['stats'], len(data['processed_files']), data['all_predictions'])
