# LMM Python Pipeline for Object Detection

## Overview
This repository contains a complete Python pipeline for performing automated object detection (such as dental caries screening) using General-Purpose Large Multimodal Models (LMMs). 

Specifically, this pipeline leverages the **Gemini 3 Pro** model via the Google Cloud Vertex AI API to perform image-level classification and bounding-box localization under **zero-shot** and **few-shot** conditions.

## Key Features
* **Automated Inference:** Submits images to Gemini using stateless API calls, extracting structured bounding box coordinates from plain-text outputs.
* **Persistent Checkpointing:** Automatically saves progress to `scan_progress_checkpoint.json`, allowing you to pause and resume large-scale dataset evaluations without losing data.
* **Built-in Evaluation Metrics:** Calculates automated diagnostic metrics against YOLO-formatted ground truth files, including:
  * Tooth/Object Level: Precision, Recall, F1-Score, mIoU, and mAP@50.
  * Image/Diagnostic Level: Accuracy, Sensitivity, Specificity, Precision, and F1-Score.
* **Coordinate Mapping & Visualization:** Includes a dedicated script to parse the model's text output (`[score, ymin, xmin, ymax, xmax]`), denormalize the 0-1000 coordinate scale, and draw bounding boxes directly onto the images using OpenCV.

## Repository Structure
* `zero-shot` : Script for running inference and evaluation without reference examples.
* `few-shots` : Script for running in-context learning inference by prepending reference images (`REF_DIR`) to the prompt.
* `coordinate_mapper` : Utility script using OpenCV and Matplotlib to visualize the AI's predicted bounding boxes and confidence scores on the target images.

## Prerequisites
* Python 3.8+
* Google Cloud Project with the **Vertex AI API** enabled.
* API authentication configured (e.g., via `gcloud auth application-default login`).

### Required Libraries
```bash
pip install google-genai opencv-python matplotlib
