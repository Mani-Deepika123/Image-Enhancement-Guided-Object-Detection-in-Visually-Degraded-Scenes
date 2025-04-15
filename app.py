from flask import Flask, request, render_template, send_from_directory
import os
import cv2
from enhancement.image_dehazer import ImageDehazer
from detection.object_detector import ObjectDetector
from torchvision.transforms import ToTensor
import numpy as np
def calculate_psnr(original_image, enhanced_image):
    mse = np.mean((original_image - enhanced_image) ** 2)
    if mse == 0:
        return 100  
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))
def train_model(detector, dehazer, img):
    haze_corrected_img = dehazer.remove_haze(img)
    results, original_features = detector.detect_objects(haze_corrected_img)
    enhanced_features = detector.extract_features(haze_corrected_img)
    consistency_loss = detector.compute_feature_consistency_loss(original_features, enhanced_features)
    total_loss = consistency_loss  
    total_loss.backward()
    detector.optimizer.step()
    return haze_corrected_img, results, consistency_loss
def calculate_mse(original_image, enhanced_image):
    return np.mean((original_image - enhanced_image) ** 2)
def calculate_rmse(original_image, enhanced_image):
    mse = calculate_mse(original_image, enhanced_image)
    return np.sqrt(mse)
def calculate_nmse(original_image, enhanced_image):
    mse = calculate_mse(original_image, enhanced_image)
    return mse / np.mean(original_image ** 2)
def calculate_metrics(original_image, enhanced_image):
    original_image = original_image.astype(np.float32)
    enhanced_image = enhanced_image.astype(np.float32)
    mse_value = calculate_mse(original_image, enhanced_image)
    rmse_value = calculate_rmse(original_image, enhanced_image)
    nmse_value = calculate_nmse(original_image, enhanced_image)
    return rmse_value, nmse_value
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"   
    file = request.files['file']
    if file.filename == '':
        return "No selected file"   
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    img = cv2.imread(file_path)   
    if img is None:
        return "Error: Unable to read the image."
    dehazer = ImageDehazer()
    detector = ObjectDetector("models/yolov3u.pt")
    haze_corrected_img, results, consistency_loss = train_model(detector, dehazer, img)
    haze_corrected_path = os.path.join(OUTPUT_FOLDER, 'haze_corrected.png')
    cv2.imwrite(haze_corrected_path, haze_corrected_img)
    output_path = os.path.join(OUTPUT_FOLDER, 'detected_image.png')
    detected_img = detector.draw_detections(haze_corrected_img, results)
    cv2.imwrite(output_path, detected_img)
    rmse_value, nmse_value = calculate_metrics(img, haze_corrected_img)
    psnr_value = calculate_psnr(img, haze_corrected_img)   
    detection_boxes = []
    for r in results:
        for box in r.boxes:
            detection_boxes.append({
                'label': r.names[int(box.cls[0])],
                'confidence': float(box.conf[0])
        })
    return render_template('index.html', 
                           original_image=file.filename, 
                           rmse_value=rmse_value,
                           nmse_value=nmse_value,
                           consistency_loss=consistency_loss.item(),  
                           psnr_value=psnr_value,
                           detection_boxes=detection_boxes)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)
if __name__ == '__main__':
    app.run(debug=True, port=5001)