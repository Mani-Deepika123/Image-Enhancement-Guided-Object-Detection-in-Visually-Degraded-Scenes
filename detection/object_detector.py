import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
from torchvision import models
from torchvision.transforms import ToTensor
import torch.nn.functional as F
class FeatureGuidedModule(nn.Module):
    def __init__(self):
        super(FeatureGuidedModule, self).__init__()
        self.feature_adjustment_layer = nn.Linear(1000, 2000)  
    def forward(self, original_features, enhanced_features):
        if original_features.dim() < 4 or enhanced_features.dim() < 4:
            raise ValueError("Input tensors must have at least 4 dimensions (batch, channels, height, width).")
        max_pool = F.adaptive_max_pool2d(original_features, (1, 1))
        avg_pool = F.adaptive_avg_pool2d(original_features, (1, 1))       
        pooled_features = torch.cat((max_pool, avg_pool), dim=1)
        adjusted_enhanced_features = self.feature_adjustment_layer(enhanced_features.view(enhanced_features.size(0), -1))
        adjusted_enhanced_features = adjusted_enhanced_features.view(enhanced_features.size(0), -1, 1, 1)
        loss = F.mse_loss(pooled_features, adjusted_enhanced_features)
        return loss
class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.eval()
        self.to_tensor = ToTensor()
        self.feature_guided_module = FeatureGuidedModule()
        self.feature_adjustment_layer = nn.Linear(1000, 2000)  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  
    def extract_features(self, image):
        with torch.no_grad():
            image_tensor = self.to_tensor(image).unsqueeze(0)  
            features = self.feature_extractor(image_tensor)
            if features.dim() == 2:  
                features = features.unsqueeze(2).unsqueeze(3)         
        return features
    def compute_feature_consistency_loss(self, original_features, enhanced_features):
        loss = self.feature_guided_module(original_features, enhanced_features)
        return loss
    def detect_objects(self, image):
        original_features = self.extract_features(image)
        results = self.model(image)
        return results, original_features
    def draw_detections(self, image, results):
        detected_img = image.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = r.names[int(box.cls[0])]
                cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(detected_img, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return detected_img
