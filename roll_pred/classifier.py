import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import roi_align


class ROIClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2]) # C=512feature map
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, num_classes)
                )

    def forward(self, images, bbox_list):
        features = self.backbone(images) # (B, 512, H/32, W/32)
        all_rois = [] # [(batch_idx, x1, y1, x2, y2), ...]
        for batch_idx, boxes in enumerate(bbox_list):
            for box in boxes:
                roi = torch.cat([torch.tensor([batch_idx], device=features.device), box])
                all_rois.append(roi)
        rois = torch.stack(all_rois)
        roi_features = roi_align(features, rois, output_size=(7, 7))
        outputs = self.classifier(roi_features)
        return outputs # (total_rois, num_classes)
