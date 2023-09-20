import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from utils import adain


class Projection(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        gap = in_features - out_features
        self.fc1 = nn.Sequential(nn.Linear(in_features, in_features - gap//2, bias=True), nn.ReLU())
        self.fc2 = nn.Linear(in_features - gap//2, out_features, bias=True)

    def forward(self, in_features):
        x = self.fc1(in_features)
        x = self.fc2(x)
        return x


class Effi_B7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet_b7(weights = EfficientNet_B7_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(nn.Dropout(0.5, inplace=True), nn.Linear(2560, num_classes, bias=True))
        self.projection_head = Projection(2560, 1280)

    def forward(self, source_images, target_images):
        # AdaIN features
        z_s = self.model.features[:4](source_images) # 1 80 28 28
        z_t = self.model.features[:4](target_images)
        z_st = adain(z_s, z_t)
        z_ts = adain(z_t, z_s)

        z_s = self.model.features[4:](z_s)
        z_t = self.model.features[4:](z_t)
        z_st = self.model.features[4:](z_st)
        z_ts = self.model.features[4:](z_ts) # b 2560 7 7

        pred_z_s = self.model.avgpool(z_s)
        pred_z_t = self.model.avgpool(z_t)
        pred_z_st = self.model.avgpool(z_st)
        pred_z_ts = self.model.avgpool(z_ts)
        
        pred_z_s = nn.Flatten()(pred_z_s) # b 2560
        pred_z_t = nn.Flatten()(pred_z_t)
        pred_z_st = nn.Flatten()(pred_z_st)
        pred_z_ts = nn.Flatten()(pred_z_ts)
        
        projected_z_s = self.projection_head(pred_z_s.clone())
        projected_z_t = self.projection_head(pred_z_t.clone())
        projected_z_st = self.projection_head(pred_z_st.clone())
        projected_z_ts = self.projection_head(pred_z_ts.clone())

        pred_z_s = self.model.classifier(pred_z_s)
        pred_z_t = self.model.classifier(pred_z_t)
        pred_z_st = self.model.classifier(pred_z_st)
        pred_z_ts = self.model.classifier(pred_z_ts)
        
        return pred_z_s, pred_z_t, pred_z_st, pred_z_ts, projected_z_s, projected_z_t, projected_z_st, projected_z_ts
