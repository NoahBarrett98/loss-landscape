from essl.backbones import largerCNN_backbone
from essl.evaluate_downstream import finetune_model

def largerCNN():
    backbone = largerCNN_backbone()
    return finetune_model(backbone.backbone, backbone.in_features, 10)