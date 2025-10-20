import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

def create_vgg_faster_rcnn_model(num_classes):
 
    vgg_backbone = torchvision.models.vgg16(pretrained=True).features
    

    vgg_backbone.out_channels = 512
  
    anchor_generator = AnchorGenerator(sizes=((128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    

    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                      output_size=7,
                                      sampling_ratio=2)

    model = FasterRCNN(
        backbone=vgg_backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    print("VGG16 Backbone Faster R-CNN 모델 생성")
    print(f"모델 클래스 수: {num_classes}")
    
    return model

if __name__ == "__main__":
    

    NUM_CLASSES = 5 
    
    model = create_vgg_faster_rcnn_model(num_classes=NUM_CLASSES)
    
   
    dummy_image = torch.randn(1, 3, 640, 640)
    

    model.eval() 
    with torch.no_grad():
        output = model(dummy_image)
        
    print("\n모델 실행 테스트 완료")
    print(f"출력 형식: {type(output)}")
    print(f"출력 개수: {len(output)}")
    print("출력 상세:", output)