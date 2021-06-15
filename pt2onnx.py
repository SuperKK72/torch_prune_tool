import torch
from models import *
 
def convert_onnx():
    print('start!!!')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model_path = '/home/pi/xg_openpose_fall_detect-master/action_detect/checkPoint/action.pt' #这是我们要转换的模型
    #backone = mobilenetv3_large(width_mult=0.75)#mobilenetv3_small()  mobilenetv3_small(width_mult=0.75)  mobilenetv3_large(width_mult=0.75)
    #model = NetV2().to(device)
    cfg = 'cfg/prune_0.93_keep_0.01_16_shortcut_yolov3-spp-hand.cfg'
    img_size = 416
    model = Darknet(cfg, (img_size, img_size)).to(device)
    weights = 'weights/last.pt'
        
      
        
    checkpoint = torch.load('weights/last.pt', map_location='cpu')
    checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(checkpoint['model'], strict=False)
    print('loaded weights from', weights, '\n')
    #model.load_state_dict(torch.load(model_path, map_location=device)['model'])
 
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1,3,416,416).to(device)#输入大小   #data type nchw
    #onnx_path = '/home/pi/xg_openpose_fall_detect-master/action_detect/checkPoint/action.onnx'
    onnx_path = 'action.onnx'
    print("----- pt模型导出为onnx模型 -----")
    output_name = "action.onnx"
    torch.onnx.export(model, dummy_input,onnx_path,export_params=True, input_names=['input'], output_names=['output'])
    print('finish!!!')
    

if __name__ == "__main__" :
    convert_onnx()
