import torch
from models import *

device = "cuda"
model_def = "config/yolov3.cfg"
weights_path = "weights/yolov3.weights"
model = Darknet(model_def) # .to(device)
model.load_darknet_weights(weights_path)

torch.save(model.state_dict(), "checkpoints/test_dump.chkp")