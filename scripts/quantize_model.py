import os
import torch



model = torch.load("/home/stanik/repos/nvidia_hrnet-ocr/checkpoints/ep110_railsem19_0.pth",
                                map_location=torch.device('cpu'))


model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)

torch.save(model, '/home/stanik/repos/nvidia_hrnet-ocr/checkpoints/post_dynamic_quant0.pth')
