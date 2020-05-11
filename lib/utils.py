import os
import ssl
import torch
import time
import numpy as np
from torchsummary import summary
from thop.profile import profile


def get_root_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_real_path(path):
    if not path.startswith('/'):
        path = os.path.join(get_root_path(), path)
    return os.path.abspath(path)


def ssl_set_unverified_context():
    if hasattr(ssl, '_create_unverified_context'):
        ssl._create_default_https_context = ssl._create_unverified_context


# Modified from https://github.com/sacmehta/ESPNet/issues/57#issuecomment-479352167
def compute_inference_time(model, input_dim=(1, 3, 256, 256), cuda=True, repeats=100):
    if len(input_dim) < 4:
        input_dim = (1, *input_dim)
    inputs = torch.randn(input_dim)
    if cuda:
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    i = 0
    time_spent = []
    while i < repeats:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if cuda:
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    mean_time = np.mean(time_spent)
    print("----------------------------------------------------------------")
    print("Average inference time(ms): {:.3f}".format(mean_time))
    print("----------------------------------------------------------------")
    return mean_time


def calculate_model_complexity(model, input_dim=(1, 3, 256, 256), cuda=True):
    if len(input_dim) < 4:
        input_dim = (1, *input_dim)
    inputs = torch.randn(input_dim)
    if cuda:
        model = model.cuda()
        inputs = inputs.cuda()
    summary(model, input_size=tuple(input_dim[1:]))
    macs, params = profile(model, inputs=(inputs,))
    print("----------------------------------------------------------------")
    print("Params size (MB): {:.2f}".format(params / (1000 ** 2)))
    print("MACs (M): {:.2f}".format(macs / (1000 ** 2)))
    print("MACs (G): {:.2f}".format(macs / (1000 ** 3)))
    print("----------------------------------------------------------------")
    return macs, params
