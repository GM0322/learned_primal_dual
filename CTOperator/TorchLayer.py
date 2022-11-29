import torch
from torch.autograd import Function
import cupy
from torch.utils.dlpack import to_dlpack
from CTOperator import operator
from utils import config

class ParBeamFPFunction(Function):
    def __init__(self):
        super(ParBeamFPFunction, self).__init__()

    @staticmethod
    def forward(ctx, input):
        batch_size = input.shape[0]
        if(input.shape[1] != 1):
            raise NotImplementedError
        sp = config.getScanParam()
        out = torch.zeros((batch_size, 1, sp['nViews'], sp['nBins']), dtype=torch.float32).to(input.device)
        for i in range(batch_size):
            cupy_out = cupy.fromDlpack(to_dlpack(out[i, 0, :, :]))
            cupy_input = cupy.fromDlpack(to_dlpack(input[i, 0, :, :]))
            operator.fp(cupy_input, sp, cupy_out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        batchsize = grad_output.shape[0]
        if (grad_output.shape[1] != 1):
            raise NotImplementedError
        sp = config.getScanParam()
        grad_input = torch.zeros((batchsize, 1, sp['nSize'], sp['nSize']), dtype=torch.float32).to(grad_output.device)
        for i in range(batchsize):
            cupy_grad_in = cupy.fromDlpack(to_dlpack(grad_input[i, 0, :, :]))
            cupy_grad_out = cupy.fromDlpack(to_dlpack(grad_output[i, 0, :, :]))
            operator.bp(cupy_grad_out, sp, cupy_grad_in)
        return grad_input

class ParBeamFPLayer(torch.nn.Module):
    def __init__(self):
        super(ParBeamFPLayer, self).__init__()

    def forward(self, input):
        return ParBeamFPFunction.apply(input)

class ParBeamBPFunction(Function):
    def __init__(self):
        super(ParBeamBPFunction, self).__init__()

    @staticmethod
    def backward(ctx, input):
        batchsize = input.shape[0]
        if (input.shape[1] != 1):
            raise NotImplementedError
        sp =config.getScanParam()
        out = torch.zeros((batchsize, 1, sp['nViews'], sp['nBins']), dtype=torch.float32).to(input.device)
        for i in range(batchsize):
            cupy_out = cupy.fromDlpack(to_dlpack(out[i, 0, :, :]))
            cupy_input = cupy.fromDlpack(to_dlpack(input[i, 0, :, :]))
            operator.fp(cupy_input, sp, cupy_out)
        return out

    @staticmethod
    def forward(ctx, grad_output):
        batchsize = grad_output.shape[0]
        if (grad_output.shape[1] != 1):
            raise NotImplementedError
        sp = config.getScanParam()
        grad_input = torch.zeros((batchsize, 1, sp['nSize'], sp['nSize']), dtype=torch.float32).to(grad_output.device)
        for i in range(batchsize):
            cupy_grad_in = cupy.fromDlpack(to_dlpack(grad_input[i, 0, :, :]))
            cupy_grad_out = cupy.fromDlpack(to_dlpack(grad_output[i, 0, :, :]))
            operator.bp(cupy_grad_out, sp, cupy_grad_in)
        return grad_input

class ParBeamBPLayer(torch.nn.Module):
    def __init__(self):
        super(ParBeamBPLayer, self).__init__()

    def forward(self,proj):
        return ParBeamBPFunction.apply(proj)
