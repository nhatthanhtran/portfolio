import os
import torch
from model import Transformer

class Exp_Basic(object):
    def __init__(self, args, configs, logger):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.configs = configs
        self.logger = logger
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):

        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f'Use MPS') 
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
