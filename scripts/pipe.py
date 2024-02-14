
from datetime import datetime, timedelta
from time import time
import pickle
import torch

from scripts.myDataset import Model1Dataset, Model2Dataset
from scripts.myModels import UNetVGG16, vgg16_transform
from scripts.train import train_model, get_data_loader
from scripts.utils import *

class pipeline:
    class holder:
        def __init__(self):
            pass
    def __init__(self, w1, w2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
 
        self.model1 = UNetVGG16(1).to(self.device)
        self.model1.load_state_dict(torch.load(w1, map_location=torch.device(self.device)))
        self.model1.eval()

        self.model2 = UNetVGG16(2).to(self.device)
        self.model2.load_state_dict(torch.load(w2, map_location=torch.device(self.device)))
        self.model2.eval()

        self.transform = vgg16_transform
        self.result = self.holder()

    def _freeze_encoder1(self):
        for block in [self.model1.enc1, self.model1.enc2, self.model1.enc3, self.model1.enc4, self.model1.enc5]:
            for param in block.parameters():
                param.requires_grad = False

    def _freeze_encoder2(self):
        for block in [self.model2.enc1, self.model2.enc2, self.model2.enc3, self.model2.enc4, self.model2.enc5]:
            for param in block.parameters():
                param.requires_grad = False

    def _arr_to_img(self, arr, threshold=0):
        return arr_to_img(arr, threshold, cs={'R':(1, 1), 'G':(0, 1), 'B':(None, 1)})

    def save_weights(self):
        torch.save(self.model1.state_dict(), 'model1_weights.pkl')
        torch.save(self.model2.state_dict(), 'model2_weights.pkl')

    def train_model1(self, 
                     data_path="", 
                     train_fraction=0.8, 
                     batch_size=5, 
                     learning_rate=0.01, 
                     num_epochs=1):

        start = time()
        model_data = {'date_time': datetime.utcnow() + timedelta(hours=7),
                      'hyper_params': {'TRAIN_FRACTION' : train_fraction,
                                       'BATCH_SIZE'     : batch_size,
                                       'LEARNING_RATE'  : learning_rate,
                                       'NUM_EPOCH'      : num_epochs}}

        train_dataloader, test_dataloader = get_data_loader(data_path, Model1Dataset, train_fraction, batch_size)
        self._freeze_encoder1()
        train_model(self.model1, train_dataloader, test_dataloader, learning_rate, num_epochs, save_path='model1_weights.pkl')

        model_data['duration'] = time() - start
        torch.save(self.model1.state_dict(), 'model1_weights.pkl')
        with open('history1.pkl', 'wb') as file:
            pickle.dump(model_data, file)

    def train_model2(self, 
                     data_path="", 
                     train_fraction=0.8, 
                     batch_size=5, 
                     learning_rate=0.01, 
                     num_epochs=1):

        start = time()
        model_data = {'date_time': datetime.utcnow() + timedelta(hours=7),
                      'hyper_params': {'TRAIN_FRACTION' : train_fraction,
                                       'BATCH_SIZE'     : batch_size,
                                       'LEARNING_RATE'  : learning_rate,
                                       'NUM_EPOCH'      : num_epochs}}

        train_dataloader, test_dataloader = get_data_loader(data_path, Model2Dataset, train_fraction, batch_size)
        self._freeze_encoder2()
        train_model(self.model2, train_dataloader, test_dataloader, learning_rate, num_epochs, save_path='model2_weights.pkl')

        model_data['duration'] = time() - start
        torch.save(self.model2.state_dict(), 'model2_weights.pkl')
        with open('history2.pkl', 'wb') as file:
            pickle.dump(model_data, file)

    def predict(self, image_pre, image_post):
        del self.result
        self.result             = self.holder()
        self.result.image_pre   = image_pre
        self.result.image_pre_  = self.transform(image_pre)

        self.result.arr1    = get_result(self.result.image_pre_, self.model1)
        self.result.mask    = self._arr_to_img(self.result.arr1.numpy(), 0.5)

        self.result.image_post   = image_post
        self.result.image_post_  = self.transform(image_post)

        self.result.arr2    = get_result(self.result.image_post_ * self.result.arr1, self.model2)
        self.result.dmgs    = self._arr_to_img(self.result.arr2.numpy(), 0.5)
