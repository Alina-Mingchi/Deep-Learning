import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from pathlib import Path
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from others.network import *

torch.set_grad_enabled(True)

class Model(nn.Module):
    def __init__(self):
        """For mini-project 1"""
        super(Model, self).__init__()
        self.model = UNet(nch_in = 3, nch_out=3)
        self.learning_rate = 1e-3
        self.mini_batch_size = 1000
        self.nb_epochs = 200

        # optimizer: ADAM performs better than SGD (commented out)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001,betas=(0.9,0.99),eps=1e-08)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200)
        
        # loss function: L2 loss outperforms L1 loss (commented out)
        self.loss_function = nn.MSELoss()
        # self.loss_function = nn.L1Loss()

        self.record = False


    def load_pretrained_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Loading pretrained model:')
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)

    # define save_model function to have same hierarchy as the load_pretrained_model()
    def save_model(self):
        model_path = Path(__file__).parent / "bestmodel.pth"
        torch.save(self.model.state_dict(), model_path)


    def train(self, train_input, train_target, num_epochs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        train_input = train_input.float().to(device)
        train_target = train_target.float().to(device)

        n_sample = train_input.shape[0]

        # could write data to txt file for later investigation and plotting
        if self.record:
            with open(f'pytorch_record-{self.learning_rate}-n_sample_{n_sample}.txt', 'w') as file:
                file.write(f'device: {device} \n')
        
        # now = time.time()
        for e in range(0, num_epochs):
            acc_loss = 0
            start = 0
            end = self.mini_batch_size

            while start < len(train_input):
                if end >= len(train_input):
                    end = len(train_input)
                
                output = self.model(train_input[start:end])
                output = output * 255.0

                loss = self.loss_function(output, train_target[start:end])
                acc_loss += loss.item()
                self.optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()


                start = end
                end += self.mini_batch_size
        
            # self.optimizer.step()
            self.scheduler.step()
            # print('epoch:{:d}, loss:{:.3f}, lr:{:.3f}*e-2'.format(e+1,acc_loss,self.optimizer.param_groups[0]['lr']*1e2))

            if self.record:
                with open(f'pytorch_record-{self.learning_rate}-n_sample_{n_sample}.txt', 'a') as file:
                    file.write('epoch:{:d}, loss:{:.3f}, lr:{:.3f}*e-2\t'.format(e+1,acc_loss,self.optimizer.param_groups[0]['lr']*1e2))

    # function used to train the network, and observer the performance on validation set
    def train_advanced(self, train_input, train_target, val_input, val_target):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        train_input = train_input.float().to(device)
        train_target = train_target.float().to(device)
        val_input = val_input.float().to(device)
        val_target = val_target.float().to(device)

        n_sample = train_input.shape[0]

        if self.record:
            with open(f'pytorch_record-{self.learning_rate}-n_sample_{n_sample}.txt', 'w') as file:
                file.write(f'device: {device} \n')
        
        now = time.time()
        for e in range(0, self.nb_epochs):
            acc_loss = 0
            start = 0
            end = self.mini_batch_size

            while start < len(train_input):
                if end >= len(train_input):
                    end = len(train_input)
                
                output = self.model(train_input[start:end])
                output = output * 255.0

                loss = self.loss_function(output, train_target[start:end])
                acc_loss += loss.item()
                self.optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()


                start = end
                end += self.mini_batch_size
            
            self.scheduler.step()
            # self.optimizer.step()
            consume_time = time.time() - now
            now = time.time()
            print('epoch:{:d}, loss:{:.3f}, time:{:.1f}s, lr:{:.3f}*e-2'.format(e+1,acc_loss,consume_time,self.optimizer.param_groups[0]['lr']*1e2))

            val_output = self.model.forward(val_input)
            val_loss = self.loss_function(val_output, val_target)
            psnr = compute_psnr(val_output, val_target/255.0)

            print(f'after training {e+1:d} epoch, psnr for validation data is:{psnr:.3f}')
            if self.record:
                with open(f'/content/drive/MyDrive/Colab Notebooks/pytorch_record-{self.learning_rate}-n_sample_{n_sample}.txt', 'a') as file:
                    file.write('{:d}, {:.3f}, {:.3f}, {:.3f}\n'.format(e+1,acc_loss,val_loss,psnr))

    # predict the output with trained model
    def predict (self, test_input):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.train(False)
        test_input = test_input.float().to(device)
        output = self.model(test_input)
        return output * 255.0

# compute psnr
def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()


# #######################################
# # Used for checking the performance of our implementation

# if __name__=="__main__":

#     # load input data      
#     noisy_imgs_1, noisy_imgs_2 = torch.load('train_data.pkl')
#     noisy_imgs, clean_imgs = torch.load('val_data.pkl')
#     noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1.float(), noisy_imgs_2.float()
#     noisy_imgs, clean_imgs = noisy_imgs.float(), clean_imgs.float()

#     net = Model()
#     net.train_advanced(noisy_imgs_1, noisy_imgs_2, noisy_imgs, clean_imgs, is_normalize=False)

#     # net.save_model()

#     ######################################
#     ##          Data Augmentation       ##
#     ######################################
#     import torch.nn.functional as F
#     import torchvision.transforms as TF 
#     # we use torchvision only for data augmentation purpose

#     torch.manual_seed(10)

#     # Augmentation by flipping vertically, horizontally, p=1 indicates that this applies to all
#     tf1_1 = TF.RandomHorizontalFlip(p=1)
#     tf1_imgs1 = tf1_1(noisy_imgs_1)
#     tf1_2 = TF.RandomHorizontalFlip(p=1)
#     tf1_imgs2 = tf1_2(noisy_imgs_2)

#     tf2_1 = TF.RandomVerticalFlip(p=1)
#     tf2_imgs1 = tf2_1(noisy_imgs_1)
#     tf2_2 = TF.RandomVerticalFlip(p=1)
#     tf2_imgs2 = tf2_2(noisy_imgs_2)

#     # Concatenate dataset
#     increased_imgs1 = torch.utils.data.ConcatDataset([tf1_imgs1,tf2_imgs1,noisy_imgs_1])
#     increased_imgs2 = torch.utils.data.ConcatDataset([tf1_imgs2,tf2_imgs2,noisy_imgs_2])  

#     increased_imgs1.datasets[0].index_select(0, torch.randperm(increased_imgs1.datasets[0].shape[0]))
#     increased_imgs2.datasets[0].index_select(0, torch.randperm(increased_imgs2.datasets[0].shape[0]))

#     net = Model()
#     net.train_advanced(increased_imgs1.datasets[0], increased_imgs2.datasets[0], noisy_imgs, clean_imgs)
#     # net.save_model()