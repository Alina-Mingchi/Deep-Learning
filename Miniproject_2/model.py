from torch import empty
from torch.nn.functional import unfold
import math
import time
import pickle
import os
import random

import torch
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)


class Module(object):
    """The Module class represents the basic structure of each module"""

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        None


class ReLU(Module):
    """The module ReLu applies the rectified linear unit activation function to convert input value into non-negative value"""

    def __init__(self):
        super(ReLU, self).__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        x[x < 0] = 0.01 * x[x < 0]
        return x

    def backward(self, gradwrtoutput):
        x = self.x
        x[x >= 0] = 1
        x[x < 0] = 0.01
        return x * gradwrtoutput


class Sigmoid(Module):
    """The module Sigmoid applies the sigmoidal activation function to convert input value into range (0,1)"""

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.x = None
    
    def forward(self, x):
        self.x = x
        return 1 / (1 + (-x).exp())
    
    def backward(self, gradwrtoutput):
        x = self.x
        result = 1 / (1 + (-x).exp())
        return gradwrtoutput * result * (1-result)


class MSE(Module):
    """The module MSE implements Mean Squared Error as a loss function"""

    def __init__(self):
        self.error = None
        self.n = None

    def forward(self, pred, label):
        self.error = pred - label
        self.n = pred.shape[0]
        return (self.error**2).mean()

    def backward(self):
        return 2* self.error / self.n


class SGD(Module):
    """The module SGD implements the Stochastic Gradient Descent (SGD) optimizer to optimize the parameters of the model"""

    def __init__(self, sequential, lr=0.01):
        self.sequential = sequential
        self.lr_max = lr
        self.lr = self.lr_max

    def step(self):
        for i in range(len(self.sequential.model)):
            # only for Conv2d and Upsampling module, we need to do gradient descent
            if not self.sequential.model[i].param() == []:
                self.sequential.model[i].weight -= self.lr * self.sequential.model[i].dw
                self.sequential.model[i].bias -= self.lr * self.sequential.model[i].db        
    
    def CosineAnnealingLR(self, T_cur, T_max):
        """Set the learning rate of each parameter group using a cosine annealing schedule"""
        self.lr = 0.5 * (1 + math.cos(math.pi * T_cur / T_max)) * self.lr_max
    
       
class Sequential(Module):
    """The module Sequential puts together an arbitrary configuration of model together"""

    def __init__(self, *model):
        super(Sequential, self).__init__()
        self.model = []
        for mod in model:
            self.model.append(mod)

    def forward(self, input):
        out = input
        for module in self.model:
            out = module.forward(out)
        return out

    def backward(self, grdwrtoutput):
        out = grdwrtoutput
        for module in reversed(self.model):
            out = module.backward(out)
        return out

    def param(self):
        parameters = []
        for module in self.model:
            for p in module.param():
                parameters.append(p)
        return parameters
    
    def zero_grad(self):
        for module in self.model:
            module.zero_grad()

                
class Conv2d(Module):
    """The module Conv2d is implemented as a standard convolution for 2d images"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, is_cuda=False):
        super(Conv2d, self).__init__()
        # parameters for convolution
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.is_cuda = is_cuda
        # input
        self.x = None
        # matrix for sliding local blocks of input
        self.unfold = None

        # convert k_size into a tuple 
        if isinstance(self.k_size, tuple):
            pass
        elif isinstance(self.k_size, int):
            self.k_size = (self.k_size, self.k_size)
        else:
            raise TypeError('The kernel_size should be either tuple or int type')
        

        # assign the device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.is_cuda else "cpu")
        # weight initialization and move all the params to the assigned device
        bound = 1 / (self.in_ch*self.k_size[0]*self.k_size[1])**0.5
        self.weight = empty((self.out_ch, self.in_ch, self.k_size[0], self.k_size[1])).uniform_(-bound, bound).to(self.device)
        self.bias = empty(self.out_ch).zero_().uniform_(-bound, bound).to(self.device)
        self.dw = empty(self.weight.shape).zero_().to(self.device)
        self.db = empty(self.bias.shape).zero_().to(self.device)
        
    def forward(self, x):
        self.x = x # input x.shape: n_sample, in_ch, H, W
        self.x = self.x.to(self.device)
        if not self.x.shape[1]==self.in_ch:
            raise ValueError('The channel for input data dose not match with the in_channels of Conv2d module')

        self.unfold = unfold(self.x, kernel_size=self.k_size, dilation=self.dilation,\
                        padding=self.padding, stride=self.stride)
        # self.unfold.shape: n_sample, in_ch*k_size[0]*k_size[1], H_out*W_out
        
        # The forward_output in the form of matrix
        wxb = self.weight.view(self.out_ch, -1) @ self.unfold + self.bias.view(1, -1, 1)

        H_out = math.floor((self.x.shape[2] + 2*self.padding - self.dilation*(self.k_size[0]-1) - 1)\
                        /self.stride + 1)
        W_out = math.floor((self.x.shape[3] + 2*self.padding - self.dilation*(self.k_size[1]-1) - 1)\
                        /self.stride + 1)
        
        # The forward_output in the form of img, shape: n_sample, out_ch, H_out, W_out
        wxb_img = wxb.view(self.x.shape[0], self.out_ch, H_out, W_out)

        return wxb_img

    def backward(self, gradwrtoutput):
        # gradwrtoutput be the same shape as wxb_img: n_sample, out_ch, H_out, W_out
        gradwrtoutput = gradwrtoutput.to(self.device)
        (n_sample, out_ch, H_out, W_out) = gradwrtoutput.shape
        assert(out_ch==self.out_ch)
        

        
        # Finding dw and db: convolution between input and gradwrtoutput
        x_unfold = unfold(self.x, kernel_size=(H_out, W_out), dilation=self.stride,\
                        padding=self.padding, stride=self.dilation)
        # x_unfold.shape: n_sample, in_ch*H_out*W_out, k_size[0]*k_size[1]
        # self.dw.shape: out_ch, in_ch, k_size[0], k_size[1]
        # self.db.shape: out_ch
        self.db += gradwrtoutput[:,:,:,:].sum((0,2,3))
        for j in range(self.out_ch):
            for k in range(self.in_ch):
                self.dw[j,k,:,:] += gradwrtoutput[:,j,:,:].reshape(-1)\
                        .matmul(x_unfold[:,k*H_out*W_out:(k+1)*H_out*W_out,:]\
                        .reshape(-1,self.k_size[0]*self.k_size[1])).reshape(self.k_size[0],-1)

       

        # Finding dx: full_convolution between gradwrtoutput and weight
        # self.dx.shape: n_sample, in_ch, H, W
        # padding dx
        self.dx = empty((self.x.shape[0],self.x.shape[1],self.x.shape[2]+2*self.padding,\
                        self.x.shape[3]+2*self.padding)).zero_().to(self.device) 
        for h in range(H_out):
            for w in range(W_out):
                vert_start = h*self.stride
                vert_end = h*self.stride+self.k_size[0]
                hori_start = w*self.stride
                hori_end = w*self.stride+self.k_size[1]
                self.dx[:,:,vert_start:vert_end,hori_start:hori_end] += (gradwrtoutput[:,:,h,w]\
                .repeat(self.in_ch,self.k_size[0],self.k_size[1],1,1).permute(3,4,0,1,2)*self.weight[:,:,:,:]).sum(1)
        
        # remove the padding part
        self.dx = self.dx[:, :, self.padding:self.dx.shape[2]-self.padding, self.padding:self.dx.shape[3]-self.padding]
        # Making sure dx has the shape as x
        assert(self.dx.shape == self.x.shape)

        return self.dx

    def param(self):
        return [[self.weight, self.dw], [self.bias, self.db]]

    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()
    
     
class Upsampling(Module):
    """The module Upsampling is implemented as a combination of Nearest neighbor upsampling and Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1, scale_factor=2, is_cuda=False):
        super(Upsampling, self).__init__()
        # parameters for upsampling module
        self.scale_fc = scale_factor # scale factor for upsampling
        if not isinstance(self.scale_fc, int):
            raise TypeError('The scale_factor for nearest upsampling is preferred to be int type')
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.is_cuda = is_cuda
        # input
        self.x = None
        # The upsampled input
        self.x_up = None
        # matrix for sliding local blocks of upsampled input
        self.unfold = None

        # convert k_size into a tuple 
        if isinstance(self.k_size, tuple):
            pass
        elif isinstance(self.k_size, int):
            self.k_size = (self.k_size, self.k_size)
        else:
            raise TypeError('The kernel_size should be either tuple or int type')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.is_cuda else "cpu")
        # weight initialization and move all the params to the assigned device
        bound = 1 / (self.in_ch*self.k_size[0]*self.k_size[1])**0.5
        self.weight = empty((self.out_ch, self.in_ch, self.k_size[0], self.k_size[1])).uniform_(-bound, bound).to(self.device)
        self.bias = empty(self.out_ch).uniform_(-bound, bound).to(self.device)
        self.dw = empty(self.weight.shape).zero_().to(self.device)
        self.db = empty(self.bias.shape).zero_().to(self.device)

    def forward(self, x):
        self.x = x  # input x.shape: n_sample, in_ch, H, W
        self.x = self.x.to(self.device)

        n_sample, H, W = self.x.shape[0], self.x.shape[2], self.x.shape[3]
        if not self.x.shape[1]==self.in_ch:
            raise ValueError('The channel for input data dose not match with the in_channels of Upsampling module')
        
        # Nearest upsampling
        H_up = int(H * self.scale_fc)
        W_up = int(W * self.scale_fc)
        self.x_up = empty((n_sample, self.in_ch, H_up, W_up)).zero_().to(self.device)
        for i in range(self.scale_fc):
            for j in range(self.scale_fc):
                self.x_up[:, :, i::self.scale_fc, j::self.scale_fc] = self.x
        
        # x_up.shape: n_sample, in_ch, H_up, W_up
        self.unfold = unfold(self.x_up, kernel_size=self.k_size, dilation=self.dilation,\
                        padding=self.padding, stride=self.stride)
        # self.unfold.shape: n_sample, in_ch*k_size[0]*k_size[1], H_out*W_out
        
        # The forward output in the form of matrix
        wxb = self.weight.view(self.out_ch, -1) @ self.unfold + self.bias.view(1, -1, 1)

        H_out = math.floor((self.x_up.shape[2] + 2*self.padding - self.dilation*(self.k_size[0]-1) - 1)\
                        /self.stride + 1)
        W_out = math.floor((self.x_up.shape[3] + 2*self.padding - self.dilation*(self.k_size[1]-1) - 1)\
                        /self.stride + 1)
        
        # The forward_output in the form of img, shape: n_sample, out_ch, H_out, W_out
        wxb_img = wxb.view(self.x_up.shape[0], self.out_ch, H_out, W_out)

        return wxb_img
    
    def backward(self, gradwrtoutput):
        # gradwrtoutput be the same shape as wxb_img: n_sample, out_ch, H_out, W_out
        gradwrtoutput = gradwrtoutput.to(self.device)
        (n_sample, out_ch, H_out, W_out) = gradwrtoutput.shape
        assert(out_ch==self.out_ch)

        # Finding dw and db: convolution between input and gradwrtoutput
        # x_unfold.shape: n_sample, in_ch*H_out*W_out, k_size[0]*k_size[1]
        # self.dw.shape: out_ch, in_ch, k_size[0], k_size[1]
        # self.db.shape: out_ch
        x_unfold = unfold(self.x_up, kernel_size=(H_out, W_out), dilation=self.stride,\
                        padding=self.padding, stride=self.dilation)

        self.db += gradwrtoutput[:,:,:,:].sum((0,2,3))
        for j in range(self.out_ch):
            for k in range(self.in_ch):
                self.dw[j,k,:,:] += gradwrtoutput[:,j,:,:].reshape(-1)\
                        .matmul(x_unfold[:,k*H_out*W_out:(k+1)*H_out*W_out,:]\
                        .reshape(-1,self.k_size[0]*self.k_size[1])).reshape(self.k_size[0],-1)
        
        # Finding dx: full_convolution between gradwrtoutput and weight
        # self.dx.shape: n_sample, in_ch, H, W
        # padding dx
        self.dx = empty((self.x_up.shape[0],self.x_up.shape[1],self.x_up.shape[2]+2*self.padding,\
                        self.x_up.shape[3]+2*self.padding)).zero_().to(self.device) 
                  
        for h in range(H_out):
            for w in range(W_out):
                vert_start = h*self.stride
                vert_end = h*self.stride+self.k_size[0]
                hori_start = w*self.stride
                hori_end = w*self.stride+self.k_size[1]
                self.dx[:,:,vert_start:vert_end,hori_start:hori_end] += (gradwrtoutput[:,:,h,w]\
                .repeat(self.in_ch,self.k_size[0],self.k_size[1],1,1).permute(3,4,0,1,2)*self.weight[:,:,:,:]).sum(1)

        # remove padding
        self.dx = self.dx[:, :, self.padding:self.dx.shape[2]-self.padding, self.padding:self.dx.shape[3]-self.padding]
        # down-sampling, and sum all the gradients to the same input pixel
        self.dx = self.dx[:,:,::self.scale_fc,::self.scale_fc] * self.scale_fc**2
        # Making sure dx has the shape as x
        assert(self.dx.shape == self.x.shape)


        return self.dx

    def param(self):
        return [[self.weight, self.dw], [self.bias, self.db]]

    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()


class Model():
    def __init__(self):
        """Mini-project 2, implement the sequential network"""
        # whether or not to use the cuda, default value as True 
        self.is_cuda = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.is_cuda else "cpu")
        # Construct our model
        self.model = Sequential(Conv2d(3,16,4,2,1,1,self.is_cuda), ReLU(), Conv2d(16,32,4,2,1,1,self.is_cuda), ReLU(),\
                    Upsampling(32,16,3,1,1,1,2,self.is_cuda), ReLU(), Upsampling(16,3,3,1,1,1,2,self.is_cuda), Sigmoid())

        # some hyper-parameters for training
        self.learning_rate = 1e-7
        self.mini_batch_size = 1000
        self.nb_epochs = 1
        self.T_max = 100

        # optimizer and loss function
        self.optimizer = SGD(self.model, lr=self.learning_rate)
        self.loss_function = MSE()

    def load_pretrained_model(self):
        """loads the parameters saved in bestmodel.pkl into the model"""
        dir_name = os.path.dirname(__file__)
        file_name =os.path.join(dir_name, 'bestmodel.pkl')

        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        
        for i in range(len(self.model.model)):
            # only for Conv2d and Upsampling module, we need to load the params
            if not self.model.model[i].param() == []:
                self.model.model[i].weight = params[i][0].to(self.device)
                self.model.model[i].dw = params[i][1].to(self.device)
                self.model.model[i].bias = params[i+1][0].to(self.device)
                self.model.model[i].db = params[i+1][1].to(self.device)
         
    def train(self, train_input, train_target, num_epochs=1):
        """Implement training process"""
        self.nb_epochs = num_epochs
        # move input and target to the assigned device
        train_input = train_input.float().to(self.device)
        train_target = train_target.float().to(self.device)

        now = time.time()
        for e in range(0, self.nb_epochs):
            # set learning rate according to CosineAnnealingLR
            self.optimizer.CosineAnnealingLR(e, self.T_max)
            lr = self.optimizer.lr
            
            acc_loss = 0
            start = 0
            end = self.mini_batch_size

            # shuffle the dataset
            index = [i for i in range(train_input.shape[0])]
            random.shuffle(index)
            train_input = train_input[index]
            train_target = train_target[index]

            # do the batch stochastic gradient descent
            while start < len(train_input):
                if end >= len(train_input):
                    end = len(train_input)
                
                # forward propagation
                output = self.model.forward(train_input[start:end])
                output = output * 255.0

                loss = self.loss_function.forward(output, train_target[start:end])
                acc_loss += loss
                self.model.zero_grad()
                # backward propagation
                self.model.backward(self.loss_function.backward())
                # parameter update
                self.optimizer.step()

                start = end
                end += self.mini_batch_size

            consume_time = time.time() - now
            now = time.time()
            # print(f'epoch:{e+1:d}, loss:{acc_loss:.3f}, time:{consume_time:.1f}s, lr:{lr*1e7:.3f}*e-7')

    def predict(self, test_input):
        """give predicted result on test data"""
        test_input = test_input.float().to(self.device)
        test_output = self.model.forward(test_input)

        return test_output * 255.0

    def train_val(self, train_input, train_target, val_input, val_target, num_epochs=1):
        """Implement training while testing on validation data"""
        # if to record the training info in txt files
        is_record = False
        self.nb_epochs = num_epochs
        train_input = train_input.float().to(self.device)
        train_target = train_target.float().to(self.device)
        val_input = val_input.float().to(self.device)
        val_target = val_target.float().to(self.device)

        # testing the validation psnr before training
        val_output = self.model.forward(val_input)
        psnr = compute_psnr(val_output, val_target/255.0)
        print(f'before training, psnr for validation data is:{psnr:.3f}')

        if is_record:
            with open(f'others/is_record-T_max_{self.T_max}-lr_{self.learning_rate}.txt', 'w') as file:
                file.write(f'device: {self.device}, \n')

        now = time.time()
        for e in range(0, self.nb_epochs):
            self.optimizer.CosineAnnealingLR(e, self.T_max)
            lr = self.optimizer.lr
            acc_loss = 0
            start = 0
            end = self.mini_batch_size

            # shuffle dataset
            index = [i for i in range(train_input.shape[0])]
            random.shuffle(index)
            train_input = train_input[index]
            train_target = train_target[index]
        
            while start < len(train_input):
                if end >= len(train_input):
                    end = len(train_input)
                
                output = self.model.forward(train_input[start:end])
                output = output * 255.0

                loss = self.loss_function.forward(output, train_target[start:end])
                acc_loss += loss
                self.model.zero_grad()
                self.model.backward(self.loss_function.backward())           
                self.optimizer.step()
                
                start = end
                end += self.mini_batch_size

            consume_time = time.time() - now
            now = time.time()
            print(f'epoch:{e+1:d}, loss:{acc_loss:.3f}, time:{consume_time:.1f}s, lr:{lr*1e7:.3f}*e-7')
            
            val_output = self.model.forward(val_input)
            psnr = compute_psnr(val_output, val_target/255.0)
            print(f'after training {e+1:d} epoch, psnr for validation data is:{psnr:.3f}')

            if is_record:
                with open(f'others/is_record-T_max_{self.T_max}-lr_{self.learning_rate}.txt', 'a') as file:
                    file.write(f'epoch: {e+1:d}, loss: {acc_loss:.3f}, time: {consume_time:.1f}s, lr: {lr*1e7:.3f}*e-7 \t')
                    file.write(f'after training {e+1:d} epoch, psnr for validation data is: {psnr:.3f}\n')
        
        # save the model parameters
        model_name = f'others/bestmodel-T_max_{self.T_max}-lr_{self.learning_rate}.pkl'
        with open(model_name, 'wb') as f:
            pickle.dump(self.model.param(), f, pickle.HIGHEST_PROTOCOL)

    def valid(self, val_input, val_target):
        """testing the validation psnr for current model"""
        val_input = val_input.float().to(self.device)
        val_target = val_target.float().to(self.device)
        val_output = self.model.forward(val_input)
        psnr = compute_psnr(val_output, val_target/255.0)
        print(f'For the current model, psnr for validation data is:{psnr:.3f}')



def compute_psnr(X, Y, max_range=1.0):
    x, y = X, Y
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()
