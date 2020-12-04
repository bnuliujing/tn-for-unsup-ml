import torch
import torch.nn as nn
import torch.nn.functional as F

class MPS_pytorch_sharing_input(nn.Module):
    def __init__(self,h, Dmax=2,seed=50,feature_dim=2, init_std=1e-9,fixed_bias=True,device='cuda:0'):
        super().__init__()
        self.h=h
        self.width =2*h+1
        self.height=h+1
        self.Dmax = Dmax
        self.feature_dim = feature_dim
        self.fixed_bias = fixed_bias
        self.device=device
        torch.manual_seed(seed)

        shape = [(self.height-1)*self.width+h, Dmax, Dmax, feature_dim]
        self.tensors1 = nn.Parameter(init_std * torch.randn(shape))

        bias_mat = torch.eye(Dmax).unsqueeze(0)#[1,D,D]
        if fixed_bias>0:
            self.register_buffer(name='bias_mat', tensor=bias_mat)
        else:
            print('fixed_bias==False')
            self.register_parameter(name='bias_mat', param=nn.Parameter(bias_mat))


    def forward(self, input_data):
        batch_size = input_data.size(0)
        embedded_data = torch.stack([input_data, 1 - input_data], dim=4).view(input_data.size(1) * batch_size,
                                                                              self.height * self.width, 2)
        vec = torch.zeros(self.Dmax).to(self.device)
        vec[0] = 1
        vec = vec.expand([input_data.size(1) * batch_size] + [self.Dmax])
        left_vec = vec.unsqueeze(1)  # [s*b,1,D]
        for i in range((self.height-1)*self.width+self.h):
            assert not torch.isnan(self.tensors1[i, :, :, :]).any()
            assert not torch.isnan(embedded_data[:, i, :]).any()
            mats = torch.einsum('lri,bi->blr', [self.tensors1[i, :, :, :],embedded_data[:, i, :]])
            assert not torch.isnan(mats).any()
            mats = mats + self.bias_mat.expand_as(mats)
            assert not torch.isnan(mats).any()
            left_vec = torch.bmm(left_vec, mats)

        return left_vec.view(batch_size, input_data.size(1), self.Dmax)



class MPS_pytorch_sharing(nn.Module):
    def __init__(self,h, Dmax=2,seed=50, init_std=1e-9, fixed_bias=True,device='cuda:0'):
        super().__init__()
        self.h=h
        self.width =2*h+1
        self.height=h+1
        self.Dmax = Dmax
        self.fixed_bias = fixed_bias
        self.device=device
        torch.manual_seed(seed)

        shape = [(self.height-1)*self.width+h+1, Dmax, Dmax,Dmax]
        self.tensors1 = nn.Parameter(init_std * torch.randn(shape))
        bias_mat = torch.eye(Dmax).unsqueeze(0)  # [1,D,D]
        if fixed_bias >0:
            self.register_buffer(name='bias_mat', tensor=bias_mat)

        else:
            self.register_parameter(name='bias_mat', param=nn.Parameter(bias_mat))



    def forward(self,samples):
        batch_size=samples.shape[0]
        vec = torch.zeros(self.Dmax).to(self.device)
        vec[0] = 1
        vec = vec.expand([batch_size * samples.size(1)] + [self.Dmax])
        left_vec = vec.unsqueeze(1)  # [s*b,1,D]
        samples = samples.reshape(samples.shape[0]*samples.shape[1], samples.shape[2] * samples.shape[3], self.Dmax)
        for i in range((self.height-1)*self.width+self.h+1):
            mats = torch.einsum('bd,lrd->blr', samples[:, i, :], self.tensors1[i, :, :,:])
            mats = mats + self.bias_mat.expand_as(mats)
            left_vec = torch.bmm( left_vec, mats)#[s*b,1,D]
        return left_vec.view(batch_size,-1,left_vec.shape[2])


class MPS_pytorch_sharing_output(nn.Module):
    def __init__(self,h, Dmax=2,seed=50,init_std=1e-9, fixed_bias=True,device='cuda:0'):
        super().__init__()
        self.h=h
        self.width =2*h+1
        self.height=h+1
        self.Dmax = Dmax
        self.fixed_bias = fixed_bias
        self.device=device
        torch.manual_seed(seed)

        shape = [(self.height-1)*self.width+h+1, Dmax, Dmax,Dmax]
        shape2 = [ Dmax, Dmax,2]
        self.tensors = nn.Parameter(init_std * torch.randn(shape))
        self.tensors2 = nn.Parameter(init_std * torch.randn(shape2))
        bias_mat = torch.eye(Dmax).unsqueeze(0)  # [1,D,D]
        bias_mat2 = torch.stack([torch.eye(Dmax) for _ in range(2)],dim=2)
        if fixed_bias>0:
            self.register_buffer(name='bias_mat', tensor=bias_mat)
            self.register_buffer(name='bias_mat2', tensor=bias_mat2)
        else:
            self.register_parameter(name='bias_mat', param=nn.Parameter(bias_mat))
            self.register_parameter(name='bias_mat2',param=nn.Parameter(bias_mat2))

    def forward(self,samples):
        batch_size = samples.size(0)
        vec = torch.zeros(self.Dmax).to(self.device)
        vec[0] = 1
        vec = vec.expand([batch_size * samples.size(1)] + [self.Dmax])
        left_vec = vec.unsqueeze(1)  # [s*b,1,D]
        samples = samples.reshape(samples.shape[0] * samples.shape[1], samples.shape[2] * samples.shape[3],self.Dmax)
        for i in range((self.height-1)*self.width+self.h+1):
            mats = torch.einsum('bd,lrd->blr', samples[:, i, :], self.tensors[i, :, :,:])
            mats = mats + self.bias_mat.expand_as(mats)
            left_vec = torch.bmm(left_vec, mats)
        output=torch.einsum('bd,di->bi',left_vec.squeeze(),(self.tensors2+self.bias_mat2)[:,0,:]).view(batch_size, 784,2)
        assert not torch.isnan(output).any()
        return F.log_softmax(output, dim=2)


class AutoregressiveMPS_sharing(nn.Module):
    def __init__(self,select, h, Dmax=2, seed=42, init_std=1e-9, fixed_bias=True, device='cuda:0'):
        super(AutoregressiveMPS_sharing, self).__init__()
        self.h=h
        self.width = 2*h + 1
        self.height = h + 1
        self.Dmax=Dmax
        self.seed = seed
        self.device = device
        self.init_std=init_std
        self.fixed_bias=fixed_bias
        if self.seed > 0:
            torch.manual_seed(self.seed)
        self.select=select
        if self.select==0:
            self.MPS_model=MPS_pytorch_sharing_input(h, Dmax=Dmax,seed=seed,feature_dim=2, init_std=init_std, fixed_bias=fixed_bias,device=device)
        elif self.select==1:
            self.MPS_model=MPS_pytorch_sharing(h, Dmax=Dmax,seed=seed, init_std=init_std, fixed_bias=fixed_bias,device=device)
        else:
            self.MPS_model=MPS_pytorch_sharing_output(h, Dmax=Dmax,seed=seed, init_std=init_std, fixed_bias=fixed_bias,device=device)


    def forward(self,samples):
        batch_size = samples.shape[0]
        if self.select != 0:
            m = torch.nn.ConstantPad2d((self.h, self.h, self.h, 0), 0)
            samples = m(samples.view(batch_size, 28, 28, self.Dmax).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.select == 2:
            data = torch.zeros([batch_size, 784, self.height, self.width, self.Dmax]).to(self.device)
            x_hat = torch.zeros([batch_size, 784,2], device=self.device)
            for i in range(28):
                for j in range(28):
                    data[:, i * 28 + j, :, :, :] = samples[:, i:i + self.height, j:j + self.width, :]
            x_hat = self.MPS_model(data)

        else:
            x_hat = torch.zeros([batch_size, 784, self.Dmax], device=self.device)
            if self.select == 0:
                data = torch.zeros([batch_size, 784, self.height, self.width]).to(self.device)
                for i in range(28):
                    for j in range(28):
                        data[:, i * 28 + j, :, :] = samples[:, i:i + self.height, j:j + self.width]
                x_hat = self.MPS_model(data)
            else:
                data = torch.zeros([batch_size, 784, self.height, self.width, self.Dmax]).to(self.device)
                for i in range(28):
                    for j in range(28):
                        data[:, i * 28 + j, :, :, :] = samples[:, i:i + self.height, j:j + self.width, :]
                x_hat = self.MPS_model(data)
        return x_hat
    




