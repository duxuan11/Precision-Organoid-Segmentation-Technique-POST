import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from itertools import product     

class InferQuantMatMul(nn.Module):
    def __init__(self, A_bit=8, B_bit=8, mode="raw"):
        super().__init__()
        self.A_bit=A_bit
        self.B_bit=B_bit
        self.A_qmax=2**(self.A_bit-1)
        self.B_qmax=2**(self.B_bit-1)
        self.mode=mode
        # self.split=split
        
    
    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["A_bit", "B_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        
        for sti in ["n_G_A", "n_V_A", "n_H_A", "n_G_B", "n_V_B", "n_H_B"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s 
    
    def get_parameter(self, A_interval, B_interval, n_G_A, n_V_A, n_H_A, n_G_B, n_V_B, n_H_B, crb_groups_A, crb_groups_B, crb_rows_A, crb_rows_B, crb_cols_A, crb_cols_B, pad_groups_A, pad_groups_B, pad_rows_A, pad_rows_B, pad_cols_A, pad_cols_B):
        self.register_buffer('A_interval', A_interval)
        self.register_buffer('B_interval', B_interval)
        
        self.register_buffer('n_G_A', torch.tensor(n_G_A, dtype=torch.int32))
        self.register_buffer('n_V_A', torch.tensor(n_V_A, dtype=torch.int32))
        self.register_buffer('n_H_A', torch.tensor(n_H_A, dtype=torch.int32))
        self.register_buffer('n_G_B', torch.tensor(n_G_B, dtype=torch.int32))
        self.register_buffer('n_V_B', torch.tensor(n_V_B, dtype=torch.int32))
        self.register_buffer('n_H_B', torch.tensor(n_H_B, dtype=torch.int32))
        self.register_buffer('crb_groups_A', torch.tensor(crb_groups_A, dtype=torch.int32))
        self.register_buffer('crb_groups_B', torch.tensor(crb_groups_B, dtype=torch.int32))
        self.register_buffer('crb_rows_A', torch.tensor(crb_rows_A, dtype=torch.int32))
        self.register_buffer('crb_rows_B', torch.tensor(crb_rows_B, dtype=torch.int32))
        self.register_buffer('crb_cols_A', torch.tensor(crb_cols_A, dtype=torch.int32))
        self.register_buffer('crb_cols_B', torch.tensor(crb_cols_B, dtype=torch.int32))
        self.register_buffer('pad_groups_A', torch.tensor(pad_groups_A, dtype=torch.int32))
        self.register_buffer('pad_groups_B', torch.tensor(pad_groups_B, dtype=torch.int32))
        self.register_buffer('pad_rows_A', torch.tensor(pad_rows_A, dtype=torch.int32))
        self.register_buffer('pad_rows_B', torch.tensor(pad_rows_B, dtype=torch.int32))
        self.register_buffer('pad_cols_A', torch.tensor(pad_cols_A, dtype=torch.int32))
        self.register_buffer('pad_cols_B', torch.tensor(pad_cols_B, dtype=torch.int32))
        # self.register_buffer('split', torch.tensor(split, dtype=torch.float32))
    
    def forward(self, A, B):
        if self.mode=='raw':
            out=A @ B
        elif self.mode=="quant_forward":
            out=self.quant_forward(A,B)
        else:
            raise NotImplementedError
        return out
    
    def quant_input_A(self, x):
        x = F.pad(x, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A])
        x = x.view(-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)
        x = (x/self.A_interval).round_().clamp(-self.A_qmax,self.A_qmax-1).mul_(self.A_interval)
        x = x.view(-1,self.n_G_A*self.crb_groups_A,self.n_V_A*self.crb_rows_A,self.n_H_A*self.crb_cols_A)
        x = x[:,:x.shape[1]-self.pad_groups_A,:x.shape[2]-self.pad_rows_A,:x.shape[3]-self.pad_cols_A]
        return x
    
    def quant_input_B(self, x):
        x = F.pad(x, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B])
        x = x.view(-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
        x = (x/self.B_interval).round_().clamp(-self.B_qmax,self.B_qmax-1).mul_(self.B_interval)
        x = x.view(-1,self.n_G_B*self.crb_groups_B,self.n_V_B*self.crb_rows_B,self.n_H_B*self.crb_cols_B)
        x = x[:,:x.shape[1]-self.pad_groups_B,:x.shape[2]-self.pad_rows_B,:x.shape[3]-self.pad_cols_B]
        return x
    
    def quant_forward(self, A, B):
        A_sim=self.quant_input_A(A)
        B_sim=self.quant_input_B(B)
        out=A_sim@B_sim
        return out

    
class InferQuantMatMulPost(nn.Module):
    def __init__(self, A_bit=8, B_bit=8, mode="raw"):
        super().__init__()
        self.A_bit=A_bit
        self.B_bit=B_bit
        self.A_qmax=2**(self.A_bit-1)
        self.B_qmax=2**(self.B_bit-1)
        self.mode=mode
        
    
    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["A_bit", "B_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        
        for sti in ["n_G_A", "n_V_A", "n_H_A", "n_G_B", "n_V_B", "n_H_B", "split"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s 

    def forward(self, A, B):
        if self.mode=='raw':
            out=A @ B
        elif self.mode=="quant_forward":
            out=self.quant_forward(A,B)
        else:
            raise NotImplementedError
        return out
    
    def get_parameter(self, A_interval, B_interval, n_G_A, n_V_A, n_H_A, n_G_B, n_V_B, n_H_B, crb_groups_A, crb_groups_B, crb_rows_A, crb_rows_B, crb_cols_A, crb_cols_B, pad_groups_A, pad_groups_B, pad_rows_A, pad_rows_B, pad_cols_A, pad_cols_B, split):
        self.register_buffer('A_interval', A_interval)
        self.register_buffer('B_interval', B_interval)
        self.register_buffer('n_G_A', torch.tensor(n_G_A, dtype=torch.int32))
        self.register_buffer('n_V_A', torch.tensor(n_V_A, dtype=torch.int32))
        self.register_buffer('n_H_A', torch.tensor(n_H_A, dtype=torch.int32))
        self.register_buffer('n_G_B', torch.tensor(n_G_B, dtype=torch.int32))
        self.register_buffer('n_V_B', torch.tensor(n_V_B, dtype=torch.int32))
        self.register_buffer('n_H_B', torch.tensor(n_H_B, dtype=torch.int32))
        self.register_buffer('crb_groups_A', torch.tensor(crb_groups_A, dtype=torch.int32))
        self.register_buffer('crb_groups_B', torch.tensor(crb_groups_B, dtype=torch.int32))
        self.register_buffer('crb_rows_A', torch.tensor(crb_rows_A, dtype=torch.int32))
        self.register_buffer('crb_rows_B', torch.tensor(crb_rows_B, dtype=torch.int32))
        self.register_buffer('crb_cols_A', torch.tensor(crb_cols_A, dtype=torch.int32))
        self.register_buffer('crb_cols_B', torch.tensor(crb_cols_B, dtype=torch.int32))
        self.register_buffer('pad_groups_A', torch.tensor(pad_groups_A, dtype=torch.int32))
        self.register_buffer('pad_groups_B', torch.tensor(pad_groups_B, dtype=torch.int32))
        self.register_buffer('pad_rows_A', torch.tensor(pad_rows_A, dtype=torch.int32))
        self.register_buffer('pad_rows_B', torch.tensor(pad_rows_B, dtype=torch.int32))
        self.register_buffer('pad_cols_A', torch.tensor(pad_cols_A, dtype=torch.int32))
        self.register_buffer('pad_cols_B', torch.tensor(pad_cols_B, dtype=torch.int32))
        self.register_buffer('split', torch.tensor(split, dtype=torch.float32))
    
    def quant_input_A(self, x):
        x_high = (x.clamp(self.split, 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
        x_low = (x.clamp(0, self.split)/self.A_interval).round_().clamp_(0,self.A_qmax-1)*self.A_interval
        return x_high + x_low
    
    def quant_input_B(self, x):
        x = F.pad(x, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B])
        x = x.view(-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
        x = (x/self.B_interval).round_().clamp(-self.B_qmax,self.B_qmax-1).mul_(self.B_interval)
        x = x.view(-1,self.n_G_B*self.crb_groups_B,self.n_V_B*self.crb_rows_B,self.n_H_B*self.crb_cols_B)
        x = x[:,:x.shape[1]-self.pad_groups_B,:x.shape[2]-self.pad_rows_B,:x.shape[3]-self.pad_cols_B]
        return x
    
    def quant_forward(self, A, B):
        A_sim=self.quant_input_A(A)
        B_sim=self.quant_input_B(B)
        out=A_sim@B_sim
        return out

    
class InferQuantLinear(nn.Linear):
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_correction = False):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
        self.bias_correction = bias_correction

    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["w_bit", "a_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        for sti in ["n_H", "n_V", "n_a"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def get_parameter(self, n_V, n_H, n_a, a_interval, w_interval, crb_rows, crb_cols, crb_acts):
        self.register_buffer('a_interval', a_interval)
        self.register_buffer('w_interval', w_interval)
        self.register_buffer('n_V', torch.tensor(n_V, dtype=torch.int32))
        self.register_buffer('n_H', torch.tensor(n_H, dtype=torch.int32))
        self.register_buffer('n_a', torch.tensor(n_a, dtype=torch.int32))
        self.register_buffer('crb_rows', torch.tensor(crb_rows, dtype=torch.int32))
        self.register_buffer('crb_cols', torch.tensor(crb_cols, dtype=torch.int32))
        self.register_buffer('crb_acts', torch.tensor(crb_acts, dtype=torch.int32))
    
    def quant_weight_bias(self):
        w = (self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        w_sim = w.mul_(self.w_interval).view(self.out_features,self.in_features)
        if self.bias is not None:
            return w_sim, self.bias
        else:
            return w_sim, None
    
    def quant_input(self, x):
        x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2)
        x_sim=(x_sim.div_(self.a_interval)).round_().clamp_(-self.a_qmax,self.a_qmax-1)
        x_sim = x_sim.mul_(self.a_interval).reshape_as(x)
        return x_sim

    def forward(self, x):
        if self.mode=='raw':
            out=F.linear(x, self.weight, self.bias)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_forward(self,x):
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x)
        out=F.linear(x_sim, w_sim, bias_sim)
        return out

    
class InferQuantLinearPost(nn.Linear):
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_correction = False):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
        self.bias_correction = bias_correction
        tmp_a_neg_interval = torch.tensor(0.16997124254703522/self.a_qmax)
        self.register_buffer('a_neg_interval', tmp_a_neg_interval)

    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["w_bit", "a_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        for sti in ["n_H", "n_V", "n_a"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def get_parameter(self, n_V, n_H, n_a, a_interval, w_interval, crb_rows, crb_cols, crb_acts):
        self.register_buffer('a_interval', a_interval)
        self.register_buffer('w_interval', w_interval)
        self.register_buffer('n_V', torch.tensor(n_V, dtype=torch.int32))
        self.register_buffer('n_H', torch.tensor(n_H, dtype=torch.int32))
        self.register_buffer('n_a', torch.tensor(n_a, dtype=torch.int32))
        self.register_buffer('crb_rows', torch.tensor(crb_rows, dtype=torch.int32))
        self.register_buffer('crb_cols', torch.tensor(crb_cols, dtype=torch.int32))
        self.register_buffer('crb_acts', torch.tensor(crb_acts, dtype=torch.int32))
    
    def quant_weight_bias(self):
        w = (self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        w_sim = w.mul_(self.w_interval).view(self.out_features,self.in_features)
        if self.bias is not None:
            return w_sim, self.bias
        else:
            return w_sim, None
    
    def quant_input(self, x):
        x_=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2)
        x_pos=(x_/(self.a_interval)).round_().clamp_(0,self.a_qmax-1).mul_(self.a_interval)
        x_neg=(x_/(self.a_neg_interval)).round_().clamp_(-self.a_qmax,0).mul_(self.a_neg_interval)
        return (x_pos + x_neg).reshape_as(x)

    def forward(self, x):
        if self.mode=='raw':
            out=F.linear(x, self.weight, self.bias)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_forward(self,x):
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x)
        out=F.linear(x_sim, w_sim, bias_sim)
        return out

    
class InferQuantConv2d(nn.Conv2d):
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',mode='raw',w_bit=8,a_bit=8):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.mode=mode
        self.w_bit=w_bit
        self.a_bit=a_bit
        # self.bias_bit=bias_bit
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
       
        
    def get_parameter(self, n_V, n_H, a_interval, a_bias, w_interval):
        self.register_buffer('a_interval', a_interval)
        self.register_buffer('a_bias', a_bias)
        self.register_buffer('w_interval', w_interval)
        self.register_buffer('n_V', torch.tensor(n_V, dtype=torch.int32))
        self.register_buffer('n_H', torch.tensor(n_H, dtype=torch.int32))
        
    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["w_bit", "a_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        for sti in ["n_H", "n_V"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
        
    def forward(self, x):
        if self.mode=='raw':
            out=F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        else:
            raise NotImplementedError
        return out
            
    def quant_weight_bias(self):
        self.w_interval = torch.tensor([[0.02232402376830578]])
        self.a_interval = torch.tensor([
            [
        [
            [
                0.016292691230773926
            ]
        ]
    ]
])
        w_sim = (self.weight/self.w_interval).round_().clamp(-self.w_qmax,self.w_qmax-1).mul_(self.w_interval)
        return w_sim, self.bias
    
    def quant_input(self, x):        
        self.a_bias = torch.tensor([
    -0.6949046850204468,
    -0.7216329574584961,
    -0.48391973972320557,
    -0.40439724922180176,
    -0.605524480342865,
    -1.3006396293640137,
    -1.0902070999145508,
    0.3133399188518524,
    -1.0232536792755127,
    -1.1148085594177246,
    -0.7060199975967407,
    -1.0006115436553955,
    -0.6791764497756958,
    -0.5665738582611084,
    -1.5888776779174805,
    -1.1368283033370972,
    0.634133517742157,
    -0.8419190645217896,
    -0.4875425100326538,
    -0.4789593517780304,
    -0.8248142004013062,
    0.2820151150226593,
    -0.6320134997367859,
    -0.6719217896461487,
    -1.0414294004440308,
    -0.8101732134819031,
    -0.15713725984096527,
    -0.5809568166732788,
    -1.0992231369018555,
    -0.743283748626709,
    0.004000641405582428,
    -0.5510671138763428,
    -0.8344907760620117,
    -0.5864059925079346,
    -0.5682993531227112,
    -0.6996340751647949,
    -0.8574531078338623,
    -0.3890336751937866,
    -0.4607682526111603,
    0.2385789453983307,
    -1.1076576709747314,
    -0.12229060381650925,
    -1.4709420204162598,
    -0.9163281917572021,
    -0.6831067204475403,
    -0.7742517590522766,
    -0.7359070777893066,
    -0.996707558631897,
    -0.7686411142349243,
    -0.4083199203014374,
    -1.092383623123169,
    -0.8447937369346619,
    -0.6579700112342834,
    0.13502712547779083,
    -0.42704975605010986,
    -1.388222336769104,
    -0.5102431178092957,
    -0.35564714670181274,
    -1.4411418437957764,
    -0.14611005783081055,
    -0.349419504404068,
    -0.4021708071231842,
    -0.5657480955123901,
    -0.7562534213066101,
    -0.12644554674625397,
    -0.9101453423500061,
    -2.124105215072632,
    -0.4681518077850342,
    -0.3119107782840729,
    -0.7743873000144958,
    -0.41062667965888977,
    0.05069202557206154,
    -0.6462703943252563,
    -0.43139126896858215,
    -0.3994510769844055,
    -0.5118857622146606,
    -0.5655471086502075,
    -1.300215721130371,
    -0.6868348121643066,
    -1.0243045091629028,
    -0.4155464470386505,
    -0.7563987970352173,
    -0.3306097686290741,
    -0.6858069896697998,
    -0.8530672192573547,
    -0.3061814308166504,
    -0.485172837972641,
    -0.8790435791015625,
    -0.5182096362113953,
    -0.9335281252861023,
    -0.26018431782722473,
    -0.6047959327697754,
    -0.684308648109436,
    -0.15100224316120148,
    -0.7282015681266785,
    -0.7916572093963623,
    -0.7488840222358704,
    -0.19490228593349457,
    -0.48992085456848145,
    -0.4795021116733551,
    -1.2640584707260132,
    -0.029534168541431427,
    -0.5025352239608765,
    -0.37023356556892395,
    -0.5434329509735107,
    -0.2721653878688812,
    -0.5787288546562195,
    -1.0565983057022095,
    -0.6897562742233276,
    -0.36091238260269165,
    -0.9399119019508362,
    -0.7614229917526245,
    -0.6503691673278809,
    -0.8814406394958496,
    -0.04566317796707153,
    -0.6830148100852966,
    -1.1334595680236816,
    -0.8717626929283142,
    -0.22189192473888397,
    -0.30453798174858093,
    -0.875592052936554,
    -0.4904076159000397,
    -1.4628188610076904,
    -0.42318814992904663,
    -0.5127615928649902,
    0.3002058267593384,
    -1.2879389524459839,
    -0.6275812387466431,
    -0.38697242736816406,
    -0.22237926721572876,
    -0.6794669032096863,
    -0.7871740460395813,
    -0.24704284965991974,
    -0.3434857130050659,
    -1.1418285369873047,
    -0.723748505115509,
    -0.905498206615448,
    -0.5683204531669617,
    -0.7848789095878601,
    -0.41926223039627075,
    -0.2799694538116455,
    -0.1992325335741043,
    -0.30870601534843445,
    -1.063430666923523,
    -1.0091047286987305,
    -0.8210983872413635,
    -0.46659550070762634,
    -0.2788604199886322,
    -0.831149697303772,
    0.07825686782598495,
    -0.6797353625297546,
    -0.8197298645973206,
    0.2772468030452728,
    -0.6674802303314209,
    -0.5316769480705261,
    -0.05815824493765831,
    -0.744340717792511,
    -0.9998390674591064,
    -0.9196239113807678,
    -0.594813883304596,
    -0.40260574221611023,
    -0.2763810455799103,
    -0.9394387006759644,
    -0.4891781806945801,
    -0.9491142630577087,
    -0.5655480623245239,
    -0.28436142206192017,
    -0.8692461848258972,
    -0.5697410702705383,
    -0.3236825466156006,
    0.1467359960079193,
    -0.8442308306694031,
    -0.7212568521499634,
    -0.7997111678123474,
    0.0026740191970020533,
    -0.5171831250190735,
    -0.4501018226146698,
    -0.7825150489807129,
    -0.4415154457092285,
    -0.495359867811203,
    -0.8124843835830688,
    -1.2110331058502197,
    -0.40668052434921265,
    -0.856233537197113,
    -0.13714928925037384,
    -0.6052815914154053,
    -1.2782613039016724,
    -0.4581666588783264,
    -0.6971408724784851,
    -0.6809128522872925,
    0.046855174005031586,
    -0.5437613129615784,
    -0.8065581917762756,
    -0.38970059156417847,
    -0.8614296317100525,
    -0.4892770051956177,
    -0.7332187294960022,
    -0.5167959332466125,
    -0.5403949022293091,
    -0.5603113770484924,
    -0.530271589756012,
    -1.193116545677185,
    -0.6742504835128784,
    -0.6723796725273132,
    -0.3734308183193207,
    -0.7003547549247742,
    -0.7991166114807129,
    -1.1993913650512695,
    -0.7082507610321045,
    1.0854589939117432,
    -0.7734136581420898,
    -1.0543092489242554,
    -0.6460310816764832,
    -0.34603258967399597,
    -0.6741946935653687,
    -1.023760199546814,
    -1.5955686569213867,
    -0.8144329190254211,
    -0.9782738089561462,
    -1.0438694953918457,
    -0.7076583504676819,
    -0.9977332353591919,
    -1.100246548652649,
    -0.5676989555358887,
    -0.6706930994987488,
    -0.42315635085105896,
    -0.2322579175233841,
    -0.8127763867378235,
    -0.5599087476730347,
    -0.560823380947113,
    -0.4557574689388275,
    -0.8341269493103027,
    -0.5691180229187012,
    -0.8895204663276672,
    -0.9789706468582153,
    -0.37228649854660034,
    -0.49032455682754517,
    -0.23465460538864136,
    -0.4009931683540344,
    -0.38161328434944153,
    -0.7079970836639404,
    -1.2028400897979736,
    -0.46677473187446594,
    -0.41387075185775757,
    -1.0541249513626099,
    -0.8637311458587646,
    -0.565570592880249,
    -0.4964217245578766,
    -0.5172322392463684,
    -0.46622297167778015,
    -0.7688421010971069,
    -0.5037198662757874,
    -0.8939476013183594,
    -0.3812701106071472,
    -0.5608036518096924,
    -0.36686667799949646,
    -0.8277813196182251,
    -0.30286896228790283,
    -0.6837366223335266,
    -0.7177178859710693,
    -0.4828799068927765,
    -0.871724009513855,
    -0.13887839019298553,
    -1.1816692352294922,
    0.20449909567832947,
    -0.9376136064529419,
    -1.0933330059051514,
    -0.3663506507873535,
    -1.1662169694900513,
    -0.5600903034210205,
    -0.8418431282043457,
    -0.8282309174537659,
    -0.387681782245636,
    -0.6971222162246704,
    -0.783776044845581,
    -1.016007423400879,
    -1.2545645236968994,
    -0.4783110022544861,
    0.33916178345680237,
    -1.2721062898635864,
    -0.5581439733505249,
    -0.5625239610671997,
    -0.7221453785896301,
    -0.6496065855026245,
    -0.2972140312194824,
    -0.31074926257133484,
    -1.2185250520706177,
    -0.2412576675415039,
    -0.9290623664855957,
    -0.8576594591140747,
    -0.38565194606781006,
    -1.3323098421096802,
    -0.46885955333709717,
    -0.4858829975128174,
    -0.5450423955917358,
    -0.8574669361114502,
    -0.44304606318473816,
    -0.3473599851131439,
    -0.8386315107345581,
    -0.901455283164978,
    -0.625896155834198,
    -0.8760284781455994,
    -0.5238640308380127,
    -0.6618212461471558,
    -0.43242958188056946,
    -0.11846377700567245,
    -0.4859451651573181,
    0.746069073677063,
    0.819183349609375,
    0.12204517424106598,
    -1.0472813844680786,
    -0.49090254306793213,
    -0.5557078719139099,
    -1.0638068914413452,
    -0.4983268082141876,
    -0.7287279963493347,
    -0.4494047462940216,
    -0.7208350300788879,
    -0.8855769038200378,
    -0.666156530380249,
    -0.5017973780632019,
    -1.510896921157837,
    -0.6322498321533203,
    -1.1322368383407593,
    -0.4801676869392395,
    -0.4840673506259918,
    -0.458454430103302,
    -0.553544282913208,
    0.49580588936805725,
    -0.6123431324958801,
    -0.2539658844470978,
    -0.5784962773323059,
    -1.159169316291809,
    -0.3700701594352722,
    -0.8983281254768372,
    -0.6026161313056946,
    -0.0958731472492218,
    -0.5584204196929932,
    -1.1714125871658325,
    -0.7564213871955872,
    -0.6038300395011902,
    -0.6864477396011353,
    0.1865355223417282,
    -0.07402926683425903,
    -0.29421505331993103,
    0.03484708070755005,
    -0.004092344082891941,
    -0.07525483518838882,
    -0.6159265637397766,
    -0.32264459133148193,
    -0.6454939246177673,
    -0.6184969544410706,
    -0.43886831402778625,
    -0.8588863611221313,
    -0.7950125932693481,
    -0.3332905173301697,
    -1.0680429935455322,
    -0.32393699884414673,
    -0.44808289408683777,
    -0.7066627740859985,
    -1.0357792377471924,
    -1.3042426109313965,
    -0.1588342934846878,
    -0.94466632604599,
    -0.49590492248535156,
    -1.089514970779419,
    -0.46114581823349,
    -1.1990759372711182,
    -0.5269736647605896,
    -0.6577815413475037,
    -0.5001186728477478,
    -0.9337634444236755,
    -0.8297502994537354,
    -0.5877014994621277,
    -0.6533249020576477,
    -1.050358533859253,
    -0.7720218896865845,
    -0.5674206018447876,
    -1.0338172912597656,
    -0.19748809933662415,
    -0.9031424522399902,
    -1.0578644275665283,
    -1.2550601959228516,
    -0.49178412556648254,
    -0.8388397097587585,
    -0.7241265177726746,
    -0.5083971619606018,
    -0.7104106545448303,
    -0.764183521270752,
    -0.7762765288352966,
    -0.6462950706481934,
    -0.6000316143035889,
    -0.7145196795463562,
    -0.7674428820610046,
    -0.31351277232170105,
    -0.6108216047286987,
    -0.9548247456550598,
    -0.8540343046188354,
    -0.7025559544563293,
    -0.31204381585121155,
    -0.5773926973342896,
    -0.9078794121742249,
    -0.24583123624324799,
    -0.6981921195983887,
    -0.8191741704940796,
    -0.36092376708984375,
    -0.2940216362476349,
    -0.530894935131073,
    -0.42464780807495117,
    -0.44252103567123413,
    -0.295013427734375,
    -0.2821040153503418,
    -0.42663294076919556,
    -0.8628617525100708,
    -0.28385940194129944,
    -0.22372066974639893,
    -0.5208307504653931,
    -0.9766630530357361,
    -0.8156775832176208,
    -0.4657578468322754,
    -0.5158557295799255,
    -0.5483227968215942,
    -0.26923197507858276,
    -1.322176218032837,
    -0.20945565402507782,
    0.36914360523223877,
    -1.0109483003616333,
    -0.6437960863113403,
    -1.0551725625991821,
    -0.5329228043556213,
    -0.49088621139526367,
    -0.2421131134033203,
    -1.155658483505249,
    -0.7220370173454285,
    -0.7885541915893555,
    -0.4763799011707306,
    -0.8855450749397278,
    -1.0187212228775024,
    -0.5619900822639465,
    0.3845537006855011,
    -0.4987396001815796,
    -0.6484185457229614,
    -0.6093491911888123,
    -1.0734586715698242,
    -0.369630366563797,
    -0.7358591556549072,
    -0.1906411051750183,
    -0.9233601093292236,
    -0.6409560441970825,
    -1.1643649339675903,
    -0.5975015759468079,
    -1.461387038230896,
    -0.13210351765155792,
    -1.0289605855941772,
    -0.8562192320823669,
    -0.7698153257369995,
    -0.4479040801525116,
    -0.43573832511901855,
    0.18127086758613586,
    -0.8082383871078491,
    -1.136914849281311,
    -0.8997023105621338,
    -0.5192615985870361,
    -0.3253912627696991,
    -0.6508723497390747,
    -0.3306044340133667,
    -0.49116501212120056,
    -0.26262062788009644,
    -0.3593083918094635,
    -0.47805753350257874,
    -0.5515015721321106,
    -0.5049146413803101,
    -0.5799016356468201,
    -0.5003159046173096,
    -1.1686242818832397,
    -0.42175596952438354,
    -1.0156100988388062,
    -0.5146453380584717,
    -0.30226683616638184,
    -0.5477913022041321,
    -0.44946861267089844,
    -0.4855087101459503,
    -0.4579886496067047,
    -0.5097907781600952,
    -0.6623454093933105,
    -0.5838962197303772,
    -0.5083492994308472,
    -0.9853030443191528,
    -1.0605710744857788,
    -0.20395980775356293,
    -0.5167202353477478,
    -0.49385255575180054,
    -0.169283926486969,
    -0.5064892172813416,
    -0.7646316289901733,
    0.22153858840465546,
    -0.8589149713516235,
    -0.5366636514663696,
    -0.8722959160804749,
    -0.5627315640449524,
    -0.7715710401535034,
    -0.932154655456543,
    -0.5437098741531372,
    -0.7675414085388184,
    -0.7682868242263794,
    -0.8922535181045532,
    -0.379280686378479,
    -0.6435239911079407,
    0.13312305510044098,
    0.45399755239486694,
    -1.0691379308700562,
    -0.7456293702125549
])
        aq = (x - self.a_bias)/ self.a_interval
        aq = aq.round_().clamp_(-self.a_qmax, self.a_qmax-1)
        x_sim = aq * self.a_interval  + self.a_bias
        return x_sim
    
    def quant_forward(self, x):
        w_sim, bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x) if self.a_bit < 32 else x
        out=F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out

    
class InferQuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation = 1,
        padding_mode: str = 'zeros',mode='raw',w_bit=8,a_bit=8):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.w_bit=w_bit
        self.a_bit=a_bit
        self.mode=mode
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
        
    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["w_bit", "a_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        for sti in ["n_H", "n_V"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
                
    def quant_weight_bias(self):
        w_sim = (self.weight/self.w_interval).round_().clamp(-self.w_qmax,self.w_qmax-1).mul_(self.w_interval)
        return w_sim, self.bias
    
    def quant_input(self, x):
        aq = (x - self.a_bias)/ self.a_interval
        aq = aq.round_().clamp_(-self.a_qmax, self.a_qmax-1)
        x_sim = aq * self.a_interval  + self.a_bias
        return x_sim

    def get_parameter(self, n_V, n_H, a_interval, a_bias, w_interval):
        self.register_buffer('a_interval', a_interval)
        self.register_buffer('a_bias', a_bias)
        self.register_buffer('w_interval', w_interval)
        self.register_buffer('n_V', torch.tensor(n_V, dtype=torch.int32))
        self.register_buffer('n_H', torch.tensor(n_H, dtype=torch.int32))

    def quant_forward(self, x):
        w_sim, bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x) if self.a_bit < 32 else x
        out=F.conv_transpose2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        return out

    def forward(self, x):
        if self.mode=='raw':
            out=F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        else:
            raise NotImplementedError
        return out