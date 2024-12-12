import sys
import torch
from torch import nn
import torch.nn.functional as F

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        #self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        #self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        #combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output

class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut/5)
        n1 = (nOut - 4*n)
        
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16) # dilation rate of 2^4
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output

class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    def __init__(self, classes=20, p=8, q=6, r=4):
        super().__init__()
        self.level1 = CBR(3, 32, 3, 2)  # 채널 수 증가
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.sample3 = InputProjectionA(3)  # 새로운 샘플링 레벨 추가
        
        self.b1 = BR(32 + 3)
        
        self.level2_0 = DownSamplerB(32 + 3, 128)  # 채널 수 증가
        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(128, 128))
        
        self.b2 = BR(256 + 3)
        
        self.level3_0 = DownSamplerB(256 + 3, 256)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(256, 256))
        
        self.b3 = BR(512 + 3)
        
        # 새로운 레벨 추가
        self.level4_0 = DownSamplerB(512 + 3, 512)
        self.level4 = nn.ModuleList()
        for i in range(0, r):
            self.level4.append(DilatedParllelResidualBlockB(512, 512))
        
        self.b4 = BR(1024)
        
        self.classifier = C(1024, classes, 1, 1)

    def forward(self, input):
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        inp3 = self.sample3(input)  # 새로운 샘플링 레벨
        
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))
        
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        
        output2_cat = self.b3(torch.cat([output2_0, output2, inp3], 1))
        
        # 새로운 레벨 추가
        output3_0 = self.level4_0(output2_cat)
        for i, layer in enumerate(self.level4):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)
        
        output3_cat = self.b4(torch.cat([output3_0, output3], 1))
        
        classifier = self.classifier(output3_cat)
        return classifier

class ESPNet(nn.Module):
    def __init__(self, classes=20, p=6, q=16, r=4, encoderFile=None):
        super().__init__()
        self.encoder = ESPNet_Encoder(classes, p, q, r)
        if encoderFile is not None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            print('Encoder loaded!')
        
        self.modules = []
        for i, m in enumerate(self.encoder.children()):
            self.modules.append(m)
        
        # 디코더 부분 수정
        self.level3_C = C(256 + 3, classes, 1, 1)
        self.level4_C = C(512 + 3, classes, 1, 1)
        self.br = nn.BatchNorm2d(classes, eps=1e-03)
        self.conv = CBR(35 + classes, classes, 3, 1)
        
        self.up_l4 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l3_l4 = nn.Sequential(BR(2*classes), DilatedParllelResidualBlockB(2*classes, classes, add=False))
        
        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BR(2*classes), DilatedParllelResidualBlockB(2*classes, classes, add=False))
        
        self.up_l2 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False), BR(classes))
        self.classifier = nn.ConvTranspose2d(classes, 1, 2, stride=2, padding=0, output_padding=0, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Encoder 부분 (변경 없음)
        output0 = self.modules[0](input)
        inp1 = self.modules[1](input)
        inp2 = self.modules[2](input)
        inp3 = self.modules[3](input)
        
        output0_cat = self.modules[4](torch.cat([output0, inp1], 1))
        
        output1_0 = self.modules[5](output0_cat)
        for i, layer in enumerate(self.modules[6]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        
        output1_cat = self.modules[7](torch.cat([output1, output1_0, inp2], 1))
        
        output2_0 = self.modules[8](output1_cat)
        for i, layer in enumerate(self.modules[9]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        
        output2_cat = self.modules[10](torch.cat([output2_0, output2, inp3], 1))
        
        # 새로운 레벨
        output3_0 = self.modules[11](output2_cat)
        for i, layer in enumerate(self.modules[12]):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)
        
        output3_cat = self.modules[13](torch.cat([output3_0, output3], 1))
        
        # 디코더 부분 (수정됨)
        output3_c = self.up_l4(self.br(self.modules[14](output3_cat)))
        output2_C = self.level4_C(output2_cat)
        
        comb_l3_l4 = self.up_l3(self.combine_l3_l4(torch.cat([output2_C, output3_c], 1)))
        output1_C = self.level3_C(output1_cat)
        
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C, comb_l3_l4], 1)))
        
        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))
        classifier = self.classifier(concat_features)
        
        output = self.sigmoid(classifier)
        return output

'''
class ESPNet_Encoder(nn.Module):
    def __init__(self, classes=20, p=8, q=6, r=4, s=2):  # s 파라미터 추가
        super().__init__()
        self.level1 = CBR(3, 32, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.sample3 = InputProjectionA(3)
        self.sample4 = InputProjectionA(4)  # 새로운 샘플링 레벨 추가

        self.b1 = BR(32 + 3)
        self.level2_0 = DownSamplerB(32 + 3, 128)
        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(128, 128))
        
        self.b2 = BR(256 + 3)
        self.level3_0 = DownSamplerB(256 + 3, 256)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(256, 256))
        
        self.b3 = BR(512 + 3)
        self.level4_0 = DownSamplerB(512 + 3, 512)
        self.level4 = nn.ModuleList()
        for i in range(0, r):
            self.level4.append(DilatedParllelResidualBlockB(512, 512))
        
        self.b4 = BR(1024 + 3)  # 입력 채널 수 변경
        self.level5_0 = DownSamplerB(1024 + 3, 1024)  # 새로운 레벨 추가
        self.level5 = nn.ModuleList()
        for i in range(0, s):
            self.level5.append(DilatedParllelResidualBlockB(1024, 1024))
        
        self.b5 = BR(2048)
        self.classifier = C(2048, classes, 1, 1)

    def forward(self, input):
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        inp3 = self.sample3(input)
        inp4 = self.sample4(input)  # 새로운 샘플링 레벨

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2, inp3], 1))
        output3_0 = self.level4_0(output2_cat)
        for i, layer in enumerate(self.level4):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)

        output3_cat = self.b4(torch.cat([output3_0, output3, inp4], 1))  # inp4 추가
        output4_0 = self.level5_0(output3_cat)  # 새로운 레벨
        for i, layer in enumerate(self.level5):
            if i == 0:
                output4 = layer(output4_0)
            else:
                output4 = layer(output4)

        output4_cat = self.b5(torch.cat([output4_0, output4], 1))
        classifier = self.classifier(output4_cat)
        return classifier
    
class ESPNet(nn.Module):
    def __init__(self, classes=20, p=4, q=6, r=8, s=10, encoderFile=None):  # s 파라미터 추가
        super().__init__()
        self.encoder = ESPNet_Encoder(classes, p, q, r, s)
        if encoderFile is not None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            print('Encoder loaded!')
        
        self.modules = []
        for i, m in enumerate(self.encoder.children()):
            self.modules.append(m)

        # 디코더 부분 수정
        self.level3_C = C(256 + 3, classes, 1, 1)
        self.level4_C = C(512 + 3, classes, 1, 1)
        self.level5_C = C(1024 + 3, classes, 1, 1)  # 새로운 레벨 추가
        self.br = nn.BatchNorm2d(classes, eps=1e-03)
        self.conv = CBR(35 + classes, classes, 3, 1)
        
        self.up_l5 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l4_l5 = nn.Sequential(BR(2*classes), DilatedParllelResidualBlockB(2*classes, classes, add=False))
        self.up_l4 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l3_l4 = nn.Sequential(BR(2*classes), DilatedParllelResidualBlockB(2*classes, classes, add=False))
        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BR(2*classes), DilatedParllelResidualBlockB(2*classes, classes, add=False))
        self.up_l2 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False), BR(classes))
        
        self.classifier = nn.ConvTranspose2d(classes, 1, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Encoder 부분 (변경된 부분)
        output0 = self.modules[0](input)
        inp1 = self.modules[1](input)
        inp2 = self.modules[2](input)
        inp3 = self.modules[3](input)
        inp4 = self.modules[4](input)  # 새로운 샘플링 레벨

        output0_cat = self.modules[5](torch.cat([output0, inp1], 1))
        output1_0 = self.modules[6](output0_cat)
        for i, layer in enumerate(self.modules[7]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.modules[8](torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.modules[9](output1_cat)
        for i, layer in enumerate(self.modules[10]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.modules[11](torch.cat([output2_0, output2, inp3], 1))
        output3_0 = self.modules[12](output2_cat)
        for i, layer in enumerate(self.modules[13]):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)

        output3_cat = self.modules[14](torch.cat([output3_0, output3, inp4], 1))
        output4_0 = self.modules[15](output3_cat)
        for i, layer in enumerate(self.modules[16]):
            if i == 0:
                output4 = layer(output4_0)
            else:
                output4 = layer(output4)

        output4_cat = self.modules[17](torch.cat([output4_0, output4], 1))

        # 디코더 부분 (수정됨)
        output4_c = self.up_l5(self.br(self.modules[18](output4_cat)))
        output3_C = self.level5_C(output3_cat)
        comb_l4_l5 = self.up_l4(self.combine_l4_l5(torch.cat([output3_C, output4_c], 1)))
        output2_C = self.level4_C(output2_cat)
        comb_l3_l4 = self.up_l3(self.combine_l3_l4(torch.cat([output2_C, comb_l4_l5], 1)))
        output1_C = self.level3_C(output1_cat)
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C, comb_l3_l4], 1)))

        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))
        classifier = self.classifier(concat_features)
        output = self.sigmoid(classifier)

        return output
'''