import sys
from tracemalloc import start
import torch
from torch import nn
import torch.nn.functional as F

class DINOv2_ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        
        for i, (name, param) in enumerate(self.dinov2_vitg14.named_parameters()):
            param.requires_grad = False
        
    def forward(self, img):
        x = self.dinov2_vitg14.get_intermediate_layers(img)[0]
        output = x
        
        return output

class DepthEncoder(nn.Module):
    def __init__(self, in_channels, output_channels, num_patch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=14, stride=14)
        self.norm1 = nn.LayerNorm([32, num_patch, num_patch])
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm([64, num_patch, num_patch])
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm([128, num_patch, num_patch])
        
        self.conv4 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        self.norm4 = nn.LayerNorm([output_channels, num_patch, num_patch])
        
        self.gelu = nn.GELU()
        
    def forward(self, depth):
        x = self.conv1(depth)
        x = self.gelu(self.norm1(x))
        
        x = self.conv2(x)
        x = self.gelu(self.norm2(x))
        
        x = self.conv3(x)
        x = self.gelu(self.norm3(x))
        
        x = self.conv4(x)
        x = self.gelu(self.norm4(x))
        
        output = x
        
        return output

class LinearClassifier(nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = nn.Conv2d(in_channels, num_labels, (1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings):
        # embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        # embeddings = embeddings.permute(0,3,1,2)

        output = self.classifier(embeddings)
        output = self.sigmoid(output)
        
        return output

class DINOv2_SegHead(nn.Module):
    def __init__(self, in_channels, output_channels, num_patch, img_shape):
        super().__init__()

        self.img_shape = img_shape
        self.classifier = LinearClassifier(in_channels, num_patch, num_patch, output_channels)

    def forward(self, x):
        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(x)
        logits = F.interpolate(logits, size=self.img_shape, mode="nearest")#, align_corners=False)
        
        return logits

'''
class SegHead(nn.Module):
    def __init__(self, in_channels, output_channels, num_patch):
        super().__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm([128, num_patch, num_patch])
        
        self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm([64, num_patch, num_patch])
        
        self.convt3 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm([32, num_patch, num_patch])
        
        self.convt4 = nn.ConvTranspose2d(32, output_channels, kernel_size=14, stride=14)
        
        self.gelu = nn.GELU()
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.convt1(x)
        x = self.gelu(self.norm1(x))
        
        x = self.convt2(x)
        x = self.gelu(self.norm2(x))
        
        x = self.convt3(x)
        x = self.gelu(self.norm3(x))
        
        x = self.convt4(x)
        output = self.sigmoid(x)
        
        return output
'''

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 논문의 파란색 화살표
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            # Conv2d layer 정의
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        # 좌측 레이어 enc (인코더)
        # 1층 좌측 첫번째 레이어 두개 
        self.enc1_1 = CBR2d(in_channels=3, out_channels =64, kernel_size =3, stride=1, padding =1, bias =True)
        self.enc1_2 = CBR2d(in_channels=64, out_channels =64, kernel_size =3, stride=1, padding =1, bias =True)

        # 다음 빨간색 화살표 max_pool 2*2
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 2층 파란색 화살표     
        self.enc2_1 = CBR2d(in_channels=64, out_channels =128, kernel_size =3, stride=1, padding =1, bias =True)
        self.enc2_2 = CBR2d(in_channels=128, out_channels =128, kernel_size =3, stride=1, padding =1, bias =True)
        
        # 다음 빨간색 화살표 max_pool 2*2
        self.pool2 = nn.MaxPool2d(kernel_size =2)

        # 3층 파란색 화살표 
        self.enc3_1 = CBR2d(in_channels=128, out_channels =256, kernel_size =3, stride=1, padding =1, bias =True)
        self.enc3_2 = CBR2d(in_channels=256, out_channels =256, kernel_size =3, stride=1, padding =1, bias =True)

        # 다음 빨간색 화살표 max_pool 2*2
        self.pool3 = nn.MaxPool2d(kernel_size =2)


        # 4층 파란색 화살표 
        self.enc4_1 = CBR2d(in_channels=256, out_channels =512, kernel_size =3, stride=1, padding =1, bias =True)
        self.enc4_2 = CBR2d(in_channels=512, out_channels =512, kernel_size =3, stride=1, padding =1, bias =True)

        # 다음 빨간색 화살표 max_pool 2*2
        self.pool4 = nn.MaxPool2d(kernel_size =2)

        # 5층 파란색 화살표
        self.enc5_1 = CBR2d(in_channels=512, out_channels =1024, kernel_size =3, stride=1, padding =1, bias =True)


        # Expansive path
        
        # 5층 파란색 2번쨰 화살표인데 디코더로
        self.dec5_1 = CBR2d(in_channels=1024, out_channels =512, kernel_size =3, stride=1, padding =1, bias =True)

        # 초록색 화살표
        self.unpool4 = nn.ConvTranspose2d(in_channels =512, out_channels = 512, kernel_size = 2, stride = 2, padding = 0, bias = True)

        # enc4_2와 대칭이되는 점을 보면 dec4_2 input값은 512가 맞는데, unet 아키텍쳐를 보니
        # enc4_2 에서 회색 화살표로 dec4_2로 와서 copy and crop이 일어남
        # 따라서 dec4_2 in_channels = 1024로 설정
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels =512, kernel_size =3, stride=1, padding =1, bias =True)
        self.dec4_1 = CBR2d(in_channels=512, out_channels =256, kernel_size =3, stride=1, padding =1, bias =True)

        # 초록색 화살표
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # 3층 파란색 화살표
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        # 초록색 화살표
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        # 2층 파란색 화살표
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
        
        # 초록색 화살표
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        # 1층 파란색 화살표
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # segmentation에 필요한 n개의 클래스에 대한 output 정의
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # 좌측 1층 레이어 2개 연결 및 빨간색 화살표
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        # 좌측 2층 레이어 2개 연결 및 빨간색 화살표
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        # 좌측 3층 레이어 2개 연결 및 빨간색 화살표
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        # 좌측 4층 레이어 2개 연결 및 빨간색 화살표
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        # 좌측 5층 레이어
        enc5_1 = self.enc5_1(pool4)

        # 우측 5층 레이어 및 초록색 화살표
        dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1)

        # 하얀색 부분 연결하기
        cat4 = torch.cat((unpool4, enc4_2),dim =1)

        # 파란색 화살표 실행
        # cat에서 512 + 512 로 1024의 레이어 만들고 파란색 화살표 수행후 아웃풋값을 512로 만듬
        dec4_2 = self.dec4_2(cat4)

        # 여기까지 하면 우측 4층 레이어까지 생성
        dec4_1 = self.dec4_1(dec4_2)

        # 반복 3층
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2),dim =1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        # 반복 2층
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2),dim =1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        # 반복 1층
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2),dim =1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
        x = self.sigmoid(x)

        return x

class UNet_small(nn.Module):
    def __init__(self):
        super(UNet_small, self).__init__()

        # 논문의 파란색 화살표
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            # Conv2d layer 정의
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        # 좌측 레이어 enc (인코더)
        # 1층 좌측 첫번째 레이어 두개 
        self.enc1_1 = CBR2d(in_channels=3, out_channels =64, kernel_size =3, stride=1, padding =1, bias =True)
        self.enc1_2 = CBR2d(in_channels=64, out_channels =64, kernel_size =3, stride=1, padding =1, bias =True)

        # 다음 빨간색 화살표 max_pool 2*2
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 2층 파란색 화살표     
        self.enc2_1 = CBR2d(in_channels=64, out_channels =128, kernel_size =3, stride=1, padding =1, bias =True)
        self.enc2_2 = CBR2d(in_channels=128, out_channels =128, kernel_size =3, stride=1, padding =1, bias =True)
        
        # 다음 빨간색 화살표 max_pool 2*2
        self.pool2 = nn.MaxPool2d(kernel_size =2)

        # 5층 파란색 화살표
        self.enc5_1 = CBR2d(in_channels=128, out_channels =256, kernel_size =3, stride=1, padding =1, bias =True)


        # Expansive path
        
        # 5층 파란색 2번쨰 화살표인데 디코더로
        self.dec5_1 = CBR2d(in_channels=256, out_channels =128, kernel_size =3, stride=1, padding =1, bias =True)

        # 초록색 화살표
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        # 2층 파란색 화살표
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
        
        # 초록색 화살표
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        # 1층 파란색 화살표
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # segmentation에 필요한 n개의 클래스에 대한 output 정의
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # 좌측 1층 레이어 2개 연결 및 빨간색 화살표
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        # 좌측 2층 레이어 2개 연결 및 빨간색 화살표
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        # 좌측 5층 레이어
        enc5_1 = self.enc5_1(pool2)

        # 우측 5층 레이어 및 초록색 화살표
        dec5_1 = self.dec5_1(enc5_1)

        # 반복 2층
        unpool2 = self.unpool2(dec5_1)
        cat2 = torch.cat((unpool2, enc2_2),dim =1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        # 반복 1층
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2),dim =1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        output = self.fc(dec1_1)
        output = self.sigmoid(output)

        return output

class DrivableNet(nn.Module):
    def __init__(self, depth, num_patch, device):
        super().__init__()
        self.depth = depth
        self.num_patch = num_patch
        
        self.img_model = DINOv2_ImgEncoder().to(device=device)
        
        if self.depth:
            self.depth_model = DepthEncoder(in_channels=1, output_channels=256, num_patch=num_patch).to(device=device)
        
        num_channels = 1792 if self.depth else 1536
        
        self.seg_head = DINOv2_SegHead(in_channels=num_channels, output_channels=1, num_patch=num_patch, img_shape=(num_patch*14, num_patch*14)).to(device=device)
        
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)
        
    def forward(self, img, depth):
        x = self.img_model(img)
        concat = x
        
        if self.depth:
            y = self.depth_model(depth)
            y = self.flatten(y)
            y = y.permute(0,2,1)
            
            concat = torch.concat((x, y), dim=-1)
        
        concat = concat.permute(0, 2, 1)
        concat = concat.view(img.shape[0], -1, self.num_patch, self.num_patch)
        
        out = self.seg_head(concat)
        
        if img.shape[-2:]!=out.shape[-2:]:
            raise Exception('Input and Output shapes do not match.')
        
        return out
    
class LeNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(LeNetSegmentation, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=3, padding=1, stride=1)
        self.deconv1 = nn.ConvTranspose2d(120, 16, kernel_size=3, padding=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(16, 6, kernel_size=3, padding=1, stride=1)
        self.deconv3 = nn.ConvTranspose2d(6, num_classes, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        #Encoder
        x = F.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2)
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        x = F.tanh(self.conv3(x))
        x = F.avg_pool2d(x, 2)
        #Decoder
        x = F.tanh(self.deconv1(x)) # Upsampling step (64x64)
        x = F.tanh(self.deconv2(x)) # Upsampling step (128x128)
        x = self.deconv3(x) # Final output (512x512)
        x = self.sigmoid(x)
        
        return x
    
class InitialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class DilatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class EfficientSpatialPyramidBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EfficientSpatialPyramidBlock, self).__init__()
        # Dilated Convolution Layer 1 (Dilation rate = 1)
        self.dilated_conv1 = DilatedConvLayer(in_channels, out_channels, dilation=1)
        # Dilated Convolution Layer 3 (Dilation rate = 4)
        self.dilated_conv3 = DilatedConvLayer(out_channels, out_channels, dilation=4)

    def forward(self, x):
        x = self.dilated_conv1(x)
        x = self.dilated_conv3(x)
        return x


class DAS_ESPNet(nn.Module):
    def __init__(self, num_classes=1):
        super(DAS_ESPNet, self).__init__()
        # Initial convolution layer
        self.initial_conv = InitialConvLayer(in_channels=3, out_channels=64)
        # Efficient Spatial Pyramid Block
        self.esp_block = EfficientSpatialPyramidBlock(in_channels=64, out_channels=64)
        # Final convolutional layer (1x1)
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_conv(x)
        concat1 = x
        x = self.esp_block(x)
        concat2 = x
        concat = torch.cat((concat1, concat2), 1)
        output = self.final_conv(concat)
        output = self.sigmoid(output)
        return output
################################################################################################################
"""
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
        # n = int(nOut/5)
        # n1 = nOut - 4*n
        if nOut < 5:
            n = 4
            n1 = 4
        else:
            n = int(nOut / 5)
            n1 = nOut - 4 * n
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
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, classes=20, p=5, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 +3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64 , 64))
        self.b2 = BR(128 + 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128 , 128))
        self.b3 = BR(256)

        self.classifier = C(256, classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))

        classifier = self.classifier(output2_cat)

        return classifier
        
class ESPNet(nn.Module):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, classes=1, p=6, q=10, encoderFile=None):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        '''
        super().__init__()
        self.encoder = ESPNet_Encoder(classes, p, q)
        if encoderFile != None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            print('Encoder loaded!')
        # load the encoder modules
        self.modules = []
        for i, m in enumerate(self.encoder.children()):
            self.modules.append(m)

        # light-weight decoder
        self.level3_C = C(128 + 3, classes, 1, 1)
        self.br = nn.BatchNorm2d(classes, eps=1e-03)
        self.conv = CBR(19 + classes, classes, 3, 1)

        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BR(2*classes), DilatedParllelResidualBlockB(2*classes , 20*classes, add=False))

        self.up_l2 = nn.Sequential(nn.ConvTranspose2d(20*classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False), BR(classes))

        self.classifier = nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        output0 = self.modules[0](input)
        inp1 = self.modules[1](input)
        inp2 = self.modules[2](input)

        output0_cat = self.modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.modules[4](output0_cat)  # down-sampled

        for i, layer in enumerate(self.modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.modules[6](torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.modules[7](output1_cat)  # down-sampled
        for i, layer in enumerate(self.modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.modules[9](torch.cat([output2_0, output2], 1)) # concatenate for feature map width expansion

        output2_c = self.up_l3(self.br(self.modules[10](output2_cat))) #RUM

        output1_C = self.level3_C(output1_cat) # project to C-dimensional space
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C, output2_c], 1))) #RUM

        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))

        classifier = self.classifier(concat_features)
        # classifier = self.sigmoid(classifier)
        return classifier
"""