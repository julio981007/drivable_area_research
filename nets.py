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

        x = self.fc(dec1_1)
        output = self.sigmoid(x)

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