import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self, n_class=1):
        super().__init__()

        self.conv_down1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        self.conv_down2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv_down3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv_down4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv_down5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv_down6 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)

        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(32)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(128)
        self.norm5 = nn.BatchNorm2d(256)
        self.norm6 = nn.BatchNorm2d(512)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        
        self.conv_up6 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv_up5 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.conv_up4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.conv_up3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.conv_up2 = nn.ConvTranspose2d(64,16, kernel_size=4, stride=2, padding=1)
        self.conv_up1 = nn.ConvTranspose2d(32,1, kernel_size=4, stride=2, padding=1)

        self.denorm5 = nn.BatchNorm2d(256)
        self.denorm4 = nn.BatchNorm2d(128)
        self.denorm3 = nn.BatchNorm2d(64)
        self.denorm2 = nn.BatchNorm2d(32)
        self.denorm1 = nn.BatchNorm2d(16)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        conv1 = self.leaky_relu(self.norm1(self.conv_down1(x)))     #[1,16,256,128]
        conv2 = self.leaky_relu(self.norm2(self.conv_down2(conv1))) #[1,32,128,64]
        conv3 = self.leaky_relu(self.norm3(self.conv_down3(conv2))) #[1,64,64,32]
        conv4 = self.leaky_relu(self.norm4(self.conv_down4(conv3))) #[1,128,32,16]
        conv5 = self.leaky_relu(self.norm5(self.conv_down5(conv4))) #[1,256,16,8]
        conv6 = self.leaky_relu(self.norm6(self.conv_down6(conv5))) #[1,512,8,4]

        dec = self.relu(self.dropout(self.denorm5(self.conv_up6(conv6))))
        dec = self.relu(self.dropout(self.denorm4(self.conv_up5(torch.cat([dec, conv5], dim=1)))))
        dec = self.relu(self.dropout(self.denorm3(self.conv_up4(torch.cat([dec, conv4], dim=1)))))
        dec = self.relu(self.denorm2(self.conv_up3(torch.cat([dec, conv3], dim=1))))
        dec = self.relu(self.denorm1(self.conv_up2(torch.cat([dec, conv2], dim=1))))
        dec = self.conv_up1(torch.cat([dec, conv1], dim=1))

        mask = self.sigmoid(dec)
        out = mask * x
        
        return out
