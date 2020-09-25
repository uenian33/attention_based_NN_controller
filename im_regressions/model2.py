import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *
from initialize import *
'''
attention after max-pooling
'''


class AttnVGG_after(nn.Module):

    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True, init='xavierUniform'):
        super(AttnVGG_after, self).__init__()
        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size / 32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    def forward(self, x):
        # feed forward
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        l1 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # /2
        l2 = F.max_pool2d(self.conv_block4(l1), kernel_size=2, stride=2, padding=0)  # /4
        l3 = F.max_pool2d(self.conv_block5(l2), kernel_size=2, stride=2, padding=0)  # /8
        x = self.conv_block6(l3)  # /32
        g = self.dense(x)  # batch_sizex512x1x1
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1, g2, g3), dim=1)  # batch_sizexC
            # classification layer
            x = self.classify(g)  # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return [x, c1, c2, c3]


'''
attention direct predict action
'''


class AttnVGG_action(nn.Module):

    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True, init='xavierUniform'):
        super(AttnVGG_after, self).__init__()
        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size / 32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.feature_fc = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
        else:
            self.feature_fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        self.final_activate = nn.Tanh()
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    def forward(self, x):
        # feed forward
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        l1 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # /2
        l2 = F.max_pool2d(self.conv_block4(l1), kernel_size=2, stride=2, padding=0)  # /4
        l3 = F.max_pool2d(self.conv_block5(l2), kernel_size=2, stride=2, padding=0)  # /8
        x = self.conv_block6(l3)  # /32
        g = self.dense(x)  # batch_sizex512x1x1
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1, g2, g3), dim=1)  # batch_sizexC
            # classification layer
            x = self.feature_fc(g)  # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.feature_fc(torch.squeeze(g))

        x = self.final_activate(x)
        return [x, c1, c2, c3]

'''
attention for sensor mask generation
'''


class AttnVGG_sensor_mask(nn.Module):

    def __init__(self, im_size, sens_dim, attention=True, normalize_attn=True, init='xavierUniform'):
        super(AttnVGG_sensor_mask, self).__init__()
        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(1, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size / 32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            #self.feature_fc = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
            self.sensor_mask_layer = nn.Sequential(
                nn.Linear(512 * 3, 128),
                nn.Dropout(p=0.15),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.Dropout(p=0.15),
                nn.ReLU(),
                nn.Linear(64, sens_dim),
                nn.Sigmoid()
            )
        else:
            #self.feature_fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
            self.sensor_mask_layer = nn.Sequential(
                nn.Linear(512, 128),
                nn.Dropout(p=0.15),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.Dropout(p=0.15),
                nn.ReLU(),
                nn.Linear(64, sens_dim),
                nn.Sigmoid()
            )

        self.final_block = SensorMaskBlock(sens_dim)

        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    def forward(self, x, sens_x):
        # feed forward
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        l1 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # /2
        l2 = F.max_pool2d(self.conv_block4(l1), kernel_size=2, stride=2, padding=0)  # /4
        l3 = F.max_pool2d(self.conv_block5(l2), kernel_size=2, stride=2, padding=0)  # /8
        x = self.conv_block6(l3)  # /32
        g = self.dense(x)  # batch_sizex512x1x1
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1, g2, g3), dim=1)  # batch_sizexC
            # classification layer
            mask = self.sensor_mask_layer(g)  # learning sensor mask
        else:
            c1, c2, c3 = None, None, None
            mask = self.sensor_mask_layer(torch.squeeze(g))  # learning sensor mask

        # print(mask)
        x = self.final_block(mask, sens_x)
        return [x, c1, c2, c3, mask]

'''
attention for sensor mask generation
'''


class AttnVGG_sensor_dual_mask(nn.Module):

    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True, init='xavierUniform'):
        super(AttnVGG_sensor_dual_mask, self).__init__()
        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size / 32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.feature_fc = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
            self.sensor_mask_feature_layer = nn.Sequential(
                nn.Linear(512 * 3, 128),
                nn.Dropout(p=0.15),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.Dropout(p=0.15),
                nn.ReLU(),
            )
            self.sens_mask_layer = nn.Sequential(
                nn.Linear(64, x_dim),
                nn.Sigmoid()
            )
            self.act_mask_layer = nn.Sequential(
                nn.Linear(64, 3),
                nn.Softmax()
            )
        else:
            self.feature_fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
            self.sensor_mask_feature_layer = nn.Sequential(
                nn.Linear(512, 128),
                nn.Dropout(p=0.15),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.Dropout(p=0.15),
                nn.ReLU(),
            )
            self.sens_mask_layer = nn.Sequential(
                nn.Linear(64, x_dim),
                nn.Sigmoid()
            )
            self.act_mask_layer = nn.Sequential(
                nn.Linear(64, 3),
                nn.Softmax()
            )
            self.sens_mask_layer = nn.Linear(64, x_dim)

        self.final_dual_block = DualMaskBlock(sens_dim)

        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    def forward(self, x, sens_x):
        # feed forward
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        l1 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # /2
        l2 = F.max_pool2d(self.conv_block4(l1), kernel_size=2, stride=2, padding=0)  # /4
        l3 = F.max_pool2d(self.conv_block5(l2), kernel_size=2, stride=2, padding=0)  # /8
        x = self.conv_block6(l3)  # /32
        g = self.dense(x)  # batch_sizex512x1x1
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1, g2, g3), dim=1)  # batch_sizexC
            # classification layer
            x = self.feature_fc(g)  # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.feature_fc(torch.squeeze(g))

        mask_feature = self.sensor_mask_feature_layer(x)  # learning sensor mask

        sens_mask = self.sensor_mask_feature_layer(mask_feature)
        act_mask = self.act_mask_layer(mask_feature)

        x = self.final_dual_block(sens_mask, act_mask, sens_x)
        return [x, c1, c2, c3, sens_mask, act_mask]

'''
branch attention direct predict action
'''


class AttnVGG_action(nn.Module):

    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True, init='xavierUniform'):
        super(AttnVGG_after, self).__init__()
        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size / 32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.feature_fc = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
        else:
            self.feature_fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        self.final_activate = nn.Tanh()
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    def forward(self, x):
        # feed forward
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        l1 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # /2
        l2 = F.max_pool2d(self.conv_block4(l1), kernel_size=2, stride=2, padding=0)  # /4
        l3 = F.max_pool2d(self.conv_block5(l2), kernel_size=2, stride=2, padding=0)  # /8
        x = self.conv_block6(l3)  # /32
        g = self.dense(x)  # batch_sizex512x1x1
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1, g2, g3), dim=1)  # batch_sizexC
            # classification layer
            x = self.feature_fc(g)  # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.feature_fc(torch.squeeze(g))

        x = self.final_activate(x)
        return [x, c1, c2, c3]
