import torch
import torch.nn as nn
import torch.nn.functional as F
from cell import ConvLSTM
import numpy as np

class VideoSaliency(nn.Module):
    def __init__(self):
        super(VideoSaliency, self).__init__()

        ############### R1 ###############
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding_type='SAME')
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding_type='SAME')

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding_type='SAME')
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding_type='SAME')

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding_type='SAME')
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding_type='SAME')
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding_type='SAME')

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding_type='SAME')
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding_type='SAME')
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding_type='SAME')

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding_type='SAME')
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding_type='SAME')
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding_type='SAME')

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=4, dilation=4, padding_type='SAME')
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1, dilation=4, padding_type='SAME')
        self.fc8 = nn.Conv2d(4096, 1, kernel_size=1, padding_type='SAME')

        self.pool4_conv = nn.Conv2d(512, 128, kernel_size=3, padding_type='SAME')
        self.pool4_fc = nn.Conv2d(128, 128, kernel_size=1, padding_type='SAME')
        self.pool4_ms_saliency = nn.Conv2d(128, 1, kernel_size=1, padding_type='SAME')

        ############### R2 ###############
        self.conv1_1_r2 = nn.Conv2d(4, 64, kernel_size=3, padding_type='SAME')
        self.conv1_2_r2 = nn.Conv2d(64, 64, kernel_size=3, padding_type='SAME')

        self.conv2_1_r2 = nn.Conv2d(64, 128, kernel_size=3, padding_type='SAME')
        self.conv2_2_r2 = nn.Conv2d(128, 128, kernel_size=3, padding_type='SAME')

        self.conv3_1_r2 = nn.Conv2d(128, 256, kernel_size=3, padding_type='SAME')
        self.conv3_2_r2 = nn.Conv2d(256, 256, kernel_size=3, padding_type='SAME')
        self.conv3_3_r2 = nn.Conv2d(256, 256, kernel_size=3, padding_type='SAME')

        self.conv4_1_r2 = nn.Conv2d(256, 512, kernel_size=3, padding_type='SAME')
        self.conv4_2_r2 = nn.Conv2d(512, 512, kernel_size=3, padding_type='SAME')
        self.conv4_3_r2 = nn.Conv2d(512, 512, kernel_size=3, padding_type='SAME')

        self.conv5_1_r2 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding_type='SAME')
        self.conv5_2_r2 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding_type='SAME')
        self.conv5_3_r2 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding_type='SAME')

        self.fc6_r2 = nn.Conv2d(512, 4096, kernel_size=4, dilation=4, padding_type='SAME')
        self.fc7_r2 = nn.Conv2d(4096, 4096, kernel_size=1, dilation=4, padding_type='SAME')
        self.fc8_r2 = nn.Conv2d(4096, 1, kernel_size=1, padding_type='SAME')

        self.pool4_conv_r2 = nn.Conv2d(512, 128, kernel_size=3, padding_type='SAME')
        self.pool4_fc_r2 = nn.Conv2d(128, 128, kernel_size=1, padding_type='SAME')
        self.pool4_ms_saliency_r2 = nn.Conv2d(128, 1, kernel_size=1, padding_type='SAME')

        self.convLSTM = ConvLSTM((480, 480), 4, [1], (3, 3), 1, batch_first=True, return_all_layers=False)

        self.pool4_saliency_ST = nn.Conv2d(2, 1, kernel_size=1, padding_type='SAME')
        self.fc8_saliency_ST = nn.Conv2d(2, 1, kernel_size=1, padding_type='SAME')

        self.loc_estimate = nn.Linear(3600, 4)

        self.attention_first = nn.Conv2d(2, 90, kernel_size=3, padding_type='SAME')
        self.attention_second = nn.Conv2d(90, 2, kernel_size=1, padding_type='SAME')

    def forward(self, input, input_prior):

        ############### R1 ###############
        x = F.relu(self.conv1_1(input))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, 1)

        branch_pool4 = x.clone()

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, 1)

        x = F.dropout(F.relu(self.fc6(x)), 0.5)
        x = F.dropout(F.relu(self.fc7(x)), 0.5)
        x = self.fc8(x)

        branch_pool4 = F.dropout(F.relu(self.pool4_conv(branch_pool4)), 0.5)
        branch_pool4 = F.dropout(F.relu(self.pool4_fc(branch_pool4)), 0.5)
        branch_pool4 = self.pool4_ms_saliency(branch_pool4)

        up_fc8 = F.upsample_bilinear(x, size=[480, 480])
        up_pool4 = F.upsample_bilinear(branch_pool4, size=[480, 480])

        ############### R2 ###############
        x_r2 = F.relu(self.conv1_1_r2(input_prior))
        x_r2 = F.relu(self.conv1_2_r2(x_r2))
        x_r2 = F.max_pool2d(x_r2, 2)

        x_r2 = F.relu(self.conv2_1_r2(x_r2))
        x_r2 = F.relu(self.conv2_2_r2(x_r2))
        x_r2 = F.max_pool2d(x_r2, 2)

        x_r2 = F.relu(self.conv3_1_r2(x_r2))
        x_r2 = F.relu(self.conv3_2_r2(x_r2))
        x_r2 = F.relu(self.conv3_3_r2(x_r2))
        x_r2 = F.max_pool2d(x_r2, 2)

        x_r2 = F.relu(self.conv4_1_r2(x_r2))
        x_r2 = F.relu(self.conv4_2_r2(x_r2))
        x_r2 = F.relu(self.conv4_3_r2(x_r2))
        x_r2 = F.max_pool2d(x_r2, 1)

        branch_pool4_r2 = x_r2.clone()

        x_r2 = F.relu(self.conv5_1_r2(x_r2))
        x_r2 = F.relu(self.conv5_2_r2(x_r2))
        x_r2 = F.relu(self.conv5_3_r2(x_r2))
        x_r2 = F.max_pool2d(x_r2, 1)

        x_r2 = F.dropout(F.relu(self.fc6_r2(x_r2)), 0.5)
        x_r2 = F.dropout(F.relu(self.fc7_r2(x_r2)), 0.5)
        x_r2 = self.fc8_r2(x_r2)

        branch_pool4_r2 = F.dropout(F.relu(self.pool4_conv_r2(branch_pool4_r2)), 0.5)
        branch_pool4_r2 = F.dropout(F.relu(self.pool4_fc_r2(branch_pool4_r2)), 0.5)
        branch_pool4_r2 = self.pool4_ms_saliency_r2(branch_pool4_r2)

        up_fc8_r2 = F.upsample_bilinear(x_r2, size=[480, 480])
        up_pool4_r2 = F.upsample_bilinear(branch_pool4_r2, size=[480, 480])

        rnn_inputs = torch.cat((up_pool4, up_pool4_r2, up_fc8, up_fc8_r2), 1)
        rnn_inputs = rnn_inputs.unsqueeze(0)
        rnn_list, state = self.convLSTM(rnn_inputs)
        rnn_output = rnn_list[0].squeeze(0)

        pool4_saliency_cancat = torch.cat((branch_pool4, branch_pool4_r2), 1)
        pool4_saliency_ST = self.pool4_saliency_ST(pool4_saliency_cancat)
        up_pool4_ST = F.upsample_bilinear(pool4_saliency_ST, size=[480, 480])

        fc8_saliency_cancat = torch.cat((x, x_r2), 1)
        fc8_saliency_ST = self.fc8_saliency_ST(fc8_saliency_cancat)
        up_fc8_ST = F.upsample_bilinear(fc8_saliency_ST, size=[480, 480])

        # fc8_saliency_ST = F.upsample_bilinear(fc8_saliency_ST, size=[480, 480])
        fc8_saliency_ST = fc8_saliency_ST.view(fc8_saliency_ST.size(0), -1)

        local_poc = F.sigmoid(self.loc_estimate(fc8_saliency_ST))
        cap_feats = self.validate_loca_poc(local_poc)
        # rnn_output = F.upsample_bilinear(rnn_output, size=[480, 480])
        global_saliency = up_pool4_ST + up_fc8_ST + rnn_output

        local_saliency = torch.mul(global_saliency, cap_feats)

        final_saliency = torch.cat((global_saliency, local_saliency), 1)

        #channel-wise attention
        atten_weights = F.relu(self.attention_first(final_saliency))
        atten_weights = F.softmax(self.attention_second(atten_weights))

        final_saliency = torch.mul(final_saliency, atten_weights)
        final_saliency = torch.mean(final_saliency, 1, keepdim=True)

        return final_saliency, rnn_output, local_poc

    def validate_loca_poc(self, local_poc):
        points = local_poc.data.cpu().numpy()
        # points_val = np.zeros_like(points, dtype=points.dtype)
        cap_map_batch = np.zeros([points.shape[0], 1, 480, 480], dtype=np.float16)
        for i in range(0, points.shape[0]):
            point = points[i, :]
            if point[0] < point[2] and point[1] < point[3] \
                    and (point[2] - point[0]) > 0.167 \
                    and (point[3] - point[1]) > 0.167:
                # suitable point
                print('suitable point')
                point = point * 480
                point = point.astype(np.int16)
                cap_map = np.ones([point[3] - point[1], point[2] - point[0]], dtype=np.float16)
                cap_map = np.pad(cap_map, ([point[1], 480 - point[3]], [point[0], 480 - point[2]]), 'constant')
                cap_map_batch[i, 0, :, :] = cap_map
            else:
                # not suitable, choose center crop
                cap_map = np.ones([240, 240], dtype=np.float16)
                cap_map = np.pad(cap_map, ([120, 120], [120, 120]), 'constant')
                cap_map_batch[i, 0, :, :] = cap_map

        cap_map_batch = torch.from_numpy(cap_map_batch)
        cap_map_batch = cap_map_batch.type(torch.cuda.FloatTensor)

        return cap_map_batch