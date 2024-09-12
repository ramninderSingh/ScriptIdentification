
import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class=3,  # 3 classes for script identification
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)
        
        # print("the channel is",output_channel)
        # print("the output height is",output_height)
        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        # self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        # Output layer for script identification (3 classes)
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 8 == 0
        assert img_width % 4 == 0

        
        # channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]       
        # kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        # strides = [1, 1, 1, 1, 1, 1, 1]
        # paddings = [1, 1, 1, 1, 1, 1, 0]


        channels = [img_channel, 32, 64, 128, 128, 256]
        kernel_sizes = [3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            input_channel = channels[i]
            output_channel = channels[i + 1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # Define CNN backbone structure
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        cnn.add_module('dropout0', nn.Dropout(p=0.3))
        
        conv_relu(2,batch_norm=True)
      
        cnn.add_module('pooling2', nn.MaxPool2d(kernel_size=(2, 1)))
        cnn.add_module('dropout2', nn.Dropout(p=0.3)) 

        conv_relu(3, batch_norm=True)


        conv_relu(4)
        # conv_relu(5, batch_norm=True)
        # cnn.add_module('pooling3', nn.MaxPool2d(kernel_size=(2, 1)))
        # cnn.add_module('dropout3', nn.Dropout(p=0.3))
        # conv_relu(6)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 8 - 1 , img_width // 4
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()
        # print(channel)
        # print(height)

        conv = conv.view(batch, channel * height, width)
        # print(f"View shape: {conv.shape}")
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        # print(f"Permute shape: {conv.shape}")

        seq = self.map_to_seq(conv)
        # print(f"Seq shape: {seq.shape}")
        recurrent, _ = self.rnn1(seq)
       
        # recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)

