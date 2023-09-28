import paddle
from PIL import Image
import numpy as np
import paddle.vision.transforms as T
from paddle.vision.transforms import functional as F
class ONNXModel(paddle.nn.Layer):
    def __init__(self):
        super(ONNXModel, self).__init__()
        self.conv0 = paddle.nn.Conv2D(in_channels=3, out_channels=3, kernel_size=[3, 3], padding=1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=3, kernel_size=[3, 3], padding=1)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(in_channels=3, out_channels=3, kernel_size=[3, 3], padding=1)
        self.sigmoid0 = paddle.nn.Sigmoid()

    def forward(self, x2paddle_input):
        x2paddle__enhance_Constant_output_0 = paddle.full(dtype='float32', shape=[1], fill_value=9.999999747378752e-05)
        x2paddle__enhance_Constant_1_output_0 = paddle.full(dtype='float32', shape=[1], fill_value=1.0)
        x2paddle__Constant_output_0 = paddle.full(dtype='float32', shape=[1], fill_value=0.0)
        x2paddle__Constant_1_output_0 = paddle.full(dtype='float32', shape=[1], fill_value=1.0)
        x2paddle__enhance_in_conv_in_conv_0_Conv_output_0 = self.conv0(x2paddle_input)
        x2paddle__enhance_in_conv_in_conv_1_Relu_output_0 = self.relu0(x2paddle__enhance_in_conv_in_conv_0_Conv_output_0)
        x2paddle__enhance_conv_conv_0_Conv_output_0 = self.conv1(x2paddle__enhance_in_conv_in_conv_1_Relu_output_0)
        x2paddle__enhance_conv_conv_2_Relu_output_0 = self.relu1(x2paddle__enhance_conv_conv_0_Conv_output_0)
        x2paddle__enhance_Add_output_0 = paddle.add(x=x2paddle__enhance_in_conv_in_conv_1_Relu_output_0, y=x2paddle__enhance_conv_conv_2_Relu_output_0)
        x2paddle__enhance_out_conv_out_conv_0_Conv_output_0 = self.conv2(x2paddle__enhance_Add_output_0)
        x2paddle__enhance_out_conv_out_conv_1_Sigmoid_output_0 = self.sigmoid0(x2paddle__enhance_out_conv_out_conv_0_Conv_output_0)
        x2paddle__enhance_Add_1_output_0 = paddle.add(x=x2paddle__enhance_out_conv_out_conv_1_Sigmoid_output_0, y=x2paddle_input)
        x2paddle_illumination = paddle.clip(x=x2paddle__enhance_Add_1_output_0, min=x2paddle__enhance_Constant_output_0, max=x2paddle__enhance_Constant_1_output_0)
        x2paddle__Div_output_0 = paddle.divide(x=x2paddle_input, y=x2paddle_illumination)
        x2paddle_reflection = paddle.clip(x=x2paddle__Div_output_0, min=x2paddle__Constant_output_0, max=x2paddle__Constant_1_output_0)
        return x2paddle_illumination

def inference(x2paddle_input):
    # There are 1 inputs.
    # x2paddle_input: shape-[-1, 3, 600, 400], type-float32.
    im = Image.open(x2paddle_input)
    size = im.size
    paddle.disable_static()
    params = paddle.load(r'/home/aistudio/work/LLIE/model.pdparams')
    model = ONNXModel()
    model.set_dict(params, use_structured_name=True)
    model.eval()
    # resized_image = F.resize(im,(600,400))
    image = T.to_tensor(im)
    tensor  = paddle.unsqueeze(image, axis=0)
    tensor = model(tensor)
    cpu = paddle.CPUPlace()
    image_numpy=paddle.to_tensor(tensor, dtype='float32', place=cpu)
    #image_numpy = tensor[0].cpu().float().numpy()
    image_numpy=paddle.squeeze(image_numpy)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    return im

# def inference_v(road):
#     input_dir_name='/home/aistudio/work/PPYOLOE/demo_input'
#     output_dir_name='/home/aistudio/work/LLIE/image_output'
#     video_output = '/home/aistudio/work/LLIE/video'
#     BATCH_SIZE=10
#     Dataset = MemoryFriendlyLoader(img_dir=input_dir_name)
#     dataloader = DataLoader(Dataset,
#                     batch_size=BATCH_SIZE,
#                     shuffle=False,
#                     drop_last=False,
#                     num_workers=1)
#     for _, (input, image_name) in enumerate(dataloader):
#         illu_list, ref_list= inference(input)
#         for i in range(len(image_name)):
#             imname = image_name[i]
#             u_path = output_dir_name + '/' + imname
#             save_images(ref_list[i], u_path)
#     img_info_array = road.split("/")
#     imname = img_info_array[-1]
#     video_output = i_path + '/' + imname
#     image2video(demo_output,video_output,24)
#     return video_output
