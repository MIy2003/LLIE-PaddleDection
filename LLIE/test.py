import paddle
from paddle.vision.transforms import functional as F
from PIL import Image

# 加载图像
def inference(image):
    # 构建旋转模型
    # model = paddle.Model()
    # model.add_sublayer('rotate', paddle.nn.Conv2D(1, 1, 3, 1, padding=1))  # 示例模型，可根据需要自定义
    # model.prepare()
    # model.load_state_dict(paddle.load('path_to_model_weights'))  # 加载预训练模型权重
    # 数据预处理
    resized_image = F.resize(image,(600,400))
    chwtensor = paddle.to_tensor(resized_image, dtype='float32', data_format='CHW')
    tensor = paddle.unsqueeze(chwtensor, axis=0)
    # chwtensor = paddle.to_tensor(resized_image, dtype='float32', data_format='CHW')
    # 进行旋转
    # predicted = model.predict(image)
    # 显示旋转后的图像
    # transformed_image = image
    # transformed_image = transformed_image.squeeze().numpy()  # 转换为NumPy数组
    # im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    return image