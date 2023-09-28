import subprocess
from PIL import Image
import os
import shutil
from .transform import video2image,image2video
def predict(road):
    im = Image.open(road)
    path1 = 'work/PPYOLOE/demo_input'
    path2 = 'work/PPYOLOE/demo_output'
    shutil.rmtree(path1)
    os.mkdir(path1)
    img_info_array = road.split("/")
    imname = img_info_array[-1]
    i_path = path1 + '/' + imname
    im.save(i_path)
    shutil.rmtree(path2)
    os.mkdir(path2)
    command = "CUDA_VISIBLE_DEVICES=0 python work/PPYOLOE/python/infer.py --model_dir=work/PPYOLOE/ppyoloe_crn_l_300e_coco --image_dir=work/PPYOLOE/demo_input --device=gpu --run_mode=paddle --output_dir=work/PPYOLOE/demo_output"
    try:
        subprocess.check_call(command, shell=True)
        print("Execution has completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during execution: {str(e)}")

    o_path = path2 + '/' + imname
    im2 = Image.open(o_path)
    return im2

def predict_v(road):
    demo_input = 'work/PPYOLOE/demo_input'
    demo_output = 'work/PPYOLOE/demo_output'
    shutil.rmtree(demo_input)
    os.mkdir(demo_input)
    shutil.rmtree(demo_output)
    os.mkdir(demo_output)
    video2image(road,demo_input)
    # shutil.rmtree(output)
    # os.mkdir(output)
    command = "CUDA_VISIBLE_DEVICES=0 python work/PPYOLOE/python/infer.py --model_dir=work/PPYOLOE/ppyoloe_crn_l_300e_coco --image_dir=work/PPYOLOE/demo_input --device=gpu --run_mode=paddle --output_dir=work/PPYOLOE/demo_output"
    try:
        subprocess.check_call(command, shell=True)
        print("Execution has completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during execution: {str(e)}")

    video_name,suffix = os.path.splitext(road)
    video_output = video_name + "_output"+suffix
    image2video(demo_output,video_output,24)
    return video_output


# road = 'work/PPYOLOE/video_input/test2.mp4'
# road2 = predict_v(road)
# print(road2)