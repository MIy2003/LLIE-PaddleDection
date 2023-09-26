import subprocess

command = "CUDA_VISIBLE_DEVICES=0 python python/infer.py --model_dir=ppyoloe_crn_l_300e_coco --image_file=demo_input/test1.jpg --device=gpu --run_mode=paddle --output_dir=demo_output"

try:
    subprocess.check_call(command, shell=True)
    print("Execution has completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred during execution: {str(e)}")
