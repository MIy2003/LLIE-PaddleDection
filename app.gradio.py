#该应用创建工具共包含三个区域，顶部工具栏，左侧代码区，右侧交互效果区，其中右侧交互效果是通过左侧代码生成的，存在对照关系。
#顶部工具栏：运行、保存、新开浏览器打开、实时预览开关，针对运行和在浏览器打开选项进行重要说明：
#[运行]：交互效果并非实时更新，代码变更后，需点击运行按钮获得最新交互效果。
#[在浏览器打开]：新建页面查看交互效果。
#以下为应用创建工具的示例代码

import gradio as gr
from LLIE.arrange import inference
from PPYOLOE.run_video import predict,predict_v

def image_inference(input_date):
    result1 = inference(input_date)
    result2 = predict(input_date)
    return result1,result2

def video_inference(input_date):
    result = predict_v(input_date)
    return result

def clear_all():
    return None,None
def clear_all2():
    return None

with gr.Blocks() as demo:
    gr.Markdown("PP-Human Pipeline")
    with gr.Tabs():
        with gr.TabItem("image"):
            with gr.Row():
                img_in = gr.Image(label="Input",height=240,type="filepath")
                img_out1 = gr.Image(label="Low-light Enhanced Output",height=240,type="pil")
            with gr.Row():
                img_out2 = gr.Image(label="Target Detection Output",height=240,type="pil")
                img_example = gr.Image(interactive=False,label="Example",height=240)
            img_button1 = gr.Button("Submit")
            img_button2 = gr.Button("Clear")
        with gr.TabItem("video"):
            video_in = gr.Video(label="Input only support .mp4 or .avi",height=360)
            video_out = gr.Video(label="Output",height=360)
            video_button1 = gr.Button("Submit")
            video_button2 = gr.Button("Clear")
    img_button1.click(
        fn=image_inference,
        inputs=img_in,
        outputs=[img_out1,img_out2])
    img_button2.click(
        fn=clear_all,
        inputs=None,
        outputs=[img_out1,img_out2])
    video_button1.click(
        fn=video_inference,
        inputs=video_in,
        outputs=video_out)
    video_button2.click(
        fn=clear_all2,
        inputs=None,
        outputs=video_out)
    
demo.launch()