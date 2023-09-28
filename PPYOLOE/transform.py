import cv2
import os
#将视频转换为图片
def video2image(video_path, image_path):
    cap = cv2.VideoCapture(video_path)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(image_path + '/' + str(i) + '.png', frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()

def image2video(image_path, video_path,fps):
    image_names = os.listdir(image_path)
    image_names.sort(key=lambda x:int(x.split('.')[0]))
    frame = cv2.imread(os.path.join(image_path, image_names[0]))
    height, width, channels = frame.shape
    temp = os.path.splitext(video_path)
    if temp[-1] in ['.mp4']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for image_name in image_names:
            video.write(cv2.imread(os.path.join(image_path, image_name)))
        video.release()
        cv2.destroyAllWindows()
    elif temp[-1] in ['.avi']:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for image_name in image_names:
            video.write(cv2.imread(os.path.join(image_path, image_name)))
        video.release()
        cv2.destroyAllWindows()
    else:
        print('error')

