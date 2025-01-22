import ffmpeg
import cv2
import numpy as np

def decode_and_process_video(input_video, process_frame_callback):
    """
    使用 ffmpeg 解码视频流并用 OpenCV 处理每一帧。
    
    Args:
        input_video (str): 输入视频文件路径或流地址。
        process_frame_callback (function): 处理帧的回调函数，接受一个帧作为输入。
    """
    # 使用 ffmpeg-python 设置管道
    process = (
        ffmpeg
        .input(input_video)  # 输入视频文件或流
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')  # 原始视频流
        .run_async(pipe_stdout=True)  # 异步运行，输出到 stdout
    )

    # 获取视频帧的宽高
    probe = ffmpeg.probe(input_video)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    # 逐帧读取
    while True:
        # 每帧的字节数：宽 x 高 x 每像素通道数
        frame_size = width * height * 3
        in_bytes = process.stdout.read(frame_size)

        if not in_bytes:
            break  # 视频结束
        
        # 将字节流转换为 numpy 数组
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

        # 用 OpenCV 或其他工具处理帧
        process_frame_callback(frame)

    # 释放资源
    process.wait()


# 定义帧处理回调函数
def process_frame(frame):
    # 在这里用 OpenCV 处理帧，比如显示或保存
    cv2.imshow('Frame', frame)

    # 按键退出逻辑
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)

# 使用函数解码并处理视频
input_video_path = '/home/liuyang/Downloads/yanshou_laixi/1122/1122_0650/20241122065019296-22-1-main.mp4'  # 视频文件路径
decode_and_process_video(input_video_path, process_frame)
