import imageio
import cv2
import numpy as np
import time
from collections import deque
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX
from torchvision import transforms
from PIL import Image
import torch

# Plot values in opencv program
class Plotter:
    def __init__(self, plot_width, plot_height, num_plot_values):
        self.width = plot_width
        self.height = plot_height
        self.color_list = [(255, 255, 255), (0, 0 ,250),(0, 250 ,0),(0, 35 ,35),(35, 0 ,35),(250, 0, 0),(35, 35 , 0)]
        self.color  = []
        self.val = []
        self.plot = np.ones((self.height, self.width, 3))*0
        self.scale = 10

        for i in range(num_plot_values):
            self.color.append(self.color_list[i])
            
    def multiplot(self, val, label = "plot"):
        self.val.append(val)
        while len(self.val) > self.width:
            self.val.pop(0)

        self.show_plot(label)

    def show_plot(self, label):
        c_width = self.width*self.scale
        self.plot = np.ones((self.height, c_width, 3))*0
        
        self.plot[:,int(c_width*0.795),:] = 255
        self.plot[:,int(c_width)-1,:] = 255
        self.plot[0:4,int(c_width*0.795):,:] = 255
        self.plot[-3:,int(c_width*0.795):,:] = 255
        
        for i in range(len(self.val)-1):
            for j in range(len(self.val[0])):
                cv2.line(self.plot, (i*int(self.scale*0.8), int(self.height) - int(self.val[i][j]*3)), 
                         ((i+1)*int(self.scale*0.8), int(self.height) - int(self.val[i+1][j]*3)), self.color[j], 3, cv2.LINE_AA)

        if len(self.val) > 30:
            exp_score = np.mean(self.val[-30:], axis=0)
            cv2.putText(self.plot, 'NEUTRAL: {}'.format(int(exp_score[0])), (int(c_width*0.8), 30), cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[0], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'HAPPY: {}'.format(int(exp_score[1])), (int(c_width*0.8), 70), cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[1], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'SURPRISE: {}'.format(int(exp_score[2])), (int(c_width*0.8), 110), cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[2], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'SADNESS: {}'.format(int(exp_score[3])), (int(c_width*0.8), 150), cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[3], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'ANGER: {}'.format(int(exp_score[4])), (int(c_width*0.8), 200), cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[4], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'DISGUST: {}'.format(int(exp_score[5])), (int(c_width*0.8), 240), cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[5], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'FEAR: {}'.format(int(exp_score[6])), (int(c_width*0.8), 280), cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[6], 2, cv2.LINE_AA)
            
        resized = cv2.resize(self.plot, (1280,300), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(label, resized)
    
# 모델 읽어오기 및 GPU, CPU 사용여부 설정
def load_trained_model(model_path):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    return model

if __name__ == "__main__":

    # GPU 자원 사용확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 얼굴 특징점 모델(3DDFA)을 위한 변수
    cfg = { 'arch': 'mobilenet', 
            'widen_factor': 1.0, 
            'checkpoint_fp': 'weights/mb1_120x120.pth', 
            'bfm_fp': 'configs/bfm_noneck_v3.pkl', 
            'size': 120, 
            'num_params': 62}
    # 하드웨어 가속화 라이브러리 onnx 사용
    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
    emotion_dict = {0: 'NEUTRAL', 1: 'HAPPY', 2: 'SURPRISE', 3: 'SADNESS', 4: 'ANGER', 5: 'DISGUST', 6: 'FEAR'}
    # 모델 로드
    model = load_trained_model('./models/FER_trained_model.pt')
    model = model.to(device)
    # 이미지 정규화 및 텐서화
    val_transform = transforms.Compose([transforms.ToTensor()])
    # Given a camera
    reader = imageio.get_reader('<video0>')

    # the simple implementation of average smoothing by looking ahead by qn_next frames
    # assert the frames of the video >= n
    n_pre = 1
    n_next = 1
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()
    # run
    first_frame = True
    dense_flag = False
    pre_ver = None

    plot_graph = Plotter(100, 300, 7)
    prevTime = 0

    for i, frame in enumerate(reader):
        frame_bgr = frame[..., ::-1]  # RGB->BGR
        frame_bgr = cv2.resize(frame_bgr, (1280,720), interpolation=cv2.INTER_CUBIC)
        frame_rows, frame_cols, _ = np.shape(frame_bgr)
        #print(frame_rows, frame_cols)
        fn = i+1

        if first_frame:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes(frame_bgr)
            if len(boxes) > 0:
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                # refine
                param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                # padding queue
                for _ in range(n_pre):
                    queue_ver.append(ver.copy())
                queue_ver.append(ver.copy())
                for _ in range(n_pre):
                    queue_frame.append(frame_bgr.copy())
                queue_frame.append(frame_bgr.copy())
                first_frame = False
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')
            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                if len(boxes) < 1:
                    continue
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())
    
        pre_ver = ver  # for tracking
        
        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            img_draw = queue_frame[n_pre]
            gray = cv2.cvtColor(img_draw, cv2.COLOR_BGR2GRAY)
            roi_width = int(roi_box[2]-roi_box[1])
            roi_height = int(roi_box[3]-roi_box[1])
            # apply the mask
            cv2.rectangle(img_draw, (int(roi_box[0] + roi_width/20),int(roi_box[1])), 
                          (int(roi_box[2] - roi_width/20), int(roi_box[3] - roi_height/10)), (255,0,0), 2)
            resize_frame = cv2.resize(gray[int(roi_box[1]):int(roi_box[3] - roi_height/10), 
                                           int(roi_box[0] + roi_width/20):int(roi_box[2] - roi_width/20)], (48, 48))
            
            X = resize_frame/256
            X = Image.fromarray((X))
            X = val_transform(X).unsqueeze(0)
            X = X.to(device)
            
            with torch.no_grad():
                model.eval()
                log_ps = model(X)
                ps = torch.exp(log_ps)
                ps = ps.cpu()
                top_p, top_class = ps.topk(1, dim=1)
                top_p = top_p.cpu()
                top_class = top_class.cpu()
                pred = emotion_dict[int(top_class.numpy())]
                
            outputs = ps.numpy()[0]*100
            outputs = np.array(outputs, dtype=np.int32)
            plot_graph.multiplot(outputs)

            label_size, base_line = cv2.getTextSize("{}: 000".format(pred), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
            cv2.rectangle(
                img_draw,
                (int(roi_box[2] - roi_width/12), int(roi_box[1]) + 1 - label_size[1]),
                (int(roi_box[2] - roi_width/12) + label_size[0], int(roi_box[1]) + 1 + base_line),
                (223, 128, 255),
                cv2.FILLED,
            )
            
            cv2.putText(
                img_draw,
                "{} {}".format(pred, int(top_p.numpy() * 100)),
                (int(roi_box[2] - roi_width/12), int(roi_box[1]) + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

            curTime = time.time()
            sec = curTime - prevTime
            prevTime = curTime
            fps = 1/(sec)
            fps_txt = "FPS : %0.1f" % (fps)
            cv2.putText(img_draw, fps_txt, (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),2)
            cv2.imshow('img', img_draw)
            k = cv2.waitKey(1)

            if k  == ord('q'):
                break

            queue_ver.popleft()
            queue_frame.popleft()

    cv2.destroyAllWindows()
    reader.close()