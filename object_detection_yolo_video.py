import os
import time
import cv2
import numpy as np
from yolo.model.yolo_model import YOLO

os.chdir('yolo')

def process_image( img ) :
    """ 이미지 리사이즈, 차원 확장 
    img : 원본 이미지
    결과는 (64, 64, 3) 으로 프로세싱된 이미지 반환 """

    image_org = cv2.resize(img, (416, 416), interpolation = cv2.INTER_CUBIC)
    image_org = np.array(image_org, dtype='float32')
    image_org = image_org / 255.0
    image_org = np.expand_dims(image_org, axis=0)

    return image_org  

def get_classes(file) :
    """ 클래스의 이름을 가져온다. """

    with open(file) as f :
        name_of_class = f.readlines()

    name_of_class = [ class_name.strip() for class_name in name_of_class ]

    return name_of_class

def box_draw(image, boxes, scores, classes, all_classes):
    
    """ image : 오리지날 이미지
    boxes : 오브젝트의 박스데이터, ndarray
    scroes : 오브젝트의 확률, ndarray
    classes : 오브젝트의 클래스정보, ndarray
    all_classes : 모든 클래스 이름 """

    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()

def detect_image( image, yolo, all_classes) :
    """ image : 오리지날 이미지
    yolo : 욜로 모델
    all_classes : 전체 클래스 이름
    변환된 이미지 리턴 ! """
    pimage = process_image(image)

    image_boxes, image_classes, image_scores = yolo.predict(pimage, image.shape)

    if image_boxes is not None :
        box_draw(image, image_boxes, image_scores, image_classes, all_classes)
  
    return image

yolo = YOLO(0.6, 0.5)

all_classes = get_classes('data/coco_classes.txt')

cap = cv2.VideoCapture('videos/test/library1.mp4')

if cap.isOpened() == False :
    print("Error opening video stream or file")

else :
    # 반복문이 필요한 이유?  비데오는 여러 사진으로 구성되어있으니까
    # 여러개니까!
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # 사이즈를 반으로 줄이는 방법
    if int(frame_width/2) % 2 == 0 :
        frame_width = int(frame_width/2)
    else :
        frame_width = int(frame_width/2) +1
    if int(frame_height/2) % 2 == 0 :
        frame_height = int(frame_height/2)
    else :
        frame_height = int(frame_height/2) +1

    out = cv2.VideoWriter('../data/videos/video_ret2.mp4',
                    cv2.VideoWriter_fourcc(*'H264'),
                    10,
                    (frame_width, frame_height))
    while cap.isOpened() :
        # 사진을 1장씩 가져와서.
        ret, frame = cap.read()
        # 제대로 사진 가져왔으면, 화면에 표시!
        if ret == True:
            ### 이 부분을 모델 추론하고 화면에 보여주는 코드로 변경
            # cv2.imshow("Frame", frame)
            result_image = detect_image(frame, yolo, all_classes)
            cv2.imshow('result',frame)
            out.write(frame)
            # 키보드에서 esc키를 누르면 exit 하라는 것.
            if cv2.waitKey(25) & 0xFF == 27 :
                break
        else : 
            break

    cap.release()
    cv2.destroyAllWindows()