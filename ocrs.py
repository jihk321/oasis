import cv2
from easyocr import Reader
import os 
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import imutils
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc 
import time
from datetime import datetime


def calc_age(birth,month): # 만나이 계산
    now_year = datetime.now().strftime('%Y')
    now_month = datetime.now().strftime('%m')

    age = int(now_year) - int(birth)
    
    if int(month) > int(now_month) : age = age -1  
    return age

def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    #한글 출력하기 위해 폰트 셋팅 
    font_path = r'C:\Windows\Fonts\gulim.ttc'
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family = font)

    plt.figure(figsize=figsize)
    p_row = len(img)//2 + len(img)%2

    if type(img) == list:
        if type(title) == list:  titles = title
        else:  
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            # plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.subplot(p_row,2, i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

def Text_Dict(img, dic, color=(255,255,255), font_size = 18):
    font = ImageFont.truetype('C:\Windows\Fonts\gulim.ttc', font_size)
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    
    count = 1
    h,w,_ = img.shape

    if len(dic) < 1 : 
        draw.text((10,10), '주민등록번호 감지 안됨', font=font,fill=color)
        return np.array(image)
    yy = h/len(dic)

    
    for k, v in dic.items():
        if type(v) is list : continue
        text = k + ' : ' + str(v)
        draw.text((10,yy*count),text=text,font=font,fill=color)
        count = count + 1
    return np.array(image)

def putText(cv_img, text, x, y, color=(0, 0, 0), font_size=22):

  font = ImageFont.truetype('C:\Windows\Fonts\gulim.ttc', font_size)
  img = Image.fromarray(cv_img)
   
  draw = ImageDraw.Draw(img)
  draw.text((x, y), text, font=font, fill=color)
 
  cv_img = np.array(img)
  
  return cv_img

def text_ocr(img):
    start = time.time()
    reader = Reader(lang_list=['ko', 'en'], gpu=True)
    result = reader.readtext(img, detail=1)
    print(f'문자인식 처리 시간 :{time.time()-start:.2f}s')

    registration = {}

    for index,txt in enumerate(result) :
        years = 0
        if len(txt[1]) == 14 :
            # print(txt[1])
            registration['주민등록번호'] = txt[1]
            registration['num_xy'] = txt[0]

            last_num = txt[1].split('-') # 주민번호 뒷자리
            if int(last_num[1][:1]) %2 == 1 : registration['성별'] = '남자'
            else : registration['성별'] = '여자'

            if int(last_num[1][:1]) == 1 or int(last_num[1][:1]) == 2: years = '19' + str(last_num[0][:2])
            else : years = '20' + str(last_num[0][:2])

            age = calc_age(years,last_num[0][2:4])
            registration['만나이'] = age

    # if not registration : print('주민등록번호 감지 안됨')     
    # else : print(registration)
    return registration

def make_scan_image(image, width, ksize=(5,5), min_threshold=75, max_threshold=200):

    img_time = time.time()
    image = imutils.resize(image, width=width)
    ratio = org_image.shape[1] / float(image.shape[1])

    # 이미지를 grayscale로 변환하고 blur를 적용 모서리를 찾기위한 이미지 연산
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, ksize, 0)
    edged = cv2.Canny(blurred, min_threshold, max_threshold)

    # contours를 찾아 크기순으로 정렬
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = None

    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
        if len(approx) == 4:
            findCnt = approx
            break

    # 만약 추출한 윤곽이 없을 경우 오류
    if findCnt is None:  raise Exception(("경계선을 찾을 수 없음"))

    # print(f'findCnt : {len(findCnt)} {type(findCnt)}')
    output = image.copy()
    cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)

    transform_image = four_point_transform(org_image, findCnt.reshape(4,2)* ratio)
    print(f'이미지 처리 시간: {time.time()- img_time:.2f}s')

    text = text_ocr(transform_image)
    text_img = np.zeros((300,300,3), dtype=np.uint8)
    text_img = Text_Dict(text_img,text)

    if text != {}:
        mozaic_img = transform_image.copy() # 모자이크 시킬 원본 이미지 
        mozaic_rate = 0.1 # 모자이크 사용할 비율 
        
        x = text['num_xy'][0][0]
        y = text['num_xy'][0][1]
        w = text['num_xy'][2][0] - x
        h = text['num_xy'][2][1] - text['num_xy'][1][1]

        mo_roi = mozaic_img[y:y+h,x+(w//2):x+w] # 모자이크 할 영역 주민번호 뒷자리
        m_y, m_x, _ = mo_roi.shape 
        mo_roi = cv2.resize(mo_roi, dsize=(0,0),fx=mozaic_rate,fy=mozaic_rate,interpolation=cv2.INTER_NEAREST) #축소
        mo_roi = cv2.resize(mo_roi, (m_x,m_y), interpolation=cv2.INTER_NEAREST) #확대
        
        mozaic_img[y:y+h,x+(w//2):x+w] = mo_roi

        img_list = ['원본이미지','신분증 영역','문자인식','모자이크']
        plt_imshow(img_list, [org_image,transform_image,text_img,mozaic_img])
    else : 
        img_list = ['원본이미지','문자인식']
        plt_imshow(img_list, [org_image,text_img])
    return transform_image
    

img_list = [img for img in os.listdir('id') if img.endswith('.jpg')]

for index, image in enumerate(img_list):
    image_array = np.fromfile('id//' + image)
    org_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # text_ocr(org_image)
    test = make_scan_image(org_image, width=200, ksize=(5,5), min_threshold=0, max_threshold=100)

    # print(result)

