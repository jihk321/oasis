import os 
import json
import cv2

img_path = 'img/test9.jpg'
img_json = 'img/test9.json'

with open(img_json, 'r') as file : 
    img = cv2.imread(img_path)
    
    data = json.load(file)
    face_count = data['info']['faceCount']
    
    face = data['faces']
    for i in range(face_count):
        x,y,w,h = face[i]['roi'].values()
        gender = face[i]['gender']['value']
        gender = 'Man' if gender == 'male' else 'Woman'
        age = face[i]['age']['value']
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(img,gender,(x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        cv2.putText(img,age,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    if img.shape[0] < 800 : img = cv2.resize(img,(0,0),fx=1.5,fy=1.5,interpolation=cv2.INTER_LINEAR)
    cv2.imshow('test',img)
    cv2.imwrite('naver/naver_api9.jpg',img)
    cv2.waitKey(0)
    # print(gender,age)
    # print(face)

