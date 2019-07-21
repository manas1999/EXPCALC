import numpy as np
import cv2 
import random
import sys
a = ""
b=0;
dig_count = 0;
drawing = False
drawing = False
cursor = -100;
sys.path.insert(0,"C:\\Users\\manas madine\\Desktop")
from dnn_softmax import *
import pickle
file="C:\\Users\\manas madine\\Documents\\Downloads\\parameters.pkl"
inpf=open(file,"rb")
parameters=pickle.load(inpf)
inpf.close()
num = random.randint(0,9)
font1 = cv2.FONT_HERSHEY_SIMPLEX
drawing = False
window_name ='CHILD MODE'
cv2.namedWindow(window_name)
img = np.zeros((1012,1012,3),np.uint8)
img1 = np.zeros((1012,1012,3),np.uint8)
img3 = np.zeros((1012,1012,3),np.uint8)
img3 = img3+255
cv2.rectangle(img,(0,720),(1012,1012),(255,255,255),-1)
cv2.rectangle(img,(0,0),(1012,110),(0,128,255),-1)




def draw(event,x,y,flags,param):
    global drawing
    global font1
    global num
    global cursor
    global a
    global dig_count
    global b
    cv2.rectangle(img,(0,110),(253,200),(0,255,255),-1)
    cv2.rectangle(img,(0,0),(1012,110),(0,128,255),-1)
    cv2.rectangle(img,(759,110),(1012,200),(0,255,255),-1)
    cv2.putText(img,'WRITE '+str(num),(290,90),font1,2,(0,0,0),2,cv2.LINE_AA)
    if event == cv2.EVENT_FLAG_RBUTTON:
        dig_count= dig_count+1
        #img[100:690,0:1012,...]=0
        img10 = img[250:650,0:1012]
        grayimg = cv2.cvtColor(img10,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grayimg,(5,5),0)
        ret,thresh = cv2.threshold(blur,25,200,0)
        img2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
        cv2.drawContours(img3,contours,-1,(0,0,0),15)
        max1=0
        min1 = 100000000000
        max2=0;
        min2=1000000000000
        for i in contours :
            for j in i :
                for k in j :
                    if(k[0] < min1) : min1 = k[0];
                    if(k[0]> max1 ): max1 = k[0];
                    if(k[1] < min2) : min2 = k[1];
                    if(k[1]> max2 ): max2 = k[1];
        img2 = img3[min2:max2 , min1:max1]
        if img2.shape[1]>img2.shape[0]:
            pad=(img2.shape[1]-img2.shape[0])//2
            img2=np.pad(img2,((pad,pad),(0,0),(0,0)),'constant',constant_values=(255,))
        else:
            pad=(img2.shape[0]-img2.shape[1])//2
            img2=np.pad(img2,((0,0),(pad,pad),(0,0)),'constant',constant_values=(255,))
        img4 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        img4 = cv2.GaussianBlur(img4,(5,5),0)
        img4 = cv2.resize(img4,(28,28))
        img4 = np.pad(img4,((12,12),),'constant',constant_values=(255,))
        img4 = cv2.resize(img4,(28,28))
        img4=255-img4
        x = img4.reshape(784,1).astype('float32') / 255
        AL,_=L_layer_forward(x,parameters)
        b = b*10+np.argmax(AL)
        drawing = False
        img3[img3!=255]=255
        
        
        
        
    if(x>0 and x < 253 and y > 100 and y < 200 ):
        cv2.rectangle(img,(0,110),(253,200),(0,0,255),-1)
        if event == cv2.EVENT_LBUTTONDOWN:
             img10 = img[250:650,0:1012]
             grayimg = cv2.cvtColor(img10,cv2.COLOR_BGR2GRAY)
             blur = cv2.GaussianBlur(grayimg,(5,5),0)
             ret,thresh = cv2.threshold(blur,25,200,0)
             img2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
             cv2.drawContours(img3,contours,-1,(0,0,0),15)
             max1=0
             min1 = 100000000000
             max2=0;
             min2=1000000000000
             for i in contours :
                 for j in i :
                     for k in j :
                         if(k[0] < min1) : min1 = k[0];
                         if(k[0]> max1 ): max1 = k[0];
                         if(k[1] < min2) : min2 = k[1];
                         if(k[1]> max2 ): max2 = k[1];
             img2 = img3[min2:max2 , min1:max1]
             if img2.shape[1]>img2.shape[0]:
                 pad=(img2.shape[1]-img2.shape[0])//2
                 img2=np.pad(img2,((pad,pad),(0,0),(0,0)),'constant',constant_values=(255,))
             else:
                 pad=(img2.shape[0]-img2.shape[1])//2
                 img2=np.pad(img2,((0,0),(pad,pad),(0,0)),'constant',constant_values=(255,))
             img4 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
             img4 = cv2.GaussianBlur(img4,(5,5),0)
             img4 = cv2.resize(img4,(28,28))
             img4 = np.pad(img4,((12,12),),'constant',constant_values=(255,))
             img4 = cv2.resize(img4,(28,28))
             img4=255-img4
             x = img4.reshape(784,1).astype('float32') / 255
             AL,_=L_layer_forward(x,parameters)
             if b*10+np.argmax(AL) == num : 
                 cv2.putText(img,str(b*10+np.argmax(AL))+"     CORRECT",(cursor+100,820),font1,4,(0,255,0),10,cv2.LINE_AA)
             else :
                cv2.putText(img,str(b*10+np.argmax(AL))+"       WRONG",(cursor+100,820),font1,4,(0,0,255),10,cv2.LINE_AA)
             if dig_count >2: 
                 cursor = cursor +50*(dig_count+1)
             else :
                 cursor = cursor +75*(dig_count+1)
             a = a+str(b*10+np.argmax(AL))
             drawing = False
             img[200:690,0:1012,...]=0
             img3[img3!=255]=255
             b=0
    cv2.putText(img,'SUBMIT',(10,175),font1,2,(0,165,255),2,cv2.LINE_AA)
    cv2.putText(img,'NEW',(790,175),font1,2,(0,165,255),2,cv2.LINE_AA)
    if(x>759 and x < 1012 and y > 100 and y < 200 ):
        cv2.rectangle(img,(759,110),(1012,200),(0,0,255),-1)
        if event == cv2.EVENT_LBUTTONDOWN:
            cursor = -100
            cv2.rectangle(img,(0,700),(1012,1012),(255,255,255),-1)
            img[200:690,0:1012,...]=0
            img3[img3!=255]=255
            num = random.randint(0,9)
    if event == cv2.EVENT_LBUTTONDOWN :
        drawing=True;
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing == True :
        cv2.circle(img,(x,y),8,(0,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.setMouseCallback(window_name,draw)

def main():
    global contours 
    global font1
    global num
    while(True):
        cv2.imshow(window_name,img)
        if cv2.waitKey(20)==27:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

        
        
        
