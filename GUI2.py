import numpy as np
import cv2 
import sys
sys.path.insert(0,"C:\\Users\\manas madine\\Desktop")
from dnn_softmax import *
import pickle
font1 = cv2.FONT_HERSHEY_SIMPLEX
file="C:\\Users\\manas madine\\Documents\\Downloads\\parameters.pkl"
inpf=open(file,"rb")
parameters=pickle.load(inpf)
inpf.close()
a = ""
b=0;
dig_count = 0;
drawing = False
window_name ='MANAS'
cv2.namedWindow(window_name)
img = np.zeros((1012,1012,3),np.uint8)
img3 = np.zeros((1012,1012,3),np.uint8)
img3 = img3+255
cv2.rectangle(img,(0,720),(1012,1012),(255,255,255),-1)
a=""
cursor = -100;
def draw(event,x,y,flags,param):
    global drawing
    global cursor
    global font1
    global a
    global dig_count
    global b
    cv2.rectangle(img,(0,0),(1012,100),(0,255,255),-1)
    cv2.rectangle(img,(0,100),(253,200),(0,255,255),-1)
    cv2.rectangle(img,(759,100),(1012,200),(0,255,255),-1)
    cv2.line(img,(0,100),(1012,100),(0,0,255),3)
    cv2.line(img,(0,200),(253,200),(0,0,255),3)
    cv2.line(img,(759,200),(1012,200),(0,0,255),3)
    cv2.line(img,(0,0),(1012,0),(0,0,255),3)
    cv2.line(img,(253,100),(253,200),(0,0,255),3)
    cv2.line(img,(759,100),(759,200),(0,0,255),3)
    cv2.line(img,(0,0),(0,200),(0,0,255),3)
    cv2.line(img,(1010,0),(1012,200),(0,0,255),3)
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
        #cv2.putText(img,str(np.argmax(AL)),(cursor+100,820),font1,4,(0,0,255),10,cv2.LINE_AA)
        #cursor = cursor +100
        #a = a+str(np.argmax(AL))
        b = b*10+np.argmax(AL)
        drawing = False
        img3[img3!=255]=255
        
    
    if(x>0 and x < 253 and y > 0 and y < 100 ):
        cv2.rectangle(img,(0,0),(253,100),(0,0,255),-1)
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.putText(img,'+',(cursor+100,800),font1,4,(146,21,126),10,cv2.LINE_AA)
            cursor = cursor +100
            a = a+"+"
    if(x>253 and x < 506 and y > 0 and y < 100 ):
        cv2.rectangle(img,(253,0),(506,100),(0,0,255),-1)
        if event == cv2.EVENT_LBUTTONDOWN:
             cv2.putText(img,'-',(cursor+100,800),font1,4,(146,21,126),10,cv2.LINE_AA)
             cursor = cursor +100
             a = a+"-"
    if(x>506 and x < 759 and y > 0 and y < 100 ):
        cv2.rectangle(img,(506,0),(759,100),(0,0,255),-1)
        if event == cv2.EVENT_LBUTTONDOWN:
             cv2.putText(img,'*',(cursor+100,800),font1,4,(146,21,126),10,cv2.LINE_AA)
             cursor = cursor +100
             a = a+"*"
    if(x>759 and x < 1012 and y > 0 and y < 100 ):
        cv2.rectangle(img,(759,0),(1012,100),(0,0,255),-1)
        if event == cv2.EVENT_LBUTTONDOWN:
             cv2.putText(img,'/',(cursor+100,820),font1,4,(146,21,126),10,cv2.LINE_AA)
             cursor = cursor +100
             a = a+"/"
    if(x>759 and x < 1012 and y > 100 and y < 200 ):
        cv2.rectangle(img,(759,100),(1012,200),(0,0,255),-1)
        if event == cv2.EVENT_LBUTTONDOWN:
            cursor = -100
            cv2.rectangle(img,(0,700),(1012,1012),(255,255,255),-1)
            img[200:690,0:1012,...]=0
            img3[img3!=255]=255
            
    if(x>0 and x < 253 and y > 100 and y < 200 ):
        cv2.rectangle(img,(0,100),(253,200),(0,0,255),-1)
        if event == cv2.EVENT_LBUTTONDOWN:
             img10 = img[250:650,0:1012]
             grayimg = cv2.cvtColor(img10,cv2.COLOR_BGR2GRAY)
             blur = cv2.GaussianBlur(grayimg,(5,5),0)
             ret,thresh = cv2.threshold(blur,25,200,0)
             _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
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
             cv2.putText(img,str(b*10+np.argmax(AL)),(cursor+100,820),font1,4,(0,0,255),10,cv2.LINE_AA)
             if dig_count >2: 
                 cursor = cursor +50*(dig_count+1)
             else :
                 cursor = cursor +75*(dig_count+1)
             a = a+str(b*10+np.argmax(AL))
             drawing = False
             img[200:690,0:1012,...]=0
             img3[img3!=255]=255
             b=0
             dig_count=0
    
    
    cv2.line(img,(253,0),(253,100),(0,0,255),3)
    cv2.line(img,(506,0),(506,100),(0,0,255),3)
    cv2.line(img,(759,0),(759,100),(0,0,255),3)
    cv2.putText(img,'CLEAR',(790,175),font1,2,(0,165,255),2,cv2.LINE_AA)
    cv2.putText(img,'SUBMIT',(10,175),font1,2,(0,165,255),2,cv2.LINE_AA)
    cv2.putText(img,'+',(90,90),font1,4,(146,21,126),2,cv2.LINE_AA)
    cv2.putText(img,'-',(333,90),font1,4,(146,21,126),2,cv2.LINE_AA)
    cv2.putText(img,'*',(586,90),font1,4,(146,21,126),2,cv2.LINE_AA)
    cv2.putText(img,'/',(820,50),font1,4,(146,21,126),2,cv2.LINE_AA)

    if event == cv2.EVENT_LBUTTONDOWN :
        drawing=True;
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing == True :
        cv2.circle(img,(x,y),8,(0,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
       
   
cv2.setMouseCallback(window_name,draw)



def main():
    
     global contours 
     global cursor
     global font1
     global a
     
     while(True):
        cv2.imshow(window_name,img)
        
        if cv2.waitKey(20)==27:
            break
     cv2.waitKey(0)
     cv2.destroyAllWindows()
     
     b=0
     c=0
     print(a);
     if "+" in a:
         s=a.split("+")
         b=int(s[0])
         #c=int(s[1])
         print(str((b+c)))
     elif "-" in a:
         s=a.split("-")
         b=int(s[0])
         c=int(s[1])
         print(str((b-c)))
     elif "*" in a:
         s=a.split("*")
         b=int(s[0])
         c=int(s[1])
         print(str((b*c)))
     elif "/" in a:
         s=a.split("/")
         b=int(s[0])
         c=int(s[1])
         print(str((b//c)))
if __name__ == "__main__":
    main()  
