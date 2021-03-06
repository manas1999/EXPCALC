import numpy as np
#import tensorflow as tf
#import tensorflow.keras as keras
import keras
from keras.models import Model,Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Input,Flatten
from keras import backend as K
model = load_model("C:\\Users\\manas madine\\Desktop\\expcalc_parameters.h5")
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatch
#from IPython.display import Image
#from tensorflow.keras.preprocessing.image import load_img, array_to_img
import cv2 

###############################################################################
#def neuralnet(testimg):
#    
#
#    img_size=28
#    img_size_flat=img_size*img_size
#    img_shape=(img_size,img_size)
#    num_channels=1
#    num_classes=10
#    model = tf.keras.models.Sequential()
#    model.add(tf.keras.layers.Flatten())
#    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
#    model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
#    model.fit(x_train, y_train, epochs=4)
#    val_loss, val_acc = model.evaluate(x_test, y_test)
##model.save('epic_num_reader.model')
##new_model = tf.keras.models.load_model('epic_num_reader.model')
#    #predictions = model.predict(x_test)
#    #print(np.argmax(predictions[6]))
#    print(model.predict(testimg));
#    #plt.imshow(x_train[6],cmap=plt.cm.binary)
#    #plt.show()
###############################################################################


#mnist = tf.keras.datasets.mnist
#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#drawing = False
window_name ='MANAS'
cv2.namedWindow(window_name)
#cap = cv2.VideoCapture(0)
img = np.zeros((1012,1012,3),np.uint8)
img3 = np.zeros((1012,1012,3),np.uint8)
img3 = img3+255
#while True:
#    _, frame= cap.read()
#    frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#    mask = cv2.inRange(frame2,np.array([0,200,0]),np.array([0,255,0]))
#    cv2.imshow("frames",frame)
#    cv2.imshow("mask",mask)
#    if cv2.waitKey(20)==27:
#            break
#cv2.destroyAllWindows()
#print("hello world")

#for i in dir(cv2):
#    print(i)

def draw(event,x,y,flags,param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN :
        drawing = True;
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing == True :
        cv2.circle(img,(x,y),5,(0,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        





cv2.setMouseCallback(window_name,draw)



def main():
    
     global contours
     while(True):
        if cv2.waitKey(20)==27:
            break
        cv2.imshow(window_name,img)
        
        if cv2.waitKey(20)==27:
            break
        #img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grayimg,(5,5),0)
        #mask = cv2.inRange(grayimg,(0,200,0),(0,255,0))
        #cv2.imshow(window_name,mask)
        ret,thresh = cv2.threshold(blur,25,200,0)
    
        img2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
    
        #print(contours)
        cv2.drawContours(img3,contours,-1,(0,0,0),10)
        #cv2.imshow("contours",img3)
        #print(.ndim)
        #cont = sorted(contours)
        
        
        if cv2.waitKey(20)==27:
            break
        
#print(contours)
#    frame2 = cv2.cvtColor(window_name,cv2.COLOR_BGR2HSV)
#    mask = cv2.inRange(frame2,np.array([0,200,0]),np.array([0,255,0]))
#    cv2.imshow("mask",mask)  
#    if cv2.waitKey(20)==27:
     #print( min1);
     cv2.destroyAllWindows()
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

     print("min X is",min1,"max X is :",max1)
     print("min Y is",min2,"max Y is :",max2)
     img2 = img3[min2:max2 , min1:max1]
     #img4 = cv2.resize(img2,(28,28),interpolation = cv2.INTER_AREA )
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
     cv2.imshow("frame",img4)
     img4 = cv2.resize(img4,(28,28))
     img5 = img4.reshape((1,28,28,1))
    #img3[min2:max2 , min1:max1] = [255,255,255];
     cv2.imshow("hello",img2)
     cv2.waitKey(0);
     cv2.destroyAllWindows()
     print("hello world")
     #neuralnet(img4);
     print(model.predict(img5));

if __name__ == "__main__":
    main()  
