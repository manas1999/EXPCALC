import numpy as np
import cv2 
drawing = False
window_name ='MANAS'
cv2.namedWindow(window_name)
cap = cv2.VideoCapture(0)
img = np.zeros((1012,1012,3),np.uint8)
def draw(event,x,y,flags,param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN :
        drawing = True
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing == True :
        cv2.circle(img,(x,y),5,(0,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        


cv2.setMouseCallback(window_name,draw)

def main():
    while(True):
        cv2.imshow(window_name,img)
        
        if cv2.waitKey(20)==27:
            break

if __name__ == "__main__":
    main()
