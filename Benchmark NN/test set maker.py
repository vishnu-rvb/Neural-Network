def dispProgressBar(i,imax,startTxt='',endTxt=''):
    percent=int(100*i/imax)
    bar='█'*percent+'-'*(100-percent)
    print(f'{startTxt} |{bar}| {percent}% {endTxt}',end="\r")
    if i==imax:print(end='\n')

import cv2
import numpy as np

blank=np.ones((200,200,1),dtype='uint8')*255
shapes=['circle','rectangle']

xc1,xc2,dx=50,150,10
yc1,yc2,dy=50,150,10
r1,r2,dr=25,50,1
w1,w2,dw=50,100,10
h1,h2,dh=50,100,10
nxc,nyc,nr=(xc2-xc1)/dx,(yc2-yc1)/dy,(r2-r1)/dr
nw,nh=(w2-w1)/dw,(h2-h1)/dh
imax=nxc*nyc*(nr+nw*nh)
#print(f'count={int(imax)} size={imax*200*200*1*4/(1024**3):.2f} GB')
#quit()

print('making test set')
from random import choice,randrange
i,x_test,y_test=0,[],[]
t=1000
y_train=np.load('y_train.npy')
while True:
    #dispProgressBar(i,t-1)
    shape=choice(shapes)
    xc,yc=randrange(xc1,xc2),randrange(yc1,yc2)
    if shape=='circle':
        r=randrange(r1,r2)
        img=cv2.circle(blank.copy(),(xc,yc),r,0,-1)
        if [1,0,xc,yc,2*r,2*r] not in y_train.tolist():
            i+=1
            x_test.append(img)
            y_test.append([1,0,xc,yc,2*r,2*r])
            cv2.imwrite(f'test set//Tcirc_{i}.png',img)
    if shape=='rectangle':
        w,h=randrange(w1,w2),randrange(h1,h2)
        x1,y1=xc-w//2,yc-h//2
        x3,y3=x1+w//2,yc+h//2
        if [0,1,xc,yc,w,h] not in y_train.tolist():
            i+=1
            img=cv2.rectangle(blank.copy(),(x1,y1),(x3,y3),0,-1)
            x_test.append(img)
            y_test.append([0,1,xc,yc,w,h])
            cv2.imwrite(f'test set//Trect_{i}.png',img)
    if i==t:break
x_test,y_test=np.array(x_test)/255,np.array(y_test)
np.save('x_test',x_test)
np.save('y_test',y_test)
print('saved test set')
