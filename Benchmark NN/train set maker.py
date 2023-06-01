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

print('making train set')
i,x_train,y_train=0,[],[]
for xc in range(xc1,xc2,dx):
    for yc in range(yc1,yc2,dy):
        for r in range(r1,r2,dr):
            img=cv2.circle(blank.copy(),(xc,yc),r,0,-1)
            x_train.append(img)
            y_train.append([1,0,xc,yc,2*r,2*r])
            cv2.imwrite(f'train set//circ_{i}.png',img)
            i+=1
            dispProgressBar(i,imax)
        for w in range(w1,w2,dw):
            for h in range(h1,h2,dh):
                x1,y1=xc-w//2,yc-h//2
                x3,y3=x1+w//2,yc+h//2
                img=cv2.rectangle(blank.copy(),(x1,y1),(x3,y3),0,-1)
                x_train.append(img)
                y_train.append([0,1,xc,yc,w,h])
                cv2.imwrite(f'train set//rect_{i}.png',img)
                i+=1
                dispProgressBar(i,imax)
x_train,y_train=np.array(x_train)/255,np.array(y_train)
np.save('x_train',x_train)
np.save('y_train',y_train)
print('saved train set')
