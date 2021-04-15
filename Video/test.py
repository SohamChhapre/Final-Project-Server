import cv2
import numpy as np 
from keras.models import load_model
import argparse
from PIL import Image
import imutils
import imageio
import moviepy.editor


def mean_squared_loss(x1,x2):
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples

    return mean_distance


def Video_Analysis(vid_Name,model,filePath):
    cap=imageio.get_reader(filePath)
    video = moviepy.editor.VideoFileClip(filePath)
    video_duration = int(video.duration)
    print("duration",video_duration)
    print("cap",cap)
    print("iscapopened",cap.closed)
    counter=0
    flag=0
    interval=[]
    lst=[-1,-1]
    frames=[]
    for im in cap:
        frames.append(im)
    ind=0
    no_of_frames=len(frames)
    inc=video_duration/no_of_frames
    print(len(frames))
    reconstruction_array=[['time', 'reconstruction_error']]
    while ind<len(frames):
        imagedump=[]

        frame=frames[ind]
        counter+=inc
        ind+=1
        for i in range(10):
            
            if ind==len(frames):
                break
            frame=frames[ind]
            counter+=inc
            ind+=1
            image = imutils.resize(frame,width=1000,height=1200)

            frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
            gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
            gray=(gray-gray.mean())/gray.std()
            gray=np.clip(gray,0,1)
            imagedump.append(gray)
            

        imagedump=np.array(imagedump)

        imagedump.resize(227,227,10)
        imagedump=np.expand_dims(imagedump,axis=0)
        imagedump=np.expand_dims(imagedump,axis=4)

        output=model.predict(imagedump)

        loss=mean_squared_loss(imagedump,output)
        reconstruction_array.append([counter,loss*100000000])
        print(loss)
        if frame is None:
            break
        if frame.any()==None:
            print("none")
        

        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        if loss>0.00062:
            if flag==0:
                lst[0]=counter
                flag=1
                print(lst, interval)

        else:
            if flag==1:
                lst[1]=counter-1
                flag=0
                if not lst==[-1,-1]:
                    interval.append([lst[0],lst[1]])
                print(lst, interval)

    cv2.destroyAllWindows()
    
    print(interval,counter)
    return interval,reconstruction_array

