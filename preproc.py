import cv2
import os
import numpy as np
from PIL import Image

path1="H:\Test\left"
path2=r'H:\Test\DT2'
path3=r"H:\Test\project\center_camera"
######################Function for renaming the dataset
#def rename(): 
#    
#    directoryname=r"H:\Test\project\center_camera"
#    lijstmetfiles = os.listdir(directoryname)
#    print(lijstmetfiles)
#    for i in range(len(lijstmetfiles)):
#        os.rename(
#            os.path.join(directoryname, lijstmetfiles[i]),
#            os.path.join(directoryname, "center_"+str(i+1)+".png")
#              )
#        
#
#rename()

listing = os.listdir(path3)
i=1
#
###num_samples=size(listing)
for file in listing:
#    
    im = cv2.imread(path3 + '\\' + "right_"+str(i)+".png")
    im=np.array(im)
    #print(im)
    #print(img.shape)
#   im =  cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    cropped = im[80:135 , 0:320]
    gg =  cv2.resize(cropped,(28,28)) 
    cv2.imwrite('H:\Test\project\gg'+"\\"+"right_"+ str(i)+".png",gg)
    i=i+1
    print("done")
#

    


























