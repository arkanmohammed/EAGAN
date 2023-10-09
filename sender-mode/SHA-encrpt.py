"""
Encrypting an image through substitution algorithm 
using pseudo-random numbers generated from
Lorenz system of differential equations
"""

# Importing all the necessary libraries
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import lorenzSystem as key
import lorenzSystem as key1
import lorenzSystem as key2
import numpy as np
from PIL import Image
import hmac
import hashlib
import base64
#from PIL import Image
import cv2

from numpy.core.fromnumeric import size
from qrcode import constants
import qrcode 
import pyqrcode
from PIL import Image 
import cv2
from typing import Type
import cv2
import numpy as np
import PIL
import skimage.io
import cv2
import numpy as np

import cv2 
import os 
import glob 
from PIL import Image
from timeit import default_timer as timer

start = timer()
img_dir = "./Encrpt-Gan/image" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
data = [] 
for f1 in files:
    import lorenzSystem as key
    import lorenzSystem as key1
    import lorenzSystem as key2 
    from datetime import datetime

    #im=Image.open(f1)
    #data.append(im) 
  #  print (f1)
#############################################
# Accepting Image using it's path
  #  path = str(input('Enter path of the image\n'))
#Pathimage = img.imread(path)
#Pathimage = skimage.io.imread(path)
    imgorgnail = Image.open(f1)
   # data.append(imgorgnail) 

    new_image = imgorgnail.resize((700, 700))
    Pathimage = np.array(new_image)

# Displaying original image
   # plt.imshow(Pathimage)
   # plt.show()

# Storing the size of image in variables
    height = Pathimage.shape[0]
    width = Pathimage.shape[1]
    l = 0
############################################################################################
    xkey, ykey, zkey = key.lorenz_key(0.0003, 0.02, 0.03, 700*700)

#xkey, ykey, zkey = key.lorenz_key(0.01, 0.02, 0.03, height*width)
####################################################################################### XOR1
# Initializing an empty image to store the encrypted image

    encryptedImage1 = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    encryptedImage2 = np.zeros(shape=[height, width, 3], dtype=np.uint8)

# XORing each pixel with a pseudo-random number generated above/ Performing the 
# substitution algorithm
    for i in range(height):
        for j in range(width):
        # Converting the pseudo-random nuber generated into a number between 0 and 255
            zk = (int((zkey[l]*pow(10, 5))%256))
        # Performing the XOR operation
        #encryptedImage[i, j] = image[i, j]^zk
            encryptedImage1[i, j] = Pathimage[i, j]

            encryptedImage2[i, j] = zk
            l += 1


# cv2.bitwise_xor is applied over the
# image inputs with applied parameters
    dest_xor1 = cv2.bitwise_xor(encryptedImage1, encryptedImage2, mask = None)
   # date = datetime.now()
####################################################################################### XOR1

# Initializing empty index lists to store index of pixels
    xindex = []
    yindex = []

# Initializing an empty image to store the encrypted image
    encryptedImage = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    l = 0

# Populating xindex
    for i in range(width):
        xindex.append(i)

# Populating yindex
    for i in range(height):
        yindex.append(i)

# Re-arranging xindex and xkey to increase randomness
    for i in range(width):
        for j in range(width):
            if xkey[i] > xkey[j]:
                xkey[i], xkey[j] = xkey[j], xkey[i]
    for ii in range(width):
        for jj in range(width):
            if ykey[ii] > ykey[jj]:
                ykey[ii], ykey[jj] = ykey[jj], ykey[ii]    
#################################################### Hash MAC

    key = bytes(xkey[j])
    print ("key1 =",key)
    msg1 = bytes( ykey[jj])
    dig = hmac.new(key, msg=msg1 , digestmod=hashlib.sha256).hexdigest().upper()
#base64.b64encode(dig).decode()      # py3k-mode
#'Nace+U3Az4OhN7tISqgs1vdLBHBEijWcBeCqL5xN9xg='adll
    digg = int(dig, 16)
    a_list = [int(yy) for yy in str(digg)]
    print ("Hash Value=" , dig)
#################################################### creat intial value X of chaose from hash  

    aa=a_list[0:6]
    my_lst_str1 = ''.join(map(str, aa))
    my_lst_str1 = int(my_lst_str1)/100000000
    print("intial value X =", my_lst_str1)
#################################################### intial value X
    aa2=a_list[7:15]
    my_lst_str2 = ''.join(map(str, aa2))

    my_lst_str2 = int(my_lst_str2)/100000000
    print("intial value Y =", my_lst_str2)
#####################################################intial value Z
    aa3=a_list[20:25]
    my_lst_str3 = ''.join(map(str, aa3))
    my_lst_str3 = int(my_lst_str3)/100000000
    print("intial value Z =", my_lst_str3)
######################################################### Lorenz chaos


    x, y, keys = key1.lorenz_key(my_lst_str1, my_lst_str2, my_lst_str3, height*width)

#################### SHA of output gan or orignail image 
    for i in range(width):
        for j in range(width):
            if x[i] > x[j]:
                x[i], x[j] = x[j], x[i]
    key4 = bytes(x[j])
    print ("key1 =",x[j])
    msg1 = dest_xor1
    dig1 = hmac.new(key4, msg=msg1 , digestmod=hashlib.sha256).hexdigest().upper()
    print ("Hash Value2=" , dig1)
    dig1 = int(dig1, 16)
    l = 0

# Initializing an empty image to store the encrypted image
    encryptedImage = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    encryptedImage1 = np.zeros(shape=[height, width, 3], dtype=np.uint8)

# XORing each pixel with a pseudo-random number generated above/ Performing the 
# substitution algorithm
    for i in range(height):
        for j in range(width):
        # Converting the pseudo-random nuber generated into a number between 0 and 255
            zk = (int((keys[l]*pow(10, 5))%256))
        # Performing the XOR operation
        #encryptedImage[i, j] = image[i, j]^zk
            encryptedImage1[i, j] = dest_xor1[i,j]

            encryptedImage[i, j] = zk
            l += 1
    cv2.imwrite('./sender-mode/encoderGAN/assets/13_Ore.jpg',cv2.cvtColor(dest_xor1, cv2.COLOR_RGB2BGR) )
    print (len(encryptedImage), len(encryptedImage1))
    imgorgnail = Image.open("./sender-mode/image-output/Images.png")
    new_image = imgorgnail.resize((700, 700))
    Pathimage = np.array(new_image)
######## QR code
    file_name = str(dig1)
    qr = qrcode.QRCode(box_size = 3)
    qr.add_data(file_name)
    qr.make()
    img = qr.make_image()
    img.save('./Encrpt-Gan/sender-mode/Result-encrypt/qrcode.png')
#img_1 = cv2.imread('/content/drive/MyDrive/Encrpt-Gan/sender-mode/Result-encrypt/Images.png')
    img_1 = encryptedImage1

    img_2 = cv2.imread('./Encrpt-Gan/sender-mode/Result-encrypt/qrcode.png')
#img_2 = cv2.imread('/content/drive/MyDrive/Encrpt-Gan/sender-mode/Result-encrypt/qrcode.png')

    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]

    img_3 = np.zeros((h1+h2, max(w1,w2),3), dtype=np.uint8)
    img_3[:,:] = (255,255,255)

    img_3[:h1, :w1,:3] = img_1
    img_3[h1:h1+h2, :w2,:3] = img_2
    cv2.imwrite('./sender-mode/Result-encrypt/Fqrcode.png', cv2.cvtColor(img_3, cv2.COLOR_RGB2BGR) )
    cv2.imwrite(os.path.join("./sender-mode/images-QRcode/images-QRcode","Fqrcode"+os.path.basename(f1)), cv2.cvtColor(img_3, cv2.COLOR_RGB2BGR))

############ xor operation 
    Fqrcode = skimage.io.imread('./sender-mode/Result-encrypt/Fqrcode.png')
    height = Fqrcode.shape[0]
    width = Fqrcode.shape[1]
    x, y, keys = key2.lorenz_key(my_lst_str1, my_lst_str2, my_lst_str3, height*width)
    l = 0

# Initializing an empty image to store the encrypted image
    encryptedImage11 = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    encryptedImage12 = np.zeros(shape=[height, width, 3], dtype=np.uint8)

# XORing each pixel with a pseudo-random number generated above/ Performing the 
# substitution algorithm
    for i in range(height):
        for j in range(width):
        # Converting the pseudo-random nuber generated into a number between 0 and 255
            zk = (int((keys[l]*pow(10, 5))%256))
        # Performing the XOR operation
        #encryptedImage[i, j] = image[i, j]^zk
            encryptedImage11[i, j] = Fqrcode[i, j]

            encryptedImage12[i, j] = zk
            l += 1


# cv2.bitwise_xor is applied over the
# image inputs with applied parameters
    dest_xor = cv2.bitwise_xor(encryptedImage11, encryptedImage12, mask = None)
   # date = datetime.now()
    #cv2.imwrite(os.path.join("./Encrpt-Gan/sender-mode/image-output",os.path.basename(f1)), dest_xor)
    cv2.imwrite(os.path.join("./sender-mode/image-output/"+"Cipherim.png"), dest_xor)
end = timer()
print(end - start)

   # cv2.imwrite("/content/drive/MyDrive/Encrpt-Gan/sender-mode/Result-encrypt/"+str(f1)+".png", dest_xor)
