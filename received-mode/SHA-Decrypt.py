"""
Encrypting an image through substitution algorithm 
using pseudo-random numbers generated from
Lorenz system of differential equations
"""
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
# Importing all the necessary libraries
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import lorenzSystem as key
import lorenzSystem as key1
import lorenzSystem as key2
import cv2 
import os 
import hmac
import hashlib
import base64
#from PIL import Image
import cv2
from PIL import Image
import imageio
from typing import Type
from pyzbar import pyzbar
from PIL import Image
import cv2
import os 
import glob 

#############################################
# Accepting Image using it's path
#path = str(input('Enter path of the image\n'))
img_dir = "./Encrpt-Gan\sender-mode\image-output" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*png') 
files = glob.glob(data_path) 
data = [] 
for f1 in files:
    import lorenzSystem as key
    import lorenzSystem as key1
    import lorenzSystem as key2 

   # im=Image.open(f1)
  #  data.append(im) 
    #path = "Encrpt-Gan/sender-mode/image-output/33.png"
#image = img.imread(path)

#image = imageio.imread(path)
# Initializing an empty image to store the decrypted image
    imgorgnail = Image.open(f1)
   # data.append(imgorgnail) 
    image = np.array(imgorgnail)

# Displaying original image

# Initializing empty index lists to store index of pixels
    xindex = []
    yindex = []
# Storing the size of image in variables
    height = image.shape[0]
    width = image.shape[1]
# Initializing an empty image to store the encrypted image
    encryptedImage = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    l = 0

###############################################################33



############################################################################################
    xkey, ykey, zkey = key.lorenz_key(0.0003, 0.02, 0.03, 700*811)

# Populating xindex
    for i in range(700):
        xindex.append(i)

# Populating yindex
    for i in range(700):
        yindex.append(i)

# Re-arranging xindex and xkey to increase randomness
    for i in range(700):
        for j in range(700):
            if xkey[i] > xkey[j]:
                xkey[i], xkey[j] = xkey[j], xkey[i]
    for ii in range(700):
        for jj in range(700):
            if ykey[ii] > ykey[jj]:
                ykey[ii], ykey[jj] = ykey[jj], ykey[ii]    
#################################################### Hash MAC

    key = bytes(xkey[j])
    print ("key1 =",key)
    msg1 = bytes( ykey[jj])
    dig = hmac.new(key, msg=msg1 , digestmod=hashlib.sha256).hexdigest().upper()
#base64.b64encode(dig).decode()      # py3k-mode
#'Nace+U3Az4OhN7tISqgs1vdLBHBEijWcBeCqL5xN9xg='adll
#digg = int(dig, 16)
#a_list = [int(yy) for yy in str(digg)]
    print ("Hash Value=" , dig)


############################################################################################33
#dig = str(input('Enter path of the HashValues\n'))

#dig = "2B40385E6CA0B394A99A21B0AA9B1F0F01E054B7D85885230245DCF00103425B"
    digg = int(dig, 16)
    a_list = [int(yy) for yy in str(digg)]
    print ("Hash Value=" , dig)
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

    l = 0

# Initializing an empty image to store the encrypted image
    encryptedImage = np.zeros(shape=[height, width, 3], dtype=np.uint8)

# XORing each pixel with a pseudo-random number generated above/ Performing the 
# substitution algorithm
    for ii in range(height):
        for jj in range(width):
        # Converting the pseudo-random nuber generated into a number between 0 and 255
        #zk = (int((keys[l]*pow(10, 5))%256))
        # Performing the XOR operation
            encryptedImage[ii, jj] = image[ii, jj]
            l += 1
      #  print ("key=", zk)




    l = 0
# Initializing an empty image to store the encrypted image
# Initializing an empty image to store the decrypted image
    decryptedImage = np.zeros(shape=[height, width, 3], dtype=np.uint8)

# XORing each pixel with the same number it was XORed above above/
# Performing the reverse substitution algorithm
    for i in range(height):
        for j in range(width):
            zk = (int((keys[l]*pow(10, 5))%256))
            decryptedImage[i, j] = encryptedImage[i, j]^zk
            l += 1

# Displaying the decrypted image
    
    cv2.imwrite('./Encrpt-Gan/received-mode/result-middle/output.png',decryptedImage)
################################ scanning QR code 
    cv2.imwrite(os.path.join("./Encrpt-Gan/received-mode/result-QRcode","decryptedImage"+os.path.basename(f1)), cv2.cvtColor(decryptedImage, cv2.COLOR_RGB2BGR))
#load qr code imge
    img = Image.open('./Encrpt-Gan/received-mode/result-middle/output.png')

    qr_code = pyzbar.decode(img)[0]

#convert into string
    data= qr_code.data.decode("utf-8")
    type = qr_code.type
    text = f" {data}"
    print (text)

################# crop image 
    box = (0, 0, 700, 700)
    img2 = img.crop(box)
    img2.save('./Encrpt-Gan/received-mode/result-middle/Foutput.png')
#digg = int(dig, 16)
    img = Image.open('./Encrpt-Gan/received-mode/result-middle/Foutput.png')
    new_image = img.resize((700, 700))
    Pathimage = np.array(new_image)
# Displaying original image
#plt.imshow(Pathimage)

# Storing the size of image in variables
    height = Pathimage.shape[0]
    width = Pathimage.shape[1]
    x, y, keys = key1.lorenz_key(my_lst_str1, my_lst_str2, my_lst_str3, height*width)

#################### SHA of output gan or orignail image 
    for i in range(width):
        for j in range(width):
            if x[i] > x[j]:
                x[i], x[j] = x[j], x[i]
    key1 = bytes(x[j])
    print ("key1 =",x[j])
    msg1 = Pathimage
    dig1 = hmac.new(key1, msg=msg1 , digestmod=hashlib.sha256).hexdigest().upper()
    print ("Hash Value2New=" ,dig1 )
    dig1 = int(dig1, 16)
    print ("dig1=", dig1)
#text = int (text)
#print ("Hash Value2New=" , text)
########## Compare two Hash Value 

    if ( int(dig1) == int(text) ):
      print ("The Image is correct")
    else:
      print ("The Image is incorrect")
####################################################################################### XOR1
    l = 0


# Initializing an empty image to store the encrypted image
    encryptedImage1 = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    encryptedImage2 = np.zeros(shape=[height, width, 3], dtype=np.uint8)

# XORing each pixel with a pseudo-random number generated above/ Performing the 
# substitution algorithm
    for ir in range(height):
        for jr in range(width):
# Converting the pseudo-random nuber generated into a number between 0 and 255
            zk = (int((zkey[l]*pow(10, 5))%256))
        # Performing the XOR operation
        #encryptedImage[i, j] = image[i, j]^zk
            encryptedImage1[ir, jr] = Pathimage[ir, jr]

            encryptedImage2[ir, jr] = zk
            l += 1


# cv2.bitwise_xor is applied over the
# image inputs with applied parameters
    dest_xor1 = cv2.bitwise_xor(encryptedImage1, encryptedImage2, mask = None)
   # date = datetime.now()
    #cv2.imwrite('./Encrpt-Gan/received-mode/Result-Decrypt/Foutput.png',cv2.cvtColor(dest_xor1, cv2.COLOR_RGB2BGR) )
    cv2.imwrite(os.path.join("./Encrpt-Gan/received-mode/Result-Decrypt","Foutput"+os.path.basename(f1)), cv2.cvtColor(dest_xor1, cv2.COLOR_RGB2BGR))

