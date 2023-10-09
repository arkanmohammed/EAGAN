# EAGAN
Requirements are NVIDIA-24GB environment 

Sender Side 
1. python ./sender-mode/encoderGAN/encoderGAN.py --file ./sender-mode/encoderGAN/assets/13_Ore.jpg --model-name ./sender-mode/encoderGAN/weights/netG_A2B.pth --cuda

2. Run SHA-encrpt.py


Receiver Side
1. python ./sender-mode/encoderGAN/encoderGAN.py --file ./sender-mode/encoderGAN/assets/13_Ore.jpg --model-name ./sender-mode/encoderGAN/weights/netG_B2A.pth --cuda

2. Run SHA-Decrypt.py

