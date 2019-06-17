import pypylon.pylon as py

cam = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice())
cam.Open()
print(cam.PixelFormat.Value)

# cam.StartGrabbing()
# with cam.RetrieveResult(1000) as img:
with cam.GrabOne(10000) as img:
	image=img.Array
	import numpy as np
	mean=np.mean(image)
	print(mean)
		
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
		

# import cv2
# rgb_image=cv2.cvtColor(image,cv2.COLOR_BayerRG2BGR)
# plt.imshow(rgb_image)
# plt.show()
# cam.Close()

