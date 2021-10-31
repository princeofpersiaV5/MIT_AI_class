from Grad_CAM import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import imutils
import cv2
import tensorflow as tf


Model=tf.keras.models.load_model(r"C:\Users\hp\Desktop\project assignments(DL)\cnn_vgg19_v3.h5")
print("[INFO] loading model...")
PATH=r"C:\Users\hp\Desktop\project assignments(DL)\Dataset\Val\Covid_val\extubation-4.jpg"   #picture path
model=Model    #导入之前训练好的分类模型

img = cv2.imread(PATH)
orig=img
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0)

img = np.array(img) / 255.0
image=img

preds=model.predict(image)
i=np.argmax(preds[0])   #通过模型预测标签
# decoded=imagenet_utils.decode_predictions(preds)
# (imagenetID,label,prob)=decoded[0][0]
# label="{}:{:.2f}%".format(label,prob*100)
# print("[INFO]{}".format(label))
if i==0:
    text=str(i)+": COVID-19"
else:
    text=str(i)+": Normal"

cam=GradCAM(model,i)
heatmap=cam.compute_heatmap(image)   #生成heatmap

heatmap=cv2.resize(heatmap,(orig.shape[1],orig.shape[0]))
(heatmap,output)=cam.overlay_heatmap(heatmap,orig,alpha=0.5)

cv2.rectangle(output,(0,0),(340,40),(0,0,0),-1)
cv2.putText(output,text,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

output=np.vstack([orig,heatmap,output])
output=imutils.resize(output,height=700)
cv2.imshow("Output",output)   #分别输出原图、heatmap图和叠加图
cv2.waitKey(0)
