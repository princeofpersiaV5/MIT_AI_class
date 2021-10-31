from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
#添加Grad CAM类
class GradCAM:
    def __init__(self,model,class_idx,layerName=None):
        self.model=model
        self.class_idx=class_idx
        self.layerName=layerName

        if self.layerName is None:
            self.layerName=self.find_target_layer()   #如果没有指定用来CAM的层，就调用方法寻找

    def find_target_layer(self):
        for layer in reversed(self.model.layers):   #反向搜索模型中的各层，如果输出是4D的（MAXpool层或者Conv层），就返回层名
            if len(layer.output_shape)==4:
                return layer.name

        raise ValueError('could not find 4D layer')

    def compute_heatmap(self,image,eps=1e-8):
        gradModel=Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,self.model.output]
        )   #Grad CAM的输入是原模型的输入，输出是上面找到的层直接输出（相当于输出寻找到的特征而不是最终标签结果）

        with tf.GradientTape() as tape:
            inputs=tf.cast(image,tf.float32)
            (convOutputs,predictions)=gradModel(inputs)
            loss=predictions[:,self.class_idx]   #记录梯度

        grads=tape.gradient(loss,convOutputs)

        castConvOutputs=tf.cast(convOutputs>0,tf.float32)
        castGrads=tf.cast(grads>0,tf.float32)
        guideGrads=castConvOutputs*castGrads*grads   #计算guide gradient
        #castConvOutputs和castGrads可以在之后让我们可视化神经网络的哪些层被激活了
        convOutputs=convOutputs[0]
        guideGrads=guideGrads[0]

        weights=tf.reduce_mean(guideGrads,axis=(0,1))
        cam=tf.reduce_sum(tf.multiply(weights,convOutputs),axis=-1)    #把被激活部分的梯度保存下来，方便可视化

        (w,h)=(image.shape[2],image.shape[1])
        heatmap=cv2.resize(cam.numpy(),(w,h))
        numer=heatmap-np.min(heatmap)
        denom=(heatmap.max()-heatmap.min())+eps
        heatmap=numer/denom
        heatmap=(heatmap*255).astype("uint8")   #把上面激活部分梯度转化为图片大小的heatmap

        return  heatmap

    def overlay_heatmap(self,heatmap,image,alpha=0.5,colormap=cv2.COLORMAP_VIRIDIS):
        heatmap=cv2.applyColorMap(heatmap,colormap)
        output=cv2.addWeighted(image,alpha,heatmap,1-alpha,0)

        return (heatmap,output)   #设置heatmap的颜色透明度等

