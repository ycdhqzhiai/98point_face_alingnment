# 98point_face_alingnment
fork from https://github.com/goodluckcwl/Face-alignment-mobilenet-v2

## 98点人脸landmarks face alignment<br>

数据集下载地址：[WFLW](https://wywu.github.io/projects/LAB/WFLW.html)<br>

## 关于预处理

按照[Face-alignment-mobilenet-v2](https://github.com/goodluckcwl/Face-alignment-mobilenet-v2)方式预处理图片，
如果人脸检测是使用MTCNN，可能效果不会很好，因为这种预处理方式人脸下方区域太多
建议使用人脸检测模型重新生成landmarks坐标，这样效果会好很多<br>

## 关于caffe

goodluckcwl版本的[caffe](https://github.com/goodluckcwl/custom-caffe) triplet loss没有编译通过，最后直接删掉了<br>
caffe WeightEuclideanLoss层根据98点分布自行修改<br>
