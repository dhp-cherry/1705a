#encoding = utf-8
"""
@author:syj
@file:demo2(GPU).py
@time:2019/10/14 20:12:43
"""
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

import time
tic = time.time()

dir = 'Herb/'

data = os.listdir(dir)

path_list = []
label_data = []
# 获取全路径
def load_data(data):
    num_class = 0
    for num,img_files in enumerate(data):
        imgs = os.listdir(dir+img_files)

        for img in imgs:
            path = os.path.join(dir,img_files,img)
            path_list.append(path)
            label_data.append(num)
        num_class += 1

    data_y = np.array(label_data)
    return data_y,num_class

data_y,num_class = load_data(data)
print(num_class)
print(data_y)
print(len(data_y))

# 根据路径提取图片内容
def feature(filenames):
    img_list = []
    for i in filenames:
        img_data = gfile.GFile(i, 'rb').read()
        img_list.append(img_data)
    return img_list

img_list = feature(path_list)
# img_list = np.array(img_list)
# print(img_list.shape)   #(903,)

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 模型特征提取
def model_feature_extraction(img_list):
    data_x = []

    with gfile.FastGFile('model/tensorflow_inception_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # 从导入的图中得到的与return_element中的名称相对应的操作和/或张量对象的列表。
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def,return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in img_list:
            bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: i})
            bottleneck_values = np.squeeze(bottleneck_values)
            data_x.append(bottleneck_values)
        data_x = np.array(data_x)
    return data_x

data_x = model_feature_extraction(img_list)

train_x, test_x , train_y,test_y = train_test_split(data_x,data_y,test_size=0.2,random_state=7)

x = tf.placeholder(tf.float32,[None,2048])
y = tf.placeholder(tf.int64,[None])

keep_prob = tf.placeholder(tf.float32)

fc1 = tf.layers.dense(x,1024,activation=tf.nn.relu)
fc1 = tf.nn.dropout(fc1,keep_prob=keep_prob)
a5 = tf.layers.dense(fc1,num_class)

cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=a5)

optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(a5,1),y),tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

step = 0
for i in range(1,10001):
    c,a,o = sess.run([cost,accuracy,optimizer],feed_dict={x:train_x[step:step+100],y:train_y[step:step+100],keep_prob:0.7})

    step += 100
    if step >= train_x.shape[0]:
        step = 0

    if i % 1000 == 0:
        print(i,np.mean(c),a)

print(sess.run(accuracy,feed_dict={x:test_x,y:test_y,keep_prob:1}))

toc = time.time()
print('used{:.5}s'.format(toc-tic))

'''
1000 0.3056964 0.93
2000 0.16196263 0.97
3000 0.037518427 1.0
4000 0.04966169 0.98
5000 0.09897548 0.97
6000 0.020785892 1.0
7000 0.03526516 0.99
8000 0.011584333 1.0
9000 0.016937662 1.0
10000 0.007425572 1.0
0.7623153
used409.34s
'''
