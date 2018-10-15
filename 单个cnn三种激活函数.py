# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:54:41 2018

@author: lgq-yun
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 20:04:03 2018

@author: ligq
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Input,Model,Sequential
from keras.layers import Dense,Activation,Flatten,Add,Reshape,add,Average
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D,GlobalAveragePooling2D,MaxPool2D
from keras import optimizers
import random 
import keras
from sklearn.model_selection import KFold,StratifiedKFold

batch_size = 32
nb_classes = 12
epochs = 40

'''
转化标签样式，形成HOT_number
'''
def dense_to_one_hot(labels_dense,num_classes=12):
    return np.eye(num_classes)[labels_dense]

'''
获取训练和测试数据并且随机打乱训练数据,注意获取三种数据集的文件目录位置
train-root_dir1
test-root_dir2
valid-root_dir3
'''
def get_data_shuffle(root_dir1,root_dir2,root_dir3):
    state1 = [] 
    state2 = []
    state3 = []
    state4 = [] 
    state5 = []
    state6 = []
    state7 = [] 
    state8 = []
    state9 = []
    state10 = [] 
    state11 = []
    state12 = []
    for i in range(len(os.listdir(root_dir1))):
        file_dir1 = root_dir1 + 'state' + str(i+1) + '/'
        for file1 in os.listdir(file_dir1):
            file_name1 = file_dir1 + file1
            with open(file_name1) as csvfile1:
                tmp_data1 = np.loadtxt(csvfile1,delimiter=",",skiprows=0)
                if i == 1:
                    state1.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                elif i == 2:
                    state2.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                elif i == 3:
                    state3.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                elif i == 4:
                    state4.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                elif i == 5:
                    state5.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                elif i == 6:
                    state6.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                elif i == 7:
                    state7.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                elif i == 8:
                    state8.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                elif i == 9:
                    state9.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                elif i == 10:
                    state10.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                elif i == 11:
                    state11.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
                else:
                    state12.append([tmp_data1,dense_to_one_hot(i,num_classes=12)]) 
    train_x = []
    train_y = []
    for i in range(len(os.listdir(root_dir2))):
        file_dir2 = root_dir2 + 'state' + str(i+1) + '/'
        for file2 in os.listdir(file_dir2):
            file_name2 = file_dir2 + file2
            with open(file_name2) as csvfile2:
                tmp_data2 = np.loadtxt(csvfile2,delimiter=",",skiprows=0)
                if i == 1:
                    state1.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                elif i == 2:
                    state2.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                elif i == 3:
                    state3.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                elif i == 4:
                    state4.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                elif i == 5:
                    state5.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                elif i == 6:
                    state6.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                elif i == 7:
                    state7.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                elif i == 8:
                    state8.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                elif i == 9:
                    state9.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                elif i == 10:
                    state10.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                elif i == 11:
                    state11.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
                else:
                    state12.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])
    test_x = []
    test_y = []
    valid_x = []
    valid_y = []
    for i in range(len(os.listdir(root_dir3))):
        file_dir3 = root_dir3 + 'state' + str(i+1) + '/'
        for file3 in os.listdir(file_dir3):
            file_name3 = file_dir3 + file3
            with open(file_name3) as csvfile3:
                tmp_data3 = np.loadtxt(csvfile3,delimiter=",",skiprows=0)
                if i == 1:
                    state1.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                elif i == 2:
                    state2.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                elif i == 3:
                    state3.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                elif i == 4:
                    state4.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                elif i == 5:
                    state5.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                elif i == 6:
                    state6.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                elif i == 7:
                    state7.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                elif i == 8:
                    state8.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                elif i == 9:
                    state9.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                elif i == 10:
                    state10.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                elif i == 11:
                    state11.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
                else:
                    state12.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])     
    random.shuffle(state1)
    random.shuffle(state2)
    random.shuffle(state3)
    random.shuffle(state4)
    random.shuffle(state5)
    random.shuffle(state6)
    random.shuffle(state7)
    random.shuffle(state8)
    random.shuffle(state9)
    random.shuffle(state10)
    random.shuffle(state11)
    random.shuffle(state12)
    valid_x = []
    all_data_x = []
    all_data_y = []
    all_data = []
    for i in range(int(len(state1))):
        all_data_x.append(state1[i][0])
        all_data_y.append(state1[i][0])
        all_data.append([state1[i][0],state1[i][1]])
        if i < (int(0.7*len(state1))):
            train_x.append(state1[i][0])
            train_y.append(state1[i][1])
        elif i < (int(0.85*len(state1))):
            valid_x.append(state1[i][0])
            valid_y.append(state1[i][1])
        else:
            test_x.append(state1[i][0])
            test_y.append(state1[i][1])
            
    for i in range(int(len(state2))):
        all_data_x.append(state2[i][0])
        all_data_y.append(state2[i][1])
        all_data.append([state2[i][0],state2[i][1]])
        if i < (int(0.7*len(state2))):
            train_x.append(state2[i][0])
            train_y.append(state2[i][1])
        elif i < (int(0.85*len(state2))):
            valid_x.append(state2[i][0])
            valid_y.append(state2[i][1])
        else:
            test_x.append(state2[i][0])
            test_y.append(state2[i][1]) 
            
    for i in range(int(len(state3))):
        all_data_x.append(state3[i][0])
        all_data_y.append(state3[i][1])
        all_data.append([state3[i][0],state3[i][1]])
        if i < (int(0.7*len(state3))):
            train_x.append(state3[i][0])
            train_y.append(state3[i][1])
        elif i < (int(0.85*len(state3))):
            valid_x.append(state3[i][0])
            valid_y.append(state3[i][1])
        else:
            test_x.append(state3[i][0])
            test_y.append(state3[i][1])
            
    for i in range(int(len(state4))):
        all_data_x.append(state4[i][0])
        all_data_y.append(state4[i][1])
        all_data.append([state4[i][0],state4[i][1]])
        if i < (int(0.7*len(state4))):
            train_x.append(state4[i][0])
            train_y.append(state4[i][1])
        elif i < (int(0.85*len(state4))):
            valid_x.append(state4[i][0])
            valid_y.append(state4[i][1])
        else:
            test_x.append(state4[i][0])
            test_y.append(state4[i][1])
            
    for i in range(int(len(state5))):
        
        all_data_x.append(state5[i][0])
        all_data_y.append(state5[i][1])
        all_data.append([state5[i][0],state5[i][1]])
        if i < (int(0.7*len(state5))):
            train_x.append(state5[i][0])
            train_y.append(state5[i][1])
        elif i < (int(0.85*len(state5))):
            valid_x.append(state5[i][0])
            valid_y.append(state5[i][1])
        else:
            test_x.append(state5[i][0])
            test_y.append(state5[i][1])
            
    for i in range(int(len(state6))):
        all_data_x.append(state6[i][0])
        all_data_y.append(state6[i][1])
        all_data.append([state6[i][0],state6[i][1]])
        if i < (int(0.7*len(state6))):
            train_x.append(state6[i][0])
            train_y.append(state6[i][1])
        elif i < (int(0.85*len(state6))):
            valid_x.append(state6[i][0])
            valid_y.append(state6[i][1])
        else:
            test_x.append(state6[i][0])
            test_y.append(state6[i][1])
            
    for i in range(int(len(state7))):
        all_data_x.append(state7[i][0])
        all_data_y.append(state7[i][1])
        all_data.append([state7[i][0],state7[i][1]])
        if i < (int(0.7*len(state7))):
            train_x.append(state7[i][0])
            train_y.append(state7[i][1])
        elif i < (int(0.85*len(state7))):
            valid_x.append(state7[i][0])
            valid_y.append(state7[i][1])
        else:
            test_x.append(state7[i][0])
            test_y.append(state7[i][1])
            
    for i in range(int(len(state8))):
        all_data_x.append(state8[i][0])
        all_data_y.append(state8[i][1])
        all_data.append([state8[i][0],state8[i][1]])
        if i < (int(0.7*len(state8))):
            train_x.append(state8[i][0])
            train_y.append(state8[i][1])
        elif i < (int(0.85*len(state8))):
            valid_x.append(state8[i][0])
            valid_y.append(state8[i][1])
        else:
            test_x.append(state8[i][0])
            test_y.append(state8[i][1])
            
    for i in range(int(len(state9))):
        all_data_x.append(state9[i][0])
        all_data_y.append(state9[i][1])
        all_data.append([state9[i][0],state9[i][1]])
        if i < (int(0.7*len(state9))):
            train_x.append(state9[i][0])
            train_y.append(state9[i][1])
        elif i < (int(0.85*len(state9))):
            valid_x.append(state9[i][0])
            valid_y.append(state9[i][1])
        else:
            test_x.append(state9[i][0])
            test_y.append(state9[i][1])
            
    for i in range(int(len(state10))):
        all_data_x.append(state10[i][0])
        all_data_y.append(state10[i][1])
        all_data.append([state10[i][0],state10[i][1]])
        if i < (int(0.7*len(state10))):
            train_x.append(state10[i][0])
            train_y.append(state10[i][1])
        elif i < (int(0.85*len(state10))):
            valid_x.append(state10[i][0])
            valid_y.append(state10[i][1])
        else:
            test_x.append(state10[i][0])
            test_y.append(state10[i][1])
    for i in range(int(len(state11))):
        all_data_x.append(state11[i][0])
        all_data_y.append(state11[i][1])
        all_data.append([state11[i][0],state11[i][1]])
        if i < (int(0.7*len(state11))):
            train_x.append(state11[i][0])
            train_y.append(state11[i][1])
        elif i < (int(0.85*len(state11))):
            valid_x.append(state11[i][0])
            valid_y.append(state11[i][1])
        else:
            test_x.append(state11[i][0])
            test_y.append(state11[i][1])
            
    for i in range(int(len(state12))):
        all_data_x.append(state12[i][0])
        all_data_y.append(state12[i][1])
        all_data.append([state12[i][0],state12[i][1]])
        if i < (int(0.7*len(state12))):
            train_x.append(state12[i][0])
            train_y.append(state12[i][1])
        elif i < (int(0.85*len(state12))):
            valid_x.append(state12[i][0])
            valid_y.append(state12[i][1])
        else:
            test_x.append(state12[i][0])
            test_y.append(state12[i][1])
    random.shuffle(all_data)
    all_x = []
    all_y = []
    for i in range(len(all_data)):
        all_x.append(all_data[i][0])
        all_y.append(all_data[i][1])
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)
# =============================================================================
    all_x = np.reshape(all_x,[len(all_x),20,4,1])
    all_x = all_x.astype('float32')
    train_x = np.reshape(train_x,[len(train_x),20,4,1])
    train_x = train_x.astype('float32')
    train_y = np.reshape(train_y,[len(train_y),12])
    train_y = train_y.astype('float32')
    valid_x = np.reshape(valid_x,[len(valid_x),20,4,1])
    valid_x = valid_x.astype('float32')
    valid_y = np.reshape(valid_y,[len(valid_y),12])
    valid_y = valid_y.astype('float32')
    all_y = np.reshape(all_y,[len(all_y),12])
    all_y = all_y.astype('float32')
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_x = np.reshape(test_x,[len(test_x),20,4,1])
    test_x = test_x.astype('float32')
    test_y = np.reshape(test_y,[len(test_y),12])
    test_y = test_y.astype('float32')
# =============================================================================
   
    return train_x,train_y,valid_x,valid_y

def get_test_data_shuffle(root_dir1,root_dir2,root_dir3):
    test_data = []
    test_x = []
    test_y = []
    for i in range(len(os.listdir(root_dir1))):
        file_dir1 = root_dir1 + 'state' + str(i+1) + '/'
        for file1 in os.listdir(file_dir1):
            file_name1 = file_dir1 + file1
            with open(file_name1) as csvfile1:
                tmp_data1 = np.loadtxt(csvfile1,delimiter=",",skiprows=0)
                test_data.append([tmp_data1,dense_to_one_hot(i,num_classes=12)])   
    for i in range(len(os.listdir(root_dir2))):
        file_dir2 = root_dir2 + 'state' + str(i+1) + '/'
        for file2 in os.listdir(file_dir2):
            file_name2 = file_dir2 + file2
            with open(file_name2) as csvfile2:
                tmp_data2 = np.loadtxt(csvfile2,delimiter=",",skiprows=0)
                test_data.append([tmp_data2,dense_to_one_hot(i,num_classes=12)])
    for i in range(len(os.listdir(root_dir3))):
        file_dir3 = root_dir3 + 'state' + str(i+1) + '/'
        for file3 in os.listdir(file_dir3):
            file_name3 = file_dir3 + file3
            with open(file_name3) as csvfile3:
                tmp_data3 = np.loadtxt(csvfile3,delimiter=",",skiprows=0)
                test_data.append([tmp_data3,dense_to_one_hot(i,num_classes=12)])
    random.shuffle(test_data)
    for i in range(int(len(test_data))):
        test_x.append(test_data[i][0])
        test_y.append(test_data[i][1])
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_x = np.reshape(test_x,[len(test_x),20,4,1])
    test_x = test_x.astype('float32')
    test_y = np.reshape(test_y,[len(test_y),12])
    test_y = test_y.astype('float32')
    return test_x,test_y    



def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def save_figure(history):
    import matplotlib.lines as mlines
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    file_name_acc = 'E:/黎国强资料/西储大学-CNN/西储数据代码/数据/所有负载，测试集不经过滑动取样/图像数据/val_acc.csv'
    file_name_loss = 'E:/黎国强资料/西储大学-CNN/西储数据代码/数据/所有负载，测试集不经过滑动取样/图像数据/val_loss.csv'
    acc_file_name = 'E:/黎国强资料/西储大学-CNN/西储数据代码/数据/所有负载，测试集不经过滑动取样/图像数据/acc.csv'
    loss_file_name = 'E:/黎国强资料/西储大学-CNN/西储数据代码/数据/所有负载，测试集不经过滑动取样/图像数据/loss.csv'
    np.savetxt(acc_file_name,acc,delimiter=',')
    np.savetxt(loss_file_name,loss,delimiter=',')
    np.savetxt(file_name_acc,val_acc,delimiter=',')
    np.savetxt(file_name_loss,val_loss,delimiter=',')
    
    epoch=range(1,len(acc)+1)
    plt.plot(epoch,acc,'r',marker = "o")
    plt.plot(epoch,smooth_curve(val_acc),'b',marker = "^")
    #plt.title('Trainging and Validation accuracy')
    red_line1 = mlines.Line2D([], [], color='red', marker='o',
                          markersize=5, label='Training')
    blue_line1 = mlines.Line2D([], [], color='blue', marker='^',
                          markersize=5, label='validation')
    plt.xlabel('Training epoch')
    plt.ylabel('Accuracy')
    plt.legend(handles=[red_line1,blue_line1])
    plt.savefig('1.png')
    plt.figure()

    plt.plot(epoch,loss,'r',marker = "o")
    plt.plot(epoch,val_loss,'b',marker = "^")
    red_line1 = mlines.Line2D([], [], color='red', marker='o',
                          markersize=5, label='Training')
    blue_line1 = mlines.Line2D([], [], color='blue', marker='^',
                          markersize=5, label='validation')
    plt.xlabel('Training epoch')
    plt.ylabel('Loss')
    plt.legend(handles=[red_line1,blue_line1])
    #plt.title('Trainging and Validation loss')    
    plt.savefig('2.png')
    plt.figure()
def creat_model(model_input):
    x1 = Conv2D(filters=24,kernel_size=(3,3),padding='Same',activation='tanh')(model_input)
    x2 = Conv2D(filters=24,kernel_size=(3,3),padding='Same',activation='sigmoid')(model_input)
    x3 = Conv2D(filters=24,kernel_size=(3,3),padding='Same',activation='relu')(model_input)
    c1 = keras.layers.concatenate([x1,x2,x3],axis=-1)
    c1 = BatchNormalization(epsilon=1e-6,axis=-1)(c1)
    c1 = MaxPool2D(pool_size=(2,2))(c1)
    b1 = Conv2D(filters=24,kernel_size=(3,3),padding='Same',activation='tanh')(c1)
    b2 = Conv2D(filters=24,kernel_size=(3,3),padding='Same',activation='sigmoid')(c1)
    b3 = Conv2D(filters=32,kernel_size=(1,1),padding='Same',activation='relu')(c1)
    c2 = keras.layers.concatenate([b1,b2,b3],axis=-1)
    c2 = BatchNormalization(epsilon=1e-6,axis=-1)(c2)
    d1 = Conv2D(filters=24,kernel_size=(3,3),padding='Same',activation='tanh')(c2)
    d2 = Conv2D(filters=24,kernel_size=(3,3),padding='Same',activation='sigmoid')(c2)
    d3 = Conv2D(filters=32,kernel_size=(1,1),padding='Same',activation='relu')(c2)
    c3 = keras.layers.concatenate([d1,d2,d3],axis=-1)
    c3 = BatchNormalization(epsilon=1e-6,axis=-1)(c3)
    c3 = MaxPool2D(pool_size=(2,2))(c3)
    c3 = Flatten()(c3)
    c3 = Dense(512,activation='relu')(c3)
    c3 = Dense(nb_classes,activation='softmax')(c3)
    model = Model(model_input,c3)
    return model



'''
结果展示 混淆矩阵 和 报告
'''
def confusion_matrix_plot_matplotlib(y_truth, y_predict,data_type,timer,title='混淆矩阵评价指标', cmap=plt.cm.RdPu):
    """Matplotlib绘制混淆矩阵图
    parameters
    ----------
        y_truth: 真实的y的值, 1d array
        y_predict: 预测的y的值, 1d array
        cmap: 画混淆矩阵图的配色风格, 使用cm.Blues，更多风格请参考官网
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix 
    from matplotlib.font_manager import FontProperties
    #import time
    #timer = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10) 

    
    cm = confusion_matrix(y_truth, y_predict)
    print(cm)
    plt.figure()#figsize=(12,6)
    plt.title(title,fontproperties=font)
    plt.imshow(cm, interpolation='nearest',cmap=cmap)  # 混淆矩阵图
    plt.colorbar()  # 颜色标签
 
    for x in range(len(cm)):  # 数据标签 文本注释
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center')
   
    target_names = ['state1','state2','state3','state4',
                    'state5','state6','state7','state8','state9','state10','state11','state12']
    tick_marks = np.arange(12)#12类  1 2 3 4 5 6 7 8 9 10 11 12
    plt.xticks(tick_marks, target_names, rotation=45,fontproperties=font)
    plt.yticks(tick_marks, target_names,fontproperties=font)
    plt.tight_layout()
    
#    plt.ylabel('True label')  # 坐标轴标签
#    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.ylabel('真实状态',fontproperties=font)
    plt.xlabel('感知状态',fontproperties=font)
    figname="bearing"+data_type+"混淆矩阵"+timer+".png"
    plt.savefig(figname,dpi=300)
    plt.show()  # 显示作图结果
    # Normalize the confusion matrix by row (i.e by the number of samples
                                         # in each class)
# =============================================================================
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     print(cm_normalized)
#     plt.figure()#figsize=(12,6)
#     plt.title(title,fontproperties=font)
#     plt.imshow(cm_normalized,cmap=cmap,interpolation='nearest')
#     plt.colorbar()  # 颜色标签
#     plt.xticks(tick_marks, target_names, rotation=45,fontproperties=font)
#     plt.yticks(tick_marks, target_names,fontproperties=font)
#     plt.tight_layout()
#     plt.ylabel('真实状态',fontproperties=font)
#     plt.xlabel('感知状态',fontproperties=font)
#     figname="bearing"+data_type+"混淆矩阵_归一化"+timer+".png"
#     plt.savefig(figname,dpi=300)
#     plt.show()  # 显示作图结果
# =============================================================================


def my_classification_report(y_true, y_pred,title):  
    """
    主要通过sklearn.metrics函数中的classification_report()函数，
    针对每个类别给出详细的准确率、召回率和F-值这三个参数和宏平均值，用来评价算法好坏。
    另外ROC曲线的话，需要是对二分类才可以。多类别似乎不行
    """
    from sklearn.metrics import classification_report  
    print (classification_report(y_true, y_pred,target_names = ['state1','state2','state3','state4',
                    'state5','state6','state7','state8','state9','state10','state11','state12']))
    report=classification_report(y_true, y_pred,target_names = ['state1','state2','state3','state4',
                    'state5','state6','state7','state8','state9','state10','state11','state12']) 
    report = title+'\n'+report+'\n'
    return report

# =============================================================================
# import time
# timer = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
# 
# pred_train = model.predict(train_x)
# 
# pred_valid = model.predict(valid_x)
# 
# pred_train_1d = np.argmax(pred_train,axis = 1)
# 
# pred_valid_1d = np.argmax(pred_valid,axis = 1)
# 
# true_train_1d = np.argmax(train_y,axis = 1)
# 
# true_valid_1d = np.argmax(valid_y,axis = 1)
# 
# 
# report_vaildation = my_classification_report(true_test_1d,pred_test_1d,"轴承测试数据报告指标")
# confusion_matrix_plot_matplotlib(true_train_1d,pred_train_1d,"训练数据",timer=timer)
# confusion_matrix_plot_matplotlib(true_valid_1d,pred_valid_1d,"验证数据",timer=timer)
# =============================================================================


def save(filename, contents): 
    """
    保存str类型的数据到文件中
    -------------参数说明-----------
    filename:要保存的文件名 例如 report1_1.txt
    contents:要保存的str类型变量内容
    """
    fh = open(filename, 'w') 
    fh.write(contents) 
    fh.close() 
    
    
import time
timer = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
'''
获取训练数据和测试数据训练数据,注意获取三种数据集的文件目录位置
train-root_dir1
test-root_dir2
valid-root_dir3
'''

root_dir1 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-0/train/'  #定义根目录
root_dir2 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-0/test/'  #定义根目录
root_dir3 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-0/valid/'  #定义根目录
train_x,train_y,valid_x,valid_y = get_data_shuffle(root_dir1,root_dir2,root_dir3)



root_dir4 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-0/train/'  #定义根目录
root_dir5 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-0/test/'  #定义根目录
root_dir6 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-0/valid/'  #定义根目录
test_x,test_y  = get_test_data_shuffle(root_dir4,root_dir5,root_dir6)

input_shape = train_x[0,:,:,:].shape
model_input = Input(input_shape)
model = creat_model(model_input)


for i in range(8):
    
    #configuring
    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=1e-4),metrics=['acc'])
    history=model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x,valid_y))
    pred_test  =model.predict(test_x)
    pred_test_1d = np.argmax(pred_test,axis = 1)
    true_test_1d = np.argmax(test_y,axis = 1)
    confusion_matrix_plot_matplotlib(true_test_1d,pred_test_1d,"测试数据",timer=timer+str(i))
#save_figure(history) 
#model.save('E:/黎国强资料/西储大学-CNN/model/1-10.h5')
import time
timer = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))   
'''
获取训练数据和测试数据训练数据,注意获取三种数据集的文件目录位置
train-root_dir1
test-root_dir2
valid-root_dir3
'''

root_dir1 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-(-4)/train/'  #定义根目录
root_dir2 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-(-4)/test/'  #定义根目录
root_dir3 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-(-4)/valid/'  #定义根目录
train_x,train_y,valid_x,valid_y = get_data_shuffle(root_dir1,root_dir2,root_dir3)



root_dir4 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-(-4)/train/'  #定义根目录
root_dir5 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-(-4)/test/'  #定义根目录
root_dir6 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-(-4)/valid/'  #定义根目录
test_x,test_y  = get_test_data_shuffle(root_dir4,root_dir5,root_dir6)

input_shape = train_x[0,:,:,:].shape
model_input = Input(input_shape)
model = creat_model(model_input)


for i in range(8):
    
    #configuring
    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=1e-4),metrics=['acc'])
    history=model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x,valid_y))
    pred_test  =model.predict(test_x)
    pred_test_1d = np.argmax(pred_test,axis = 1)
    true_test_1d = np.argmax(test_y,axis = 1)
    confusion_matrix_plot_matplotlib(true_test_1d,pred_test_1d,"测试数据",timer=timer+str(i))
#save_figure(history) 
#model.save('E:/黎国强资料/西储大学-CNN/model/1-10.h5')  
import time
timer = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))    
'''
获取训练数据和测试数据训练数据,注意获取三种数据集的文件目录位置
train-root_dir1
test-root_dir2
valid-root_dir3
'''

root_dir1 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-4/train/'  #定义根目录
root_dir2 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-4/test/'  #定义根目录
root_dir3 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-4/valid/'  #定义根目录
train_x,train_y,valid_x,valid_y = get_data_shuffle(root_dir1,root_dir2,root_dir3)



root_dir4 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-4/train/'  #定义根目录
root_dir5 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-4/test/'  #定义根目录
root_dir6 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-4/valid/'  #定义根目录
test_x,test_y  = get_test_data_shuffle(root_dir4,root_dir5,root_dir6)

input_shape = train_x[0,:,:,:].shape
model_input = Input(input_shape)
model = creat_model(model_input)


for i in range(10):
    
    #configuring
    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=1e-4),metrics=['acc'])
    history=model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x,valid_y))
    pred_test  =model.predict(test_x)
    pred_test_1d = np.argmax(pred_test,axis = 1)
    true_test_1d = np.argmax(test_y,axis = 1)
    confusion_matrix_plot_matplotlib(true_test_1d,pred_test_1d,"测试数据",timer=timer+str(i))
#save_figure(history) 
#model.save('E:/黎国强资料/西储大学-CNN/model/1-10.h5')  
    
import time
timer = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))    
'''
获取训练数据和测试数据训练数据,注意获取三种数据集的文件目录位置
train-root_dir1
test-root_dir2
valid-root_dir3
'''

root_dir1 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-8/train/'  #定义根目录
root_dir2 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-8/test/'  #定义根目录
root_dir3 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-8/valid/'  #定义根目录
train_x,train_y,valid_x,valid_y = get_data_shuffle(root_dir1,root_dir2,root_dir3)



root_dir4 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-8/train/'  #定义根目录
root_dir5 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-8/test/'  #定义根目录
root_dir6 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-8/valid/'  #定义根目录
test_x,test_y  = get_test_data_shuffle(root_dir4,root_dir5,root_dir6)

input_shape = train_x[0,:,:,:].shape
model_input = Input(input_shape)
model = creat_model(model_input)


for i in range(10):
    
    #configuring
    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=1e-4),metrics=['acc'])
    history=model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x,valid_y))
    pred_test  =model.predict(test_x)
    pred_test_1d = np.argmax(pred_test,axis = 1)
    true_test_1d = np.argmax(test_y,axis = 1)
    confusion_matrix_plot_matplotlib(true_test_1d,pred_test_1d,"测试数据",timer=timer+str(i))
#save_figure(history) 
#model.save('E:/黎国强资料/西储大学-CNN/model/1-10.h5')  

import time
timer = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
'''
获取训练数据和测试数据训练数据,注意获取三种数据集的文件目录位置
train-root_dir1
test-root_dir2
valid-root_dir3
'''

root_dir1 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-10/train/'  #定义根目录
root_dir2 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-10/test/'  #定义根目录
root_dir3 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_train)-10/valid/'  #定义根目录
train_x,train_y,valid_x,valid_y = get_data_shuffle(root_dir1,root_dir2,root_dir3)



root_dir4 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-10/train/'  #定义根目录
root_dir5 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-10/test/'  #定义根目录
root_dir6 = 'E:/黎国强资料/西储大学-CNN/train_valid_data(add_noise_to_test)-10/valid/'  #定义根目录
test_x,test_y  = get_test_data_shuffle(root_dir4,root_dir5,root_dir6)

input_shape = train_x[0,:,:,:].shape
model_input = Input(input_shape)
model = creat_model(model_input)


for i in range(10):
    
    #configuring
    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=1e-4),metrics=['acc'])
    history=model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x,valid_y))
    pred_test  =model.predict(test_x)
    pred_test_1d = np.argmax(pred_test,axis = 1)
    true_test_1d = np.argmax(test_y,axis = 1)
    confusion_matrix_plot_matplotlib(true_test_1d,pred_test_1d,"测试数据",timer=timer+str(i))
#save_figure(history) 
#model.save('E:/黎国强资料/西储大学-CNN/model/1-10.h5')  










