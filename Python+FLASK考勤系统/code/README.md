_photo文件夹：
    用于调试

cap_frame文件夹：
    用于调试

conf文件夹：
    seting.conf:放的是百度API的cookie需要的ID和secret

face_feature文件夹：
    放置的是我们对于每个图片提取出来的特征

method文件夹：
    放置我们检测的main_solve主要的检测类

picture文件夹：
    放置的是原始的图片底库

templates文件夹：
    index.html：前端代码


util文件夹：
    BaiduApiUtil.py：百度API的相关调用


预处理文件夹：
    rename.py:
        将原来的图片底库中的图片重命名（为了解决中文路径问题）

    feature.py:
        将图片特征提取出来，方便之后的特征比对部分（特征存储在face_feature文件夹中）


app.py:
    运行文件：Python app.py即可进入网页



