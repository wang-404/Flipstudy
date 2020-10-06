import matplotlib.pyplot as plt
from random_walk import RandomWalk
"""
#创建一个RandomWalk实例 将包含的点都打印出来
rw = RandomWalk()
rw.fill_walk()
plt.scatter(rw.x_values,rw.y_values,s=15)
plt.show()  """

#只要程序处于活动中，就不断模拟随机漫步
while True:
    #创建一个RandomWalk实例 并将其包含的点都绘制出来
    rw = RandomWalk(50000)
    rw.fill_walk()
    #使用颜色映射指出各点的先后顺序 删除轮廓 更加明显
    #传递参数c 设置为一个列表 包含各个点的先后顺序  各点顺序绘制列表只要包含1-5000
    point_number = list(range(rw.num_points))
    plt.scatter(rw.x_values,rw.y_values,c=point_number,cmap=plt.cm.Blues,edgecolors='none',s=1)
    #突出起点终点
    plt.scatter(0,0,c='green',edgecolors='none',s=100)
    plt.scatter(rw.x_values[-1],rw.y_values[-1],c='red',edgecolors='none',s=100)
    #隐藏坐标轴 plt.axes(). 可能因版本升级出现bug  可用plt.gca.替代
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    #设置绘图窗口的尺寸 分辨率 dpi==128
    plt.figure(figsize=(10,6),dpi=128)
    plt.show()

    keep_running = input("make another walk?(y/n)")
    if keep_running == 'n':
        break