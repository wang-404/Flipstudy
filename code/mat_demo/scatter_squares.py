import matplotlib.pyplot as plt
#散点图

#绘制一系列点
x_values=list(range(1,1001))
y_values=[x**2 for x in  x_values]
#删除数据点轮廓 edgecolor='none  c=‘red'可修改数据点的颜色
#rgb颜色 传递颜色c=(0,0,0.8)设置一个元组 包含三个0到1的小数值分别表示红、绿、蓝 接近0颜色深
#颜色映射 颜色渐变  根据y来设置颜色c=y_values,cmap=plt.cm.Blues,
plt.scatter(x_values,y_values,c=y_values,cmap=plt.cm.Blues,edgecolors='none',s=40)
#绘制标题并给坐标轴加标签
plt.title("Square Number",fontsize=24)
plt.xlabel("Value",fontsize=14)
plt.ylabel("Square of Value",fontsize=14)
#设置刻度标记大小
plt.tick_params(axis='both',which='major',labelsize=14)
#绘制每个坐标轴取值范围
plt.axis([0,1100,0,1100000])
#plt.show()
#自动保存列表  第一个参数文件名  第二个参数将多余的空白的区域裁掉可省略
plt.savefig("F:/squares_plot.png",bbox_inches='tight')
