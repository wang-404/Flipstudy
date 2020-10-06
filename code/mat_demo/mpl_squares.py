import matplotlib.pyplot as plt
"""简单的折线图 平方数序列1、 4、 9、 16和25的折线图"""
#用于校正图形
input_values=[1,2,3,4,5]
squares=[1,4,9,16,25]
plt.plot(input_values,squares,linewidth=5)

#设置图标坐标，并给坐标轴加标签
plt.title("Square Numbers",fontsize=24)
plt.xlabel("value",fontsize=14)
plt.ylabel("Square of value",fontsize=14)

#设置刻度标记的大小
plt.tick_params(axis='both',labelsize=14)

plt.show()