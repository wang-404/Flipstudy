from matplotlib import pyplot as plt
#x_number = [1,2,3,4,5]
x_number = list(range(1,5001))
y_number = [x**3 for x in x_number]
#edgecolor边框 c y渐变camp颜色
plt.scatter(x_number,y_number,edgecolors='none',c=y_number,cmap=plt.cm.Blues)

#坐标 标题
plt.title("Cube Number",fontsize=15)
plt.xlabel("value",fontsize=12)
plt.ylabel("Cube of value",fontsize=12)
#标记刻度大小
plt.tick_params(axis='both',which='major',labelsize=14)

plt.show()