from die import Die
import pygal

#三个骰子
die_1 = Die()
die_2 = Die()
die_3 = Die()
#掷骰子解果
results = []
for num_roll in range(50000):
    result = die_1.roll() + die_2.roll() + die_3.roll()
    results.append(result)

#分析结果
frequencies = []
max_value = die_1.num_sides + die_2.num_sides + die_3.num_sides
for value in range(3,max_value+1):
    frequency = results.count(value)
    frequencies.append(frequency)

#可视化结果
hist = pygal.Bar()
hist.title = "Result of rolling three D6 50000 times"
hist.x_labels = []
for value in range(3,max_value+1):
    hist.x_labels.append(value)
hist.x_title = "Result"
hist.y_title = "Frequency of Result"

hist.add('D6+D6+D6',frequencies)
hist.render_to_file("F:/three.svg")