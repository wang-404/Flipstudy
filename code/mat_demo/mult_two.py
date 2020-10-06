from die import  Die
import pygal

die_1 = Die()
die_2 = Die()

results = []
for num_roll in range(5000) :
    result = die_1.roll() * die_2.roll()
    results.append(result)

frequencies = []
max_value = die_1.num_sides * die_2.num_sides
for value in range(1,max_value+1):
    frequency = results.count(value)
    frequencies.append(frequency)

hist = pygal.Bar()
hist.title = "Result of mult two D6 5000 times"
hist.x_labels = []
for value in range(1,max_value+1):
    hist.x_labels.append(value)
hist.x_title = "Result"
hist.y_title = "Frequency of Result"

hist.add("D6*D6",frequencies)
hist.render_to_file("F:/mult.svg")