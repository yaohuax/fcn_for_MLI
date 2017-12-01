import os
with open("train.txt", 'w') as f:
	for i in range(1,405):
		str1 = "Raw"+ str(i) +".jpg" + " " + "label" + str(i) + ".png"
		f.write(str1)
		f.write('\n')