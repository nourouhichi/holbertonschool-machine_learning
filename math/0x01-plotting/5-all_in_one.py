#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

grid= (3, 2)
fig = plt.figure(figsize=(10,10))
fig.suptitle('All in One')

ax_0 = plt.subplot2grid(grid, (0,0))
ax_0.plot(y0, color='red')
ax_0.set_xlim([0, 10])


ax_1 = plt.subplot2grid(grid, (0,1))
ax_1.plot(x2, y2)
ax_1.set_yscale("log")
ax_1.set_xlim([0,28650])
ax_1.set_title("Exponential Decay of C-14", fontsize='x-small')
ax_1.set_xlabel("Time (years)", fontsize='x-small')
ax_1.set_ylabel("Fraction Remaining",fontsize='x-small')

ax_2 = plt.subplot2grid(grid, (1,0))
ax_2.scatter(x1, y1, c="magenta")
ax_2.set_xlabel("Height (in)")
ax_2.set_ylabel("Weight (lbs)")
ax_2.set_title("Men's Height vs Weight", fontsize='x-small')

ax_3 = plt.subplot2grid(grid, (1,1))
ax_3.plot(x3,y31, "--",label="C-14", color="red")
ax_3.plot(x3,y32, label="Ra-226", color="green")
leg = ax_3.legend(loc='upper right')
ax_3.set_xlim([0, 20000])
ax_3.set_ylim([0,1])
ax_3.set_xlabel("Time (years)", fontsize='x-small')
ax_3.set_ylabel("Fraction Remaining", fontsize='x-small')
ax_3.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')

ax_4 = plt.subplot2grid(grid, (2,0), colspan=2)
ax_4.hist(student_grades, bins=[i for i in range(0, 101, 10)], edgecolor="black")
ax_4.set_ylim([0,30])
ax_4.set_xlim([0,100])
ax_4.set_xlabel("Grades", fontsize='x-small')
ax_4.set_ylabel("Number of Students", fontsize='x-small')
ax_4.set_title("Project A", fontsize='x-small')

fig.tight_layout()
plt.show()