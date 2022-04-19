import matplotlib.pyplot as plt

str_4plot = open("p1b_rv.out", "r").read()

cores = []
time = []

for i in str_4plot.split('\n'):
    time_core = i.split(' cores: ')
    if time_core != ['']:
        cores.append(int(time_core[0][-2:]))
        time.append(float(time_core[1]))

fig = plt.figure()
ax = plt.axes()

ax.plot(cores, time)
plt.xlabel('cores')
plt.title('Seconds it took to run per number of cores used')
plt.savefig("seconds_per_core.png")