from matplotlib import pyplot as plt

y=[0,1,2,3,4,5,6,7,8,9]
x=[18,19,20,21,22,23,24,25,26,27]

fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
for ncol in axs:
    for nrow in ncol:
        nrow.scatter(x,y)
plt.show()