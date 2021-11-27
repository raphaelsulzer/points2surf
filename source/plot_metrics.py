import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

data_path = "/mnt/raphael/ModelNet10_out/p2s/"

# methods =["sensor_2_2", "mean_neighborhood_one", "mean_neighborhood_uniform_4", "scan41", "scan421", "scan421_normal"]
## normals
# methods = ["conventional_plane", "conventional_grid",
#             "scan421", "scan421_normal",
#            "conventional_plane_normal", "conventional_plane_est_normal",
#            "sensor_vec_norm",
#            "conventional_plane_est_normal_orient_mst"]

## best

a = 0
b = 100000000

methods = ["conventional2", "sensor_vec_norm2", "sensor_vec_norm3"]

plt.figure("Loss")
plt.figure("IoU")

colors = ["r","g","b"]

for i,m in enumerate(methods):

    file = os.path.join(data_path, m, 'metrics',"results.csv")
    df = pd.read_csv(file, sep=',', header=0)
    loss_cl = df["train_loss_cl"].values
    loss_reg = df["train_loss_reg"].values
    loss_total = df["train_loss_total"].values
    iou = df["test_current_iou"].values

    # its = np.arange(start=10000,stop=len(loss_total),step=10)
    # loss_total = loss_total[its]
    # loss_reg = loss_reg[its]
    # loss_cl = loss_cl[its]
    # iou = iou[its]
    its = df["iteration"].values
    a = 100
    its = its[a:]
    loss_total = loss_total[a:]
    loss_reg = loss_reg[a:]
    loss_cl = loss_cl[a:]
    iou = iou[a:]

    plt.figure("Loss")
    plt.plot(its,loss_cl,':',color=colors[i])
    plt.plot(its,loss_reg,'--',color=colors[i])
    plt.plot(its,loss_total,'-',color=colors[i])
    plt.figure("IoU")
    plt.plot(its,iou,'-',color=colors[i])
    # plt.plot(df.values[int(a/1000):int(b/1000),0], df.values[int(a/1000):int(b/1000),1], '-')

plt.figure("Loss")
plt.grid()
l = ["loss_sign","loss_dist","loss_total"]
legend = ["conventional_"+l[0],"conventional_"+l[1],"conventional_"+l[2],
          "sensor_vec_"+l[0],"sensor_vec_"+l[1],"sensor_vec_"+l[2],
          "sensor_vec2_"+l[0],"sensor_vec2_"+l[1],"sensor_vec2_"+l[2]]
plt.legend(legend)
plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.savefig(os.path.join(data_path, 'train_loss.png'),dpi=200)

plt.figure("IoU")
plt.grid()
plt.legend(methods)
plt.xlabel("Training Iterations")
plt.ylabel("Validation IoU")
plt.savefig(os.path.join(data_path, 'validation_iou.png'),dpi=200)