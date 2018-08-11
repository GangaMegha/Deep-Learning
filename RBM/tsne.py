import numpy as np
# from skdata.mnist.views import OfficialImageClassification
from matplotlib import pyplot as plt
from tsne import bh_sne
import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
n=10000
y_data = np.loadtxt('data_bin/train_labels.csv')[:10000]
models=['cd_lr_0.0001_k_1_n_128.csv','cd_lr_1e-05_k_1_n_128.csv','cd_lr_0.0001_k_2_n_128.csv','cd_lr_1e-05_k_2_n_128.csv',
'cd_lr_0.001_k_1_n_128.csv','cd_lr_1e-06_k_1_n_128.csv','cd_lr_0.001_k_2_n_128.csv','cd_lr_1e-06_k_2_n_128.csv',
'cd_lr_0.001_k_4_n_128.csv','cd_lr_0.001_k_8_n_128.csv']

for model in models:
  plt.figure(0)
  plt.clf()
  x_data = np.loadtxt('hidden_rep/'+model,delimiter =',')

  # load up data
  # data = OfficialImageClassification(x_dtype="float32")
  # x_data = data.all_images
  # y_data = data.all_labels
  palette = np.array(sns.color_palette("hls", 10))

  # convert image data to float64 matrix. float64 is need for bh_sne
  x_data = np.asarray(x_data).astype('float64')
  x_data = x_data.reshape((x_data.shape[0], -1))

  # For speed of computation, only run on a subset
  # n = 20000
  x_data = x_data[:n]
  y_data = y_data[:n]

  # perform t-SNE embedding
  vis_data = bh_sne(x_data)

  # plot the result
  vis_x = vis_data[:, 0]
  vis_y = vis_data[:, 1]


  plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
  plt.colorbar(ticks=range(10))
  # plt.clim(-0.5, 9.5)


  # txts = []
  # for i in range(10):
  #     # Position of each label.
  #     xtext, ytext = np.median(vis_data[y_data == i, :2], axis=0)
  #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
  #     txt.set_path_effects([
  #         PathEffects.Stroke(linewidth=5, foreground="w"),
  #         PathEffects.Normal()])

# Optionally add a colorbar
  # cax, _ = plt.colorbar.make_axes(ax)
  # cbar = plt.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize,ticks=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot' ])
  # plt.colorbar(sc,ticks=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot' ])
  # plt.clim(-0.5, 9.5)
  plt.title('TSNE plot for '+model[:-4])
  # plt.show()
  plt.savefig('plots/'+model[:-3]+'png', dpi=120)
