import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

filter_funtions = {
  "cricket-semantic": lambda name: "__" not in name
}

def get_img_list(imgs_path, dataset_name):
  filter_func = filter_funtions[dataset_name]
  return list(filter(filter_func, os.listdir(imgs_path)))

def img_sizes(imgs_path, dataset_name):
  imgs = get_img_list(imgs_path, dataset_name)
  sizes = []
  for img_name in imgs:
    img = Image.open(f"{imgs_path}/{img_name}")
    size = str((img.width, img.height))
    size_available_list = list(map(lambda size_dict: size_dict["size"]==size, sizes))
    current_size_idx = size_available_list.index(True) if True in size_available_list else -1
    if current_size_idx == -1:
      sizes.append({"size":size, "count":1})
    else:
      sizes[current_size_idx]["count"] += 1
  return sizes

class ImageMemoryObj:
  def __init__(self, imgs_path, img_nm_lst, img_grid_disp_n):
    self.imgs_path=imgs_path
    self.img_nm_lst=img_nm_lst
    self.img_grid_disp_n=img_grid_disp_n
    window_size = img_grid_disp_n**2
    prev_start, start, end, next_end = 0, 0, window_size, window_size*2
    self.window_size = window_size
    self.prev_start = prev_start
    self.start = start
    self.end = end
    self.next_end = next_end
    self.prev_grid = self.get_grid_img(img_nm_lst[prev_start:start], imgs_path, img_grid_disp_n)
    self.curt_grid = self.get_grid_img(img_nm_lst[start:end], imgs_path, img_grid_disp_n)
    self.next_grid = self.get_grid_img(img_nm_lst[end:next_end], imgs_path, img_grid_disp_n)

  def get_grid_img(self, disp_nm_lst, imgs_path, img_grid_disp_n):
    disp_path_lst = list(map(lambda nm: f"{imgs_path}/{nm}", disp_nm_lst))
    fig, ax = plt.subplots(img_grid_disp_n, img_grid_disp_n, figsize=(8,5))

    disp_img_lst = list(map(lambda path: plt.imread(path), disp_path_lst))

    for i, img, img_nm in zip(range(len(disp_path_lst)), disp_img_lst, disp_nm_lst):
      row, col = int(i//img_grid_disp_n), int(i%img_grid_disp_n)
      ax[row][col].imshow(img)
      ax[row][col].set_title(img_nm)
      ax[row][col].set_xticks([]), ax[row][col].set_yticks([])

    plt.close()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    img = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    return img

  def next(self):
    self.start = self.end
    self.end = self.end + self.window_size
    self.next_end = self.end + self.window_size
    self.prev_start = self.start - self.window_size
    self.prev_grid = self.curt_grid
    self.curt_grid = self.next_grid

  def load_next(self):
    self.next_grid = self.get_grid_img(self.img_nm_lst[self.end:self.next_end], self.imgs_path, self.img_grid_disp_n)

  def prev(self):
    self.end = self.start
    self.start = self.start - self.window_size
    self.next_end = self.end + self.window_size
    self.prev_start = self.start - self.window_size
    self.next_grid = self.curt_grid
    self.curt_grid = self.prev_grid

  def load_prev(self):
    self.prev_grid = self.get_grid_img(self.img_nm_lst[self.prev_start:self.start], self.imgs_path, self.img_grid_disp_n)

  def show(self):
    fig, ax = plt.subplots(facecolor = "gray")
    ax.imshow(self.curt_grid)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def img2ColorMat(img):
  shape = img.shape
  b = np.array(list(map(lambda num: hex(num)[2:].rjust(2, "0"), img[:,:,0].reshape(-1))))
  g = np.array(list(map(lambda num: hex(num)[2:].rjust(2, "0"), img[:,:,1].reshape(-1))))
  r = np.array(list(map(lambda num: hex(num)[2:].rjust(2, "0"), img[:,:,2].reshape(-1))))
  colorMat = np.array(list(map(lambda c: "#" + c, list(np.char.add(np.char.add(r, g), b)))))
  colorMat = colorMat.reshape(shape[:-1])
  return colorMat
  
def findColors(img):
  b = np.array(list(map(lambda num: hex(num)[2:].rjust(2, "0"), img[:,:,0].reshape(-1))))
  g = np.array(list(map(lambda num: hex(num)[2:].rjust(2, "0"), img[:,:,1].reshape(-1))))
  r = np.array(list(map(lambda num: hex(num)[2:].rjust(2, "0"), img[:,:,2].reshape(-1))))
  colors = set(map(lambda c: "#" + c, set(np.char.add(np.char.add(r, g), b))))
  return colors