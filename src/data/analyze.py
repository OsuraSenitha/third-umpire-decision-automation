import os
from PIL import Image

def img_sizes(imgs_path, filterFunc):
  imgs = list(filter(filterFunc, os.listdir(imgs_path)))
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