import math
import h5py
import numpy as np
import tensorflow as tf
from skimage.transform import resize

def getCubes(img, msk, cube_size):
    # img and mask in [x,y,z]
    # cube_size in [x,y,z]
  sizeX = img.shape[0]
  sizeY = img.shape[1]
  sizeZ = img.shape[2]

  cubeSizeX = cube_size[0]
  cubeSizeY = cube_size[1]
  cubeSizeZ = cube_size[2]

  n_z = int(math.ceil(float(sizeZ)/cubeSizeZ))
  n_x = int(math.ceil(float(sizeX)/cubeSizeX))
  n_y = int(math.ceil(float(sizeY)/cubeSizeY))

  sizeNew = [n_x*cubeSizeX, n_y*cubeSizeY, n_z*cubeSizeZ]

  imgNew = np.zeros(sizeNew, dtype=np.float16)
  imgNew[0:sizeX, 0:sizeY, 0:sizeZ] = img

  mskNew = np.zeros(sizeNew, dtype=np.int)
  mskNew[0:sizeX, 0:sizeY, 0:sizeZ] = msk

  n_ges = n_x * n_y * n_z
  n_4 = int(math.ceil(float(n_ges)/4.)*4)

  imgCubes = np.zeros((n_4, cubeSizeX, cubeSizeY, cubeSizeZ)) - 1  # -1 = air
  mskCubes = np.zeros((n_4, cubeSizeX, cubeSizeY, cubeSizeZ))

  count = 0
  for z in range(n_z):
    for y in range(n_y):
      for x in range(n_x):
        imgCubes[count] = imgNew[x*cubeSizeX:(x+1)*cubeSizeX, y*cubeSizeY:(y+1)*cubeSizeY,z*cubeSizeZ:(z+1)*cubeSizeZ]
        mskCubes[count] = mskNew[x*cubeSizeX:(x+1)*cubeSizeX, y*cubeSizeY:(y+1)*cubeSizeY,z*cubeSizeZ:(z+1)*cubeSizeZ]
        count += 1

  return imgCubes, mskCubes

def preproc(img_list):
    img_arr = np.concatenate(img_list, axis=0)
    img_arr = tf.expand_dims(img_arr, axis=4)
    img_arr = tf.cast(img_arr, tf.float32)
    return img_arr


def processing(file, cube_size, batch_size):
    ct, lesion = [], []
    datafile = h5py.File(file, 'r')
    patient_ids = list(datafile['ct'].keys())

    for patient_id in patient_ids:
        img, mask = datafile['ct'][patient_id][:], datafile['lesion'][patient_id][:]
        img = np.transpose(img, (1,2,0))
        mask = np.transpose(mask, (1,2,0))
        imgCubes, mskCubes = getCubes(img, mask, cube_size)
        ct.append(imgCubes)
        lesion.append(mskCubes)

    datafile.close()
    
    ct_arr = preproc(ct)
    lesion_arr = preproc(lesion)

    data_loader = tf.data.Dataset.from_tensor_slices((ct_arr, lesion_arr))
    dataset = (data_loader.shuffle(len(ct_arr)).batch(batch_size).prefetch(2))

    return dataset