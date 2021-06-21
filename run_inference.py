from cacseg_model import *
# from load_data import *
# from data_generator import DataGenerator
from tensor_slices import processing


def run_inference(train_data, val_data = None, cube_size = [64,64,16], batch_size = 1, 
                  model_weights = None, mgpu = 0, model_file = 'cacseg_weights.hdf5'):
  
  print("\nDeep Learning model inference:")

  verbose = 2
  pool_size = (2, 2, 2)
  conv_size = (3, 3, 3)
  input_shape = (cube_size[0], cube_size[1], cube_size[2], 1)

  optimizer = 'ADAM'
  extended = True
  drop_out = 0.5
  lr = 0.001
  lr_drop = 0.7
  drop_epochs = 100
  num_epochs = 500
  
  # number of the model downsampling steps 
  down_steps = 3  

  model = getUnet3d(down_steps = down_steps, input_shape = input_shape, pool_size = pool_size,
                                 conv_size = conv_size, initial_learning_rate = lr, mgpu = mgpu,
                                 extended = extended, drop_out = drop_out, optimizer = optimizer)


  if model_weights:
    model.load_weights(model_weights)

  # train_data = load_data(train_data)
  # x_train, y_train = train_data['ct'], train_data['lesion']
  # x_train_gen = customImageDataGenerator()
  # y_train_gen = customImageDataGenerator()
  # x_train_datagen = x_train_gen.flow(x_train, batch_size=batch_size, seed=seed)
  # y_train_datagen = y_train_gen.flow(y_train, batch_size=batch_size, seed=seed)
  # train_datagen = zip(x_train_datagen, y_train_datagen)
  # train_datagen = DataGenerator(x=x_train, y=y_train, batch_size=batch_size, dim=(512,512,8,1), shuffle=True)
  train_datagen = processing(train_data,cube_size,batch_size)

  if val_data:
    # val_data = load_data(val_data)
    # x_val, y_val = val_data['ct'], val_data['lesion']
    # x_val_gen = customImageDataGenerator()
    # y_val_gen = customImageDataGenerator()
    # x_val_datagen = x_val_gen.flow(x_val, batch_size=batch_size, seed=seed)
    # y_val_datagen = y_val_gen.flow(y_val, batch_size=batch_size, seed=seed)
    # val_datagen = zip(x_val_datagen, y_val_datagen)
    # val_datagen = DataGenerator(x=x_val,y=y_val, batch_size=batch_size, dim=(512,512,8,1), shuffle=False)
    val_datagen = processing(val_data,cube_size,batch_size)

    model.fit(train_datagen, epochs = num_epochs, shuffle = True, verbose = verbose,
              callbacks = get_callbacks(model_file, lr, lr_drop, drop_epochs),
              validation_data = val_datagen, initial_epoch = 0)
  else:
    model.fit(train_datagen, epochs = num_epochs, verbose = verbose, 
              callbacks = get_callbacks(model_file, lr, lr_drop, drop_epochs), 
              initial_epoch = 0)
