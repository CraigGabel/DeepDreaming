# from here:
# https://www.tensorflow.org/tutorials/generative/deepdream

import time
startTime = time.time()

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config....)

# import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(0.75)
# tf.config.gpu.set_per_process_memory_growth(True)

# import tensorflow as tf
# if (tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)):
#   gpus = tf.config.experimental.list_physical_devices('GPU')
#   tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*2)])

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np

import matplotlib as mpl

import IPython.display as display
import PIL.Image

from tensorflow.keras.preprocessing import image

def get(path, max_dim=None):
  img = PIL.Image.open(path)
  if max_dim:
    img.thumbnail((max_dim, max_dim))
  return np.array(img)

# Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

# Display an image
def show(img, save_name):
  from matplotlib import pyplot as plt
  plt.imshow(img, interpolation='nearest')
  plt.savefig(save_name)
  plt.show()
  # display.display(PIL.Image.fromarray(np.array(img)))
  # img = PIL.Image.fromarray(img, 'RGB')
  #img.save('my.png')
  # img.show()


def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.reduce_sum(losses)

# class DeepDream(tf.Module):
#   def __init__(self, model):
#     self.model = model

#   @tf.function(
#       input_signature=(
#         tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
#         tf.TensorSpec(shape=[], dtype=tf.int32),
#         tf.TensorSpec(shape=[], dtype=tf.float32),)
#   )
#   def __call__(self, img, steps, step_size):
#       print("Tracing")
#       loss = tf.constant(0.0)
#       for n in tf.range(steps):
#         with tf.GradientTape() as tape:
#           # This needs gradients relative to `img`
#           # `GradientTape` only watches `tf.Variable`s by default
#           tape.watch(img)
#           loss = calc_loss(img, self.model)

#         # Calculate the gradient of the loss with respect to the pixels of the input image.
#         gradients = tape.gradient(loss, img)

#         # Normalize the gradients.
#         gradients /= tf.math.reduce_std(gradients) + 1e-8

#         # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
#         # You can update the image by directly adding the gradients (because they're the same shape!)
#         img = img + gradients*step_size
#         img = tf.clip_by_value(img, -1, 1)

#       return loss, img

##########################################
# deepdream = DeepDream(dream_model)
# show(deepdream)

# print(type(deepdream))

###########################################
# def run_deep_dream_simple(img, steps=100, step_size=0.01):
#   # Convert from uint8 to the range expected by the model.
#   img = tf.keras.applications.inception_v3.preprocess_input(img)
#   img = tf.convert_to_tensor(img)
#   step_size = tf.convert_to_tensor(step_size)
#   steps_remaining = steps
#   step = 0
#   while steps_remaining:
#     if steps_remaining>100:
#       run_steps = tf.constant(100)
#     else:
#       run_steps = tf.constant(steps_remaining)
#     steps_remaining -= run_steps
#     step += run_steps

#     loss, img = deepdream(img, run_steps, tf.constant(step_size))

#     display.clear_output(wait=True)
#     # show(deprocess(img))
#     print ("Step {}, loss {}".format(step, loss))


#   result = deprocess(img)
#   display.clear_output(wait=True)
#   # show(result)

#   return result

# dream_img = run_deep_dream_simple(img=original_img,
#                                   steps=100, step_size=0.01)

# show(dream_img)

# ##############################################

# import time
# start = time.time()

# OCTAVE_SCALE = 1.30

# img = tf.constant(np.array(original_img))
# base_shape = tf.shape(img)[:-1]
# float_base_shape = tf.cast(base_shape, tf.float32)

# for n in range(-2, 3):
#   new_shape = tf.cast(float_base_shape*(OCTAVE_SCALE**n), tf.int32)

#   img = tf.image.resize(img, new_shape).numpy()

#   img = run_deep_dream_simple(img=img, steps=50, step_size=0.01)

# display.clear_output(wait=True)
# img = tf.image.resize(img, base_shape)
# img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
# show(img)

# end = time.time()
# end-start

# ################################################3
def random_roll(img, maxroll):
  # Randomly shift the image to avoid tiled boundaries.
  shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
  shift_down, shift_right = shift[0],shift[1]
  img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)
  return shift_down, shift_right, img_rolled

# shift_down, shift_right, img_rolled = random_roll(np.array(original_img), 512)
# show(img_rolled)



class TiledGradients(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),)
  )
  def __call__(self, img, tile_size=512):
    shift_down, shift_right, img_rolled = random_roll(img, tile_size)

    # Initialize the image gradients to zero.
    gradients = tf.zeros_like(img_rolled)

    # Skip the last tile, unless there's only one tile.
    xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
    if not tf.cast(len(xs), bool):
      xs = tf.constant([0])
    ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
    if not tf.cast(len(ys), bool):
      ys = tf.constant([0])

    for x in xs:
      for y in ys:
        # Calculate the gradients for this tile.
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img_rolled`.
          # `GradientTape` only watches `tf.Variable`s by default.
          tape.watch(img_rolled)

          # Extract a tile out of the image.
          img_tile = img_rolled[x:x+tile_size, y:y+tile_size]
          loss = calc_loss(img_tile, self.model)

        # Update the image gradients for this tile.
        gradients = gradients + tape.gradient(loss, img_rolled)

    # Undo the random shift applied to the image and its gradients.
    gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)

    # Normalize the gradients.
    gradients /= tf.math.reduce_std(gradients) + 1e-8

    return gradients

# get_tiled_gradients = TiledGradients(dream_model)

# def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01,
#                                 octaves=range(-2,3), octave_scale=1.3):
def run_deep_dream_with_octaves(img,
                                steps_per_octave,
                                step_size,
                                octaves,
                                octave_scale,
                                img_gradients):
  base_shape = tf.shape(img)
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = tf.keras.applications.inception_v3.preprocess_input(img)

  initial_shape = img.shape[:-1]
  img = tf.image.resize(img, initial_shape)
  for octave in octaves:
    # Scale the image based on the octave
    new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)
    img = tf.image.resize(img, tf.cast(new_size, tf.int32))

    for step in range(steps_per_octave):
      gradients = img_gradients(img)
      img = img + gradients*step_size
      img = tf.clip_by_value(img, -1, 1)

      if step % 10 == 0:
        display.clear_output(wait=True)
        # show(deprocess(img))
        print ("Octave {}, Step {}".format(octave, step))

  result = deprocess(img)
  return result

def main():
  # Downsizing the image makes it easier to work with.
  # filename = 'dog_orig.jpg'
  # filename = 'sportscar_car_sports.jpg'
  # filename = 'bird_bird_of_prey_raptor.jpg'
  # filename = 'shapes.jpg'
  # filename = 'butterfly.jpg'
  filename = 'butterfly.jpg'

  original_img = get(filename, max_dim=500)

  # show(original_img)
  # display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))


  my_steps_per_octave_l=(100,)
  my_step_size_l=(0.01,)
  my_octaves_l=(range(-3, 4),)
  my_octave_scale_l=(0.5,
                     0.55,
                     0.6,
                     0.65,
                     0.7,
                     0.75,
                     0.8,
                     0.85,
                     0.9,
                     0.95,
                     1.0,
                     1.05,
                     1.1,
                     1.15,
                     1.2,
                     1.25,
                     1.3,
                     1.35,
                     1.4,
                     1.45,
                     1.5,
                     1.55
                    )
  my_names_l = ([
        # 'mixed0',  # neat...eyes, ripples
        'mixed1',  # waves, some eyes
        # 'mixed2',  # neat...lots of dots
        # 'mixed3',  # eyes, sorta faces
        # 'mixed4',  # eyes, ripples
        # 'mixed5',  # good one...eyes/face
        # 'mixed6',  # tessellation?
        # 'mixed7',  # tessellation?
        # 'mixed8',  # puzzle pieces / tesselation (best tessellation)
        # 'mixed9',  # tessellation (high detail)
        # 'mixed9_0',  # tessellation (high detail)
        # 'mixed9_1',  # tessellation (high detail), bubbly (kinda)
        # 'mixed10'  # tessellation (high detail), bubbly (kinda)
      ],['mixed2'])

  for i in range(len(my_steps_per_octave_l)):
    for j in range(len(my_step_size_l)):
      for k in range(len(my_octaves_l)):
        for l in range(len(my_octave_scale_l)):
          for m in range(len(my_names_l)):
            my_steps_per_octave=my_steps_per_octave_l[i]
            my_step_size=my_step_size_l[j]
            my_octaves=my_octaves_l[k]
            my_octave_scale=my_octave_scale_l[l]
            my_names = my_names_l[m]

            layers_str = ''
            for name in my_names:
              layers_str += name + '_'

            save_name1 = 'dreamed_output/' + filename.split(".")[0] + '_' + layers_str + str(my_steps_per_octave) + '_' + str(my_step_size) + '_' + str(my_octaves[0]) + '_' + str(my_octaves[-1]) + '_' + str(my_octave_scale) + '_' + '.png'
            import os.path
            from os import path
            if path.exists(save_name1):
              continue

            base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

            layers = [base_model.get_layer(name).output for name in my_names]

            # Create the feature extraction model
            dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

            img = tf.constant(np.array(original_img))
            base_shape = tf.shape(img)[:-1]

            get_tiled_gradients = TiledGradients(dream_model)


            img = run_deep_dream_with_octaves(img=original_img,
                                              steps_per_octave=my_steps_per_octave,
                                              step_size=my_step_size,
                                              octaves=my_octaves,
                                              octave_scale=my_octave_scale,
                                              img_gradients=get_tiled_gradients)
            get_tiled_gradients = TiledGradients(dream_model)
            display.clear_output(wait=True)
            img = tf.image.resize(img, base_shape)
            img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)

            show(img, save_name1)

main()

print('total runtime = ' + str(time.time() - startTime) + '\n')
