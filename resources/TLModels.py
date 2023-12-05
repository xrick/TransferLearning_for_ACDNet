import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.layers as L
import numpy as np

class TLACDNet:
	def __init__(self, pretrained_model_path=None,opt=None, num_class=11):
		self.opt = opt
		self.pretrained_model_path = pretrained_model_path
		self.new_model = None
		self.num_class = num_class

	def Create_TLACDNet():
		model = load_model(self.pretrained_model_path)
		print(f"original model loaded....")
		# for l in model.layers:
		# 	print(f"layer:{l} trainable weight length is {len(l.wei)}")
		total_layers_num = len(model.layers)
		replaced_layers_num = 2
		freeze_layers_num = total_layers_num-replaced_layers_num

		## freeze layers
		for i in range(freeze_layers_num):
			model.layers[i].trainable = False

		for j in range(freeze_layers_num, total_layers_num):
			model.layers[j].trainable = True

		custom_layers = model.layers[freeze_layers_num-1].output
		custom_layers = Dense(num_classes)(custom_layers)
		# custom_layers = Softmax()(custom_layers)
		custom_layers = Dense(num_classes,activation="softmax")(custom_layers)

		new_model = Model(inputs=model.input,outputs=custom_layers)
		print("new model info:\n")
		print(new_model.summary())
		print("\n")
		return new_model




