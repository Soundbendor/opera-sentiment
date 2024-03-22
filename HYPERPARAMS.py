from ENV import target_second
import math
sr = 16000

hyperparams = {}
hyperparams['input_size'] = 1024
hyperparams['input_length'] = sr*target_second
hyperparams['channel_size'] = math.ceil(hyperparams["input_length"]/hyperparams["input_size"])
hyperparams['flatten_size'] = hyperparams['input_size']*hyperparams['channel_size']

hyperparams['batch_size'] = 32
hyperparams['output_size'] = 1

hyperparams['dense_units'] = hyperparams['output_size']
hyperparams['dropout'] = 0.3
hyperparams['lr'] = 0.001
hyperparams["loss"] = 'BCE' # or 'categorical_crossentropy'