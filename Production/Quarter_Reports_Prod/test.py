import pickle

MODELS_PATH = '../../Models_&_Files/Quarter_Reports_Models/'

with open(MODELS_PATH + 'rus_models_30mins.dictionary', 'rb') as config_dictionary_file:
    models = pickle.load(config_dictionary_file)

print(models[list(models.keys())[0]])
