## Colocamos los inputs
import pickle

# Abrimos el archivo donde estan las cosas que nos permitiran hacer predicciones
with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

# Permite realizar las predicciones, hace el preprocesamiento y prediccion de los datos
def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])