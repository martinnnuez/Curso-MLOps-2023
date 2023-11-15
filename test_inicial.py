# Importamos dentro de este script el script de predict_incial para poder utilizar sus funciones
import predict_inicial

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

features = predict_inicial.prepare_features(ride)
pred = predict_inicial.predict(features)
print(pred)