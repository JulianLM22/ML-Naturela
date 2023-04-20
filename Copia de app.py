from flask import Flask, request, jsonify
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.dump import dump
from surprise.dump import load
svd = SVD(verbose=True, n_epochs=10)

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    # Cargar datos de ventas de clientes
    reader = Reader(line_format='timestamp user rating item', sep=';', rating_scale=(0, 174699), skip_lines=1)
    sales_data = Dataset.load_from_file('DatosPreparados.csv', reader=reader)
    cross_validate(svd, sales_data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    # Dividir los datos en conjuntos de entrenamiento y prueba
    trainset1, testset = train_test_split(sales_data, test_size=0.2)
    tren = sales_data.build_full_trainset() 
    svd.fit(tren)
    # Evaluar modelo
    predictions = svd.test(testset)
    # Guardar el modelo
    dump_file = 'model.pkl'
    dump(dump_file, algo=svd)
    return jsonify({'message': 'Model trained successfully.'})

@app.route('/predict', methods=['POST'])
def predict():
    import surprise

    # Read data from POST request
    data = request.get_json()

    # Load trained model
    model_file = 'model.pkl'
    model_tuple = surprise.dump.load(model_file)
    model = model_tuple[0]

    # Check if the model is valid and loaded correctly
    if model is not None and isinstance(model, surprise.prediction_algorithms.matrix_factorization.SVD):
        print("Model loaded successfully")
        # Make predictions
        user_id = data['user_id']
        item_id = data['item_id']
        prediction = model.predict(user_id, item_id)

        # Return prediction
        return jsonify({'prediction': prediction.est})
    else:
        print("Failed to load model")
        return jsonify({'message': 'Failed to load model'})


@app.route('/predict1', methods=['POST'])
def predict1():
    import surprise

    # Read data from POST request
    data = request.get_json()

    # Load trained model
    model_file = 'model.pkl'
    model, _ = load(model_file)

    # Check if the model is valid and loaded correctly
    if model is not None and isinstance(model, surprise.prediction_algorithms.matrix_factorization.SVD):
        print("Model loaded successfully")
        # Make predictions
        user_id = data['user_id']
        item_id = data['item_id']
        prediction = model.predict(user_id, item_id)

        # Return prediction
        return jsonify({'prediction': prediction.est})
    else:
        print("Failed to load model")
        return jsonify({'message': 'Failed to load model'})
if __name__ == '__main__':
    app.run()