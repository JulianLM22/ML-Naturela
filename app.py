from flask import Flask, request, jsonify
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.dump import dump, load
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load dataset and train model on server start
reader = Reader(line_format='timestamp user rating item', sep=';', rating_scale=(0, 174699), skip_lines=1)
sales_data = Dataset.load_from_file('DatosPreparados.csv', reader=reader)
svd = SVD(verbose=True, n_epochs=10)
trainset = sales_data.build_full_trainset()
svd.fit(trainset)
dump_file = 'model.pkl'
dump(dump_file, algo=svd)

@app.route('/train', methods=['POST'])
def train():
    # ESCALAR LOS VALORES POR CLIENTE
    df_compras = pd.read_csv('DatosPreparados.csv', sep=';')
    from sklearn.preprocessing import MinMaxScaler
    # Agrupar las compras por cliente
    compras_por_cliente = df_compras.groupby(['identificacion','producto'])['Cantidad'].sum().reset_index()
    # Crear un objeto MinMaxScaler
    scaler = MinMaxScaler(feature_range=(1, 5))
    # Escalar las compras por cliente
    compras_por_cliente['Cantidad'] = scaler.fit_transform(compras_por_cliente[['Cantidad']])

    #GUARDA LOS DATOS ESCALADOS
    compras_por_cliente.to_csv('scaled_data.csv', index=False)


    # Cargar datos de ventas de clientes
    reader = Reader(line_format='user item rating ', sep=',', rating_scale=(1, 5), skip_lines=1)
    sales_data = Dataset.load_from_file('scaled_data.csv', reader=reader)
    cross_validate(svd, sales_data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    # Dividir los datos en conjuntos de entrenamiento y prueba
    trainset1, testset = train_test_split(sales_data, test_size=0.2)
    #tren = trainset1.build_full_trainset() 
    svd.fit(trainset1)
    # Evaluar modelo
    predictions = svd.test(testset)
    # Guardar el modelo
    dump_file = 'model.pkl'
    dump(dump_file, algo=svd)
    return jsonify({'prediction': predictions})

@app.route('/predict', methods=['POST'])
def predict():
    import surprise

    # Check if all required fields are present in the request
    if not request.json or 'user_id' not in request.json or 'item_id' not in request.json:
        return jsonify({'error': 'Missing required fields.'}), 400

    try:
        # Load trained model
        model_tuple = load('model.pkl')
        model = model_tuple[1]

        # Check if the model is valid and loaded correctly
        if not isinstance(model, surprise.prediction_algorithms.matrix_factorization.SVD):
            return jsonify({'error': 'Invalid model.'}), 500

        # load data from request
        user_id = request.json['user_id']
        num_items = request.json['item_id']

        # Load data from items
        items = pd.read_csv('nombre_productos.csv', sep=';')
        #items = items.rename(columns={'item_id': 'iid'})
        # Get top n recommendations for the user

        # Get Items from User
        user_items = model.trainset.ur[model.trainset.to_inner_uid(user_id)]
        user_unseen_items = [iid for iid in model.trainset.all_items() if iid not in user_items]

        # Make Predictions
        predictions = [model.predict(uid=user_id, iid=str(iid)) for iid in user_unseen_items]

        #Order By value
        top_n = pd.DataFrame(predictions).sort_values(by='est', ascending=False).head(num_items)
        top_n['item_id'] = top_n['iid'].astype(int)

        top_n = pd.concat([top_n.set_index('item_id'), items.set_index('item_id')], axis=1, join='inner')
        top_n = top_n[['item_name', 'est']].reset_index().to_dict('records')
        # Return prediction
        return jsonify({'prediction': top_n}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
