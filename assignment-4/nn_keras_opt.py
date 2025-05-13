import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

def nn_keras(directory, dataset, layers, units_per_layer, epochs):

    training_file = f"{directory}/{dataset}_training.txt"
    train_data = np.loadtxt(training_file,dtype=str)
    X_train = train_data[:, :-1].astype(float)
    y_train = train_data[:, -1]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    max_abs_value = np.max(np.abs(X_train))
    X_train = X_train / max_abs_value
    
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_train.shape[1],)))
    
    for _ in range(0,layers - 2,1):
        model.add(keras.layers.Dense(units_per_layer, activation='tanh'))

    num_classes = len(np.unique(y_train))+1
    model.add(keras.layers.Dense(num_classes, activation='sigmoid'))
    
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_encoded, epochs=epochs, verbose=1)
    
    test_file = f"{directory}/{dataset}_test.txt"
    test_data = np.loadtxt(test_file,dtype=str)
    X_test = test_data[:, :-1].astype(float)
    y_test = test_data[:, -1]
    
    X_test = X_test / max_abs_value
    y_test_encoded = le.fit_transform(y_test)
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    

    total_accuracy = 0
    for i in range(0,len(y_test),1):
        pred_class = y_pred[i]
        true_class = y_test_encoded[i]
        max_proba = np.max(y_pred_proba[i])
        ties = np.sum(y_pred_proba[i] == max_proba)
        if ties == 1:
            if pred_class == true_class:
                accuracy = 1.0  
            else:
                accuracy = 0.0
        else:
            if true_class in np.where(y_pred_proba[i] == max_proba)[0]:
                accuracy = 1.0 / ties
            else:
                accuracy=0.0
        
        total_accuracy += accuracy
        pred_class = le.inverse_transform([y_pred[i]])[0]
        true_class = le.inverse_transform([y_test_encoded[i]])[0]
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % 
           (i+1, pred_class, true_class, accuracy))
    
    classification_accuracy = total_accuracy / len(y_test)
    print('classification accuracy=%6.4f\n' % (classification_accuracy))