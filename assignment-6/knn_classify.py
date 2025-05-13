import numpy as np
import random

def knn_classify(training_file,test_file,k):
    train_data = np.loadtxt(training_file,dtype=float)
    X_train = train_data[:,:-1]
    y_train = train_data[:,-1]
    
    test_data = np.loadtxt(test_file,dtype=float)
    X_test = test_data[:,:-1]
    y_test = test_data[:,-1]
    
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    stds = np.where(stds == 0, 1, stds)
    X_train_normalized = (X_train - means) / stds
    X_test_normalized = (X_test - means) / stds
    
    final_accurary = []
    for i in range(0,len(X_test_normalized),1):
        
        distance = np.linalg.norm(X_train_normalized - X_test_normalized[i], axis=1)
        index = np.argpartition(distance, k)[:k]
        neighbor_y = y_train[index]
        
        values, count = np.unique(neighbor_y, return_counts=True)
        max_count = count.max()
        best_label = values[count == max_count]
        predicted_class = random.choice(best_label)
        true_class = y_test[i]

        if len(best_label) == 1 and predicted_class == true_class:
            accuracy = 1
        else:
            accuracy = 0

        if true_class in best_label:
            accuracy = 1/len(best_label)
        else:
            accuracy = 0

        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' % (i+1, predicted_class, true_class, accuracy))
        final_accurary.append(accuracy)

    classification_accuracy = sum(final_accurary) / len(final_accurary)
    print('classification accuracy=%6.4f' % (classification_accuracy))