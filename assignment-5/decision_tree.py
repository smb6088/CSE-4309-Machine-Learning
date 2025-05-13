import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

class Node:
    def __init__(self, feature=-1, threshold=-1, info_gain=0, left=None, right=None, class_dist=-1):
        self.feature = feature
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right
        self.class_dist = class_dist

def DISTRIBUTION(class_train):
    class_train = np.array(class_train)
    if class_train.size == 0:
        return np.zeros(0)
    num_classes = int(np.max(class_train)) + 1 
    dist = np.zeros(num_classes)
    
    unique_classes, counts = np.unique(class_train, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        dist[int(cls)] = count
    
    return dist/len(class_train)

def INFORMATION_GAIN(fea_train,class_train,A,threshold):
    H_E = H_EL = H_ER = 0
    fea_train = np.array(fea_train)
    class_train = np.array(class_train)
    left_mask = fea_train[:, A] < threshold
    fea_train_left = fea_train[left_mask]
    class_train_left = class_train[left_mask]
    fea_train_right = fea_train[~left_mask]
    class_train_right = class_train[~left_mask]
    total_distribution = DISTRIBUTION(class_train)
    left_distribution = DISTRIBUTION(class_train_left)
    right_distribution = DISTRIBUTION(class_train_right)
    for i in total_distribution:
        if i > 0:
            H_E -= (i * np.log2(i))
    for i in left_distribution:
        if i > 0:
            H_EL -= (i * np.log2(i))
    for i in right_distribution:
        if i > 0:
            H_ER -= (i * np.log2(i))
    K = len(fea_train)
    K1 = len(fea_train_left)
    K2 = len(fea_train_right)
    entropy = H_E - ((K1 / K) * H_EL) - ((K2 / K) * H_ER)
    return entropy

def CHOOSE_ATTRIBUTE(fea_train, class_train, attributes, option):
    if option == "optimized":
        max_gain = best_attribute = best_threshold = -1
        for A in attributes:
            attribute_values = fea_train[:, A]
            L = min(attribute_values)
            M = max(attribute_values)
            for K in range(1, 51, 1):
                threshold = L + K*(M-L)/51
                gain = INFORMATION_GAIN(fea_train, class_train, A, threshold)
                if gain > max_gain:
                    max_gain = gain
                    best_attribute = A
                    best_threshold = threshold
        return (best_attribute, best_threshold, max_gain)
    else:
        A = random.choice(attributes)
        max_gain = best_threshold = -1
        attribute_values = fea_train[:, A]
        L = min(attribute_values)
        M = max(attribute_values)
        for K in range(1, 51, 1):
            threshold = L + K*(M-L)/51
            gain = INFORMATION_GAIN(fea_train, class_train, A, threshold)
            if gain > max_gain:
                max_gain = gain
                best_threshold = threshold
        return (A, best_threshold, max_gain)

def DTL(fea_train, class_train, attributes, default, pruning, option):
    if len(fea_train) < pruning:
        return Node(class_dist=default)
    unique_classes = np.unique(class_train)
    if len(unique_classes) == 1:
        dist = DISTRIBUTION(class_train)
        return Node(class_dist=dist)
    best_attribute, best_thr, info_gain = CHOOSE_ATTRIBUTE(fea_train, class_train, attributes, option)
    tree = Node(best_attribute, best_thr, info_gain)
    left_mask = fea_train[:, best_attribute] < best_thr
    fea_train_left = fea_train[left_mask]
    class_train_left = class_train[left_mask]
    fea_train_right = fea_train[~left_mask]
    class_train_right = class_train[~left_mask]
    total_dist = DISTRIBUTION(class_train)
    tree.left = DTL(fea_train_left, class_train_left, attributes, total_dist, pruning, option)
    tree.right = DTL(fea_train_right, class_train_right, attributes, total_dist, pruning, option)
    return tree

def DTL_TopLevel(fea_train,class_train,num_features,pruning,option):
    attributes = []
    for i in range(0,num_features,1):
        attributes.append(i)
    default = DISTRIBUTION(class_train)
    return DTL(fea_train,class_train,attributes,default,pruning,option)

def predict(tree, test_object):
    if tree.left is None and tree.right is None:
        return tree.class_dist
    if test_object[tree.feature] < tree.threshold:
        return predict(tree.left, test_object)
    return predict(tree.right, test_object)

def print_tree(tree, i):
    if not tree:
        return
    node_id = 1
    queue = [(tree, node_id)]
    while queue:
        current_node, nid = queue.pop(0)
        if current_node.feature != -1:
            feature_id = current_node.feature 
        else:
            feature_id = -1
        print("tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f" % 
              (i, nid, feature_id, current_node.threshold, current_node.info_gain))
        if current_node.left:
            node_id += 1
            queue.append((current_node.left, node_id))
        if current_node.right:
            node_id += 1
            queue.append((current_node.right, node_id))

def decision_tree(training_file, test_file, option, pruning_thr):

    train_data = np.loadtxt(training_file,dtype=str)
    fea_train = train_data[:,:-1].astype(float)
    class_train = train_data[:,-1]
    le = LabelEncoder()
    class_train_encoded = le.fit_transform(class_train)
    num_features = fea_train.shape[1]
    
    trees = []
    if option == "optimized":
        trees.append(DTL_TopLevel(fea_train, class_train_encoded, num_features, pruning_thr, "optimized"))
    else:
        for i in range(0,option,1):
            trees.append(DTL_TopLevel(fea_train, class_train_encoded, num_features, pruning_thr, option))
    
    for i in range(0,len(trees),1):
        print_tree(trees[i], i+1)
    test_data = np.loadtxt(test_file, dtype=str)
    fea_test = test_data[:,:-1].astype(float)
    class_test = test_data[:,-1]
    class_test_encoded = le.fit_transform(class_test)
    
    total_correct = 0
    for i in range(0,len(test_data),1):
        if len(trees) > 1:
            predictions = [np.argmax(predict(tree, fea_test[i])) for tree in trees]
            pred_counts = np.bincount(predictions)
            max_count = np.max(pred_counts)
            pred_classes = np.where(pred_counts == max_count)[0]
            
            if len(pred_classes) > 1:
                if int(class_test_encoded[i]) in pred_classes:
                    accuracy = 1.0 / len(pred_classes)
                else:
                    accuracy = 0.0
                pred_class = pred_classes[0]
            else:
                pred_class = pred_classes[0]
                accuracy = 1.0 if pred_class == int(class_test_encoded[i]) else 0.0
                
        else:
            prob_dist = predict(trees[0], fea_test[i])
            pred_class = np.argmax(prob_dist)
            accuracy = 1.0 if pred_class == int(class_test_encoded[i]) else 0.0
            
        total_correct += accuracy
        pred_class = le.inverse_transform([pred_class])[0]
        true_class = le.inverse_transform([class_test_encoded[i]])[0]
        print("ID=%5d, predicted=%3s, true=%3s, accuracy=%4.2f" % 
              (i+1, pred_class, true_class, accuracy))
    
    print("classification accuracy=%6.4f" % (total_correct/len(test_data)))