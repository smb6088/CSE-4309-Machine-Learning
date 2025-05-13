#Siddharth Bhagvagar
#1001986088
import numpy as np
import sys

def linear_regression(train_file,test_file,degree,lamda):

    train_data = np.loadtxt(train_file,dtype=float)
    y_train = train_data[:,-1]
    y_train = y_train.reshape((len(y_train),1))
    num_features = len(train_data[1,:])-1
    
    phi_func = []
    phi_func.append([1]*len(train_data[:,0]))
    
    
    if(1<=degree<=10):
        for d in range(0,num_features,1):
            for i in range(1,degree+1,1):
                value = train_data[:,d]**i
                phi_func.append(value)
    else:
        print("The Degree is out of range (1 to 10)")
        sys.exit(1)
    phi_func = np.array(phi_func).T
    #print(phi_func)
    #print(phi_func.shape)

        

    if lamda > 0:
        dim = (num_features)*degree+1
        regular_matrix = np.identity(dim)
        regular_matrix =lamda*regular_matrix
        #print(regular_matrix.shape)
    else:
        dim = (num_features)*degree+1
        regular_matrix = np.zeros((dim,dim))
    #print(phi_func.shape)
    #print(regular_matrix.shape)
    #print(y_train.shape)
    weight = np.linalg.pinv(regular_matrix + (phi_func.T)@(phi_func))@phi_func.T@y_train
    #print(weight)

    for i in range(0,len(weight),1):
        print("w{:d} = {:.4f}".format(int(i),float(weight[i])))

    test_data = np.loadtxt(test_file,dtype=float)

    test_phi_func = []
    test_phi_func.append([1]*len(test_data[:,0]))

    for d in range(0,num_features,1):
        for i in range(1,degree+1,1):
            value = test_data[:,d]**i
            test_phi_func.append(value)
    test_phi_func = np.array(test_phi_func)
    
    #print(test_phi_func.shape)
    #print(weight.shape)
    predict_value = test_phi_func.T @ weight

    for  i in range(0,test_data.shape[0],1):
        print("ID={:5d}, output={:14.4f}, target value= {:10.4f}, squared error={:.4f}".format(int(i+1),float(predict_value[i]),float(test_data[i,-1]),float((predict_value[i] - test_data[i,-1])**2)))
    #print(predict_value)
