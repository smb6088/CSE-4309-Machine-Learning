import numpy as np

def gaussian_probability(num,mean,std):
    num1 = (1/(np.sqrt(2*(np.pi)*std**2)))
    num2 = np.exp(-np.square(num-mean)/(2*std**2))
    return num1 * num2
    

def naive_bayes(training_file,test_file):
    train_data = np.loadtxt(training_file,dtype=float)

    num_classes = len(np.unique(train_data[:,-1]))
    num_cols = train_data.shape[1]-1
    cl_at_curve = []
    class_prob_list = []
    for c_col in range(1,num_classes+1,1):
        temp = train_data[:,-1] == c_col
        temp = train_data[temp]
        class_prob = (len(temp))/(len(train_data))
        class_prob_list.append(class_prob)
        for f_col in  range(1,num_cols+1,1):
            mean = np.mean(temp[:,f_col-1])
            std = np.std(temp[:,f_col-1])
            if std < 0.01:
                std = 0.01
            cl_at_curve.append([c_col,f_col,mean,std])
        
    for num in range(0,len(cl_at_curve),1):
        print("Class {:d}, attribute {:d}, mean = {:.2f}, std={:.2f}".format(int(cl_at_curve[num][0]),int(cl_at_curve[num][1]),float(cl_at_curve[num][2]),float(cl_at_curve[num][3])))

    cl_at_curve = np.array(cl_at_curve,dtype=float)
    test_data = np.loadtxt(test_file,dtype=float)
    result = []
    for test_case in range(0,len(test_data)):
        gaussian_curve = []
        for c_col in range(1,num_classes+1,1):
            temp = 1
            for f_col in range(1,num_cols+1,1):
                temp2 = cl_at_curve[(cl_at_curve[:,0] == c_col) & (cl_at_curve[:,1]==f_col)]
                mean = temp2[0,2]
                std = temp2[0,3]
                temp *= gaussian_probability(float(test_data[test_case,f_col-1]),float(mean),float(std))
            gaussian_curve.append(temp)
        
        total_prob_x = 0
        for i in range(0,num_classes,1):
            total_prob_x += (gaussian_curve[i]*class_prob_list[i])
        
        prob_c_x = []
        for i in range(0,num_classes,1):
            prob_c_x.append((gaussian_curve[i]*class_prob_list[i])/total_prob_x)

        max_prob = max(prob_c_x)
        best_class = []
        for i in range(num_classes):
            if prob_c_x[i] == max_prob:
                best_class.append(i+1)
        predicted_class = -1
        if len(best_class) > 1:
            predicted_class = np.random.choice(best_class)
        else:
            predicted_class = best_class[0]
        
        true_class = test_data[test_case,-1]

        if predicted_class == true_class:
            accuracy = 1.0
        else:
            accuracy = 0.0

        result.append([test_case+1,predicted_class,prob_c_x[predicted_class-1],true_class,accuracy])

    for i in sorted(result):
        print("ID={:5d}, Predicted={:3d}, Probability={:.4f}, True={:3d}, accuracy={:3d}".format(int(i[0]),int(i[1]),float(i[2]),int(i[3]),int(i[4])))

    total = 0
    for i in result:
        total += i[4]
    


    total_accuracy = total/test_data.shape[0]
    print(f"Classification accuracy={total_accuracy:6.4f}")

    

             