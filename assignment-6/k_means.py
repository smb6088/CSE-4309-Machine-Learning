import numpy as np

def k_means(data_file, K, initialization):
    data_point = np.loadtxt(data_file)
    if data_point.ndim == 1:
        data_point = data_point.reshape(-1, 1)
    
    n_points = len(data_point)
    cluster_id = np.zeros(n_points, dtype=int)
    
    if initialization == "random":
        cluster_id = np.random.randint(1,K + 1,size=n_points)
    else:  
        for i in range(0,n_points,1):
            cluster_id[i] = (i % K) + 1
            
    while True:
        temp_cluster_id = cluster_id.copy()
        centroids = []
        for k in range(1,K + 1,1):
            same_id_point = data_point[cluster_id == k]
            if len(same_id_point) > 0:
                centroid = np.mean(same_id_point, axis=0)
            else:
                centroid = data_point[np.random.randint(len(data_point))]
            centroids.append(centroid)
        centroids = np.array(centroids)
        dist = np.zeros((len(data_point), len(centroids)))

        for i in range(0,len(centroids),1):
            dist[:, i] = np.sum((data_point - centroids[i]) ** 2, axis=1)
        cluster_id= np.argmin(dist, axis=1) + 1
        
        if np.array_equal(temp_cluster_id, cluster_id):
            break
    
    for i in range(0,len(data_point),1):
        if data_point.shape[1] == 1:
            print('%10.4f --> cluster %d' % (data_point[i][0], cluster_id[i]))
        else:
            print('(%10.4f, %10.4f) --> cluster %d' % (data_point[i][0], data_point[i][1], cluster_id[i]))