import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def euclidean_distance(x1,x2):
  return np.sqrt(np.sum((x2- x1)**2))

def manhattan_distance(x1,x2):
  return np.sum(np.abs(x2-x1))

def minkowski_distance(x1,x2,p):
  return np.sum(np.abs(x2-x1)**p) ** (1/p)

distance_matrix={
   'euclidean': euclidean_distance,
   'manhattan': manhattan_distance,
   'minkowski': minkowski_distance
}

class KDNode:
   def __init__(self,point , label, left=None , right=None,axis=0):
     self.point=point
     self.label=label
     self.left=left
     self.right=right
     self.axis=axis

class KDTree:
  def __init__(self,x,y,depth=0):
    if len(x)==0:
      self.node=None
      return
    
    k=x.shape[1]
    axis= depth % k
    sorted_idx = x[:, axis].argsort()
    median = len(sorted_idx) // 2
    self.node = KDNode(
              point=x[sorted_idx[median]],
              label=y[sorted_idx[median]],
              left=KDTree(x[sorted_idx[:median]], y[sorted_idx[:median]], depth + 1).node,
              right=KDTree(x[sorted_idx[median + 1:]], y[sorted_idx[median + 1:]], depth + 1).node,
              axis=axis
      )
  
  def _search(self, node, point, k, heap, distance_func):
        if node is None:
            return
        dist = distance_func(point, node.point)
        heap.append((dist, node.label, node.point))
        heap.sort(key=lambda x: x[0])
        if len(heap) > k:
            heap.pop()
        axis = node.axis
        diff = point[axis] - node.point[axis]
        if diff <= 0:
         nearer, further = node.left, node.right
        else:
         nearer, further = node.right, node.left
         
        self._search(nearer, point, k, heap, distance_func)  
        if len(heap) < k or abs(diff) < heap[-1][0]:
            self._search(further, point, k, heap, distance_func)


  def query(self, point, k, distance_func):
        heap = []
        self._search(self.node, point, k, heap, distance_func)
        return heap
  


class KNN:
    def __init__(self, k=3, task="classification", distance="euclidean", p=3, weighted=False):
        self.k = k
        self.task = task
        self.distance = distance
        self.p = p
        self.weighted = weighted   

    def fit(self, training_features, training_labels):
        self.x_train = np.array(training_features)
        self.y_train = np.array(training_labels)
        self.tree = KDTree(self.x_train, self.y_train)

    def _predict_point(self, single_point):
        
        if self.distance == "minkowski":
            distance_function = lambda a, b: minkowski_distance(a, b, self.p)
        else:
            distance_function = distance_matrix[self.distance]

        
        neighbors = self.tree.query(single_point, self.k, distance_function)

        
        distances = [distance for distance, label, neighbor_point in neighbors]
        labels = [label for distance, label, neighbor_point in neighbors]

        
        if self.task == "classification":
            if self.weighted:
                
                weighted_votes = {}
                for distance, label in zip(distances, labels):
                    weight = 1 / (distance + 1e-9)   
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                return max(weighted_votes.items(), key=lambda item: item[1])[0]
            else:
                
                return Counter(labels).most_common(1)[0][0]

        
        else:
            if self.weighted:
                weights = [1 / (distance + 1e-9) for distance in distances]
                return np.average(labels, weights=weights)
            else:
                return np.mean(labels)

    def predict(self, input_data):
        input_data = np.atleast_2d(input_data)
        return np.array([self._predict_point(single_point) for single_point in input_data])