# new1
import numpy as np
from utils.features import prepare_for_training
class LinearRegression:
    def __init__(self,data,labels,polynomial_degree=0,sinusoid_degree=0,nomalize_data=True):
      (data_processed,features_mean,features_deviation)=prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0)
      self.data=data_processed
      self.labels=labels
      self.features_mean=features_mean
      self.features_deviation=features_deviation
      self.polynomial_degree=polynomial_degree
      self.nomalize_data=nomalize_data
      num_features=self.data.shape[1]    #求列的值
      self.theta=np.zeros((num_features,1)) #对theta初始化
      
    def train(self,alpha,num_iterations=500):
      cost_history=self.gradient_descent(alpha,num_iterations)
      return self.theta,cost_history
    
    def gradient_descent(self,alpha,num_iterations):
      cost_history=[]
      for _ in range(num_iterations):
        self.gradient_step(alpha)
        cost_history.append(self.cost_function(self.data,self.labels))
      return cost_history  
        
    def cost_function(self,data,labels):
      num_examples=self.data.shape[0]
      delta=LinearRegression.hypothesis(self.data,self.theta)-labels
      cost=(1/2)*np.dot(delta.T,delta)
      return cost[0][0]
        
    def gradient_step(self,alpha):
      num_examples=self.data.shape[0]
      prediction=LinearRegression.hypothesis(self.data,self.theta)
      delta=prediction-self.labels
      theta=theta-alpha*(1/num_examples)*(no.dot(delta.T,self.data)).T
      self.theta=theta
      
   @staticmethod 
    def hypothesis(data,theta):
      prediction=np.dot(data,theta)
      return prediction
      
    def get_cost(self,data,labels):
      data_processed=prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree)[0]
      return self.cost_function( data_processed,labels)
      
    def predict(self,data):
      data_processed=prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.nomalize_data)[0]
      predictions=LinearRegression.hypothesis( data_processed,self.theta)
      
 import numpy as np
 import matplotlib as plt
 from utils.features import prepare_for_training
 class Linearegression:
       def _init_(self,data,theta,labels,polynomial_degree=0,sinusoid_degree=0,nomalize_data=True):
           (data_processed,features_mean,features_deviation)=prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0)
           self.data = data_processed
           self.labels = labels
           self.polynomial_degree = polynomial_degree
           self.nomalize_data = nomalize_data
           self.features_mean = features_mean
           self.features_deviation = features_deviation
           num_features = self.data.shape[1]
           self.theta = np.zeros((num_features,1))
           
       @staticmethod
       def gradient_step(self,data,alpha):
           num_examples = self.data.shape[0]
           prediction = LinearRegression.hypothesis(self.data,self.theta)
           delta = prediction - self.labels
           theta=theta-alpha*(1/num_examples)*(no.dot(delta.T,self.data)).T
           self.theta = theta
           
       @staticmethod   
       def hypothesis(self,data,theta):
           prediction = np.dot(data,theta)
           return prediction
           
       def predict(self,data):
           data_processed = prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0)
           prediction = LinearRegression.hypothesis(data_processed,self.theta)
           return prediction
           
       def gradient_descent(self,alpha,num_iterations):
           cost_history = []
           for _ in num_iterations:
               LinearRegression.gradient_step(data,alpha)
               cost_history.append(cost_function(self.data,self.labels))
           return cost_history
           
       def cost_function(self,data,labels)
           


























