import numpy as np
import numpy.linalg as nla

from itertools import product

class SelfOrganizingMaps():
    
    def __init__(self,grid_size,dimension=2):
        
        self.q = dimension # Dimension of the principal subspace
        self.grid_size = grid_size # number of points for each direction of the grid
    
    # Initialize the prototypes onto the principal hyperplane
    def initialize_prototypes(self,X):
        
        _,p = X.shape
        
        # Compute SVD for the data
        X_mean = np.mean(X,axis=0)
        U,D,Vt = nla.svd(X-X_mean)
        
        # Get the first q principal directions and the their range
        principal_range = []
        for i in range(self.q):
            v = U[:,i] * D[i] + X_mean @ Vt[i] # principal component corrected by X_mean
            
            # Get range and incremet size
            v_min = min(v)
            v_max = max(v)
            delta = (v_max-v_min)/self.grid_size[i]
            
            # Record position of first prototype along principal direction, and increment size
            principal_range.append((v_min+delta/2,delta))
        
        # Create grid for initialization
        if self.q == 1:
            grid = [(m,) for m in range(self.grid_size[0])]
        else:
            grid = product(*[range(m) for m in self.grid_size])
        
        # Position the prototypes onto the principal hyperplane
        self.prototypes = {}
        for position in grid:
            
            # Compute the position of the prototypes in the original space
            coordinates = np.zeros(p)
            for i in range(self.q):
                v_min,v_delta = principal_range[i]
                v_pos = position[i]
                coordinates += (v_min+v_pos*v_delta)*Vt[i]
                
            self.prototypes[position] = coordinates
    
    # Compute the reconstruction error for the model
    def reconstruction_error(self,X):
        
        N,_ = X.shape
        
        # Compute distance of each observation from the prototypes
        distances = np.empty((N,0))
        for m in self.prototypes.values():
            dist_m = np.sum((X-m)**2,axis=1)
            distances = np.hstack((distances,dist_m.reshape((N,1))))
        
        # Sum the minimum distance for each observation
        error = np.sum(np.min(distances,axis=1))
        
        return error
        
    # Online update of the prototypes
    def online_update(self,x):
        
        # In the original space (p-dimensional), find closest prototype
        distance = [(nla.norm(x-m), key) for key, m in self.prototypes.items()]
        _, opt_key = min(distance)
        opt_position = np.array(opt_key)
        
        # In the principal subspace (q-dimension), update prototypes close to optimal one
        for key in self.prototypes.keys():
            position = np.array(key)
            if nla.norm(position - opt_position) < self.r:
                self.prototypes[key] += self.alpha * (x - self.prototypes[key])
    
    # Fit the data by updating the prototypes iteratively
    def fit(self,X,iterations,alpha,r):
        
        N,_ = X.shape
        self.alpha = alpha
        self.r = r
        
        # Initialize the position of the prototypes
        self.initialize_prototypes(X)
        
        # Parameters alpha and r decrease by this amount at each iteration
        delta_alpha = alpha/iterations
        delta_r = (r-1)/iterations
        
        # Store the reconstraction errror
        self.learning_curve = []
        
        for i in range(iterations):
            x = X[i%N] # We could sample X, if we prefer
            
            self.online_update(x)
            
            # Decrease alpha and r at each step
            self.alpha -= delta_alpha
            self.r -= delta_r
            
            if i%N == 0:
                error = self.reconstruction_error(X)
                self.learning_curve.append(error)
                
    # We use this function to check which classes ends where, but y is in general not known
    def cluster(self,X,y):
        
        N,_ = X.shape
        
        # Compute distance of each observation from the prototypes
        distances = np.empty((N,0))
        for m in self.prototypes.values():
            dist_m = np.sum((X-m)**2,axis=1)
            distances = np.hstack((distances,dist_m.reshape((N,1))))
        
        # Find for each observation its closest prototype
        classification = np.argmin(distances,axis=1)
        
        # Cluster observations together
        clusters = {}
        for i,m in enumerate(self.prototypes.keys()):
            clusters[m] = y[classification == i]
            
        return clusters