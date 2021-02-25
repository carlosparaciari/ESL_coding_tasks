import numpy as np
from math import ceil

# Find box containing the input points of X
def find_minmax_box(X):
    
    _,feature = X.shape
    
    x_min = [np.inf for _ in range(feature)]
    x_max = [-np.inf for _ in range(feature)]

    for x in X:
        for p in range(feature):
            if x[p] < x_min[p]:
                x_min[p] = x[p]
            if x[p] > x_max[p]:
                x_max[p] = x[p]
                
    return [*x_min,*x_max]

# Given the boundary, check which input points belong to the box
def contained_observations(X,boundary):
    
    _,features = X.shape
    
    boundary_min = boundary[:features]
    boundary_max = boundary[features:]
    
    bool_contained = []
    
    for i,x_min in enumerate(boundary_min):
        bool_contained.append(X[:,i] >= x_min)
        
    for i,x_max in enumerate(boundary_max):
        bool_contained.append(X[:,i] <= x_max)
        
    return np.all(bool_contained,axis=0)
    
# Find the optimal peeling along a given direction
def find_peeling(X,y,boundary,alpha,p,is_min):
    
    _,features = X.shape
    
    # Order the input points wrt the feature we are interested to peel
    contained = contained_observations(X,boundary)
    
    X_line = X[:,p]
    X_bounded = sorted(X_line[contained])
    N_bounded = len(X_bounded)
    
    # The number of points we wish to peel off
    N_peel = max(ceil(N_bounded*alpha),1)
    
    next_boundary = boundary.copy()
    
    if is_min:
        # In case the next min point is equal to the previous one, we move forward by one point
        x_min_prev = boundary[p]
        while True:
            x_min_next = X_bounded[N_peel]
            if x_min_prev == x_min_next:
                N_peel += 1
            else:
                break
        next_boundary[p] = x_min_next
    else:
        # Similarly for the next max point
        x_max_prev = boundary[features+p]
        while True:
            x_max_next = X_bounded[N_bounded-(N_peel+1)]
            if x_max_prev == x_max_next:
                N_peel += 1
            else:
                break
        next_boundary[features+p] = x_max_next
    
    next_contained = contained_observations(X,next_boundary)
    next_y = np.mean(y[next_contained])
    
    return next_y, next_boundary

# Shrink the boundary box by an amount alpha
def shrink_box(X,y,boundary,alpha):
    
    _,features = X.shape
    left_side = [True,False]
    
    next_y_list = []
    next_boundaries = []
    
    for side in left_side:
        for p in range(features):
            y_average, next_boundary = find_peeling(X,y,boundary,alpha,p,is_min=side)
            next_y_list.append(y_average)
            next_boundaries.append(next_boundary)
    
    # Select the optimal peeling direction
    opt_index = np.argmax(next_y_list)
    
    opt_y = next_y_list[opt_index]
    opt_boundary = next_boundaries[opt_index]
        
    return opt_y, opt_boundary

# Find the next possible expansion for the peeled boundary
def find_expansion(X,y,boundary,delta=1e-10):
    
    _,features = X.shape
    full_boundary = find_minmax_box(X)
    
    new_boundary_points = [0 for _ in range(2*features)]
    
    for p in range(features):
        
        # Check on the left of the box wrt feature p
        left_boundary = boundary.copy()
        left_boundary[p] = full_boundary[p]
        left_boundary[features+p] = boundary[p] - delta
        left_contained = contained_observations(X,left_boundary)
        
        if np.sum(left_contained) != 0:
            # If there are points on the left, then find the closest
            x_left = np.max(X[left_contained][:,p])
        else:
            # Else keep the previous point
            x_left = None
        
        new_boundary_points[p] = x_left
        
        # Check on the right of the box wrt feature p
        right_boundary = boundary.copy()
        right_boundary[p] = boundary[features+p] + delta
        right_boundary[features+p] = full_boundary[features+p]
        right_contained = contained_observations(X,right_boundary)
        
        if np.sum(right_contained) != 0:
            # If there are points on the right, then find the closest
            x_right = np.min(X[right_contained][:,p])
        else:
            # Else keep the previous point
            x_right = None
        
        new_boundary_points[features+p] = x_right
        
    return new_boundary_points

def expand_box(X,y,boundary):
    
    contained = contained_observations(X,boundary)
    
    y_optimal = np.mean(y[contained])
    output_boundary = boundary
    
    # Check which points can be expanded
    expansion_points = find_expansion(X,y,boundary)
    
    # For each expansion, check if adding the points imporve the y average
    for i,x_val in enumerate(expansion_points):
        if x_val == None:
            continue
            
        next_boundary = boundary.copy()
        next_boundary[i] = x_val
        
        next_contained = contained_observations(X,next_boundary)
        y_next = np.mean(y[next_contained])
        
        if y_next >= y_optimal:
            output_boundary = next_boundary
            y_optimal = y_next
            
    return y_optimal, output_boundary

# Shrink and expand a single box to find a bump
def single_box(X,y,alpha,min_obs=10):
        
    # Shrinkning part of PRIM
    boundary = find_minmax_box(X)
    y_average = np.mean(y)
    
    contained = contained_observations(X,boundary)
    tot_num = np.sum(contained)
    
    y_average_list = [y_average]
    boundary_list = [boundary]
    number_list = [tot_num]

    while tot_num > min_obs:
        
        y_average, boundary = shrink_box(X,y,boundary,alpha)

        contained = contained_observations(X,boundary)
        tot_num = np.sum(contained)
        
        y_average_list.append(y_average)
        boundary_list.append(boundary)
        number_list.append(tot_num)
    
    # After the above round, boundary is the smallest box and tot_num is the # of points contained
    
    # Expanding part of PRIM
    while True:
        y_average, boundary = expand_box(X,y,boundary)
        
        contained = contained_observations(X,boundary)
        previous_num = tot_num
        tot_num = np.sum(contained)
        
        if previous_num == tot_num:
            break
        
        y_average_list.append(y_average)
        boundary_list.append(boundary)
        number_list.append(tot_num)
        
    return y_average_list, boundary_list

# Check each boxes created by the algorithm on a validation set. Select the one with higher y_av
def cross_validation(X_val,y_val,boundary_list):

    y_opt = -np.inf
    
    for boundary in boundary_list:
        contained_val = contained_observations(X_val,boundary)
        y_average_val = np.mean(y_val[contained_val])
        
        if y_average_val > y_opt:
            y_opt = y_average_val
            boundary_opt = boundary

    return boundary_opt

# Patient Rule Induction Method
def PRIM(X,y,alpha=0.05,min_obs=10,cv_cut=0.3,num_boxes=3,use_cv=True):
    
    # Divide data into train and validation sets, according to CV cut
    if use_cv:
        N,_ = X.shape
        n_cv = ceil(N*(1-cv_cut))

        X_train = X[:n_cv]
        y_train = y[:n_cv]

        X_val = X[n_cv:]
        y_val = y[n_cv:]
    else:
        X_train = X
        y_train = y
    
    # Find boxes with maximum y_average
    boxes_list = []
    
    for _ in range(num_boxes):
        # Find optimal box
        _, boundary_list = single_box(X_train,y_train,alpha,min_obs)
        
        if use_cv:
            box = cross_validation(X_val,y_val,boundary_list)
            # Remove boxed observation from validation set
            outside_val = np.logical_not(contained_observations(X_val,box))
            X_val = X_val[outside_val]
            y_val = y_val[outside_val]
        else:
            box = boundary_list[-1]

        # Remove boxed observation from training set
        outside_train = np.logical_not(contained_observations(X_train,box))
        X_train = X_train[outside_train]
        y_train = y_train[outside_train]

        boxes_list.append(box)
        
    return boxes_list