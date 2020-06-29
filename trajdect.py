import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def point(df, nb_vehicle, min_frame, max_frame) :
	# This fonction get every entry and exit position of the vehicles and stock them into df_point.
    # We don't keep the vehicles who where present at the beginig of the video and the ones who were present at the end.

    # df_point contain: - vehicle identification (track_id)
    #                   - position x of the vehicle entrance (x_entry)
    #                   - position x of the vehicle exit (y_entry)
    #                   - position y of the vehicle entrance (x_exit)
    #                   - position y of the vehicle exit (y_exit)
    # X_entry contain:  - all the (x,y) positions of the entry points 
    # X_exit contain:   - all the (x,y) positions of the exit points 


    data = []
    for i in range(1, nb_vehicle):
        df_i = df[(df['track_id'] == i)]
        
        if len(df_i) > 0:
            entry_frame = df_i.min(axis = 0)['frame_id']
            exit_frame = df_i.max(axis = 0)['frame_id']
            # df_entry_exit only contain two line: the line where the vehicle enter the video and the one where it exit
            df_entry_exit = df_i[(df_i['frame_id'] == entry_frame) | (df_i['frame_id'] == exit_frame)]
            if 'csv' in df.columns:
                print('csv')
                if entry_frame not in min_frame and exit_frame not in max_frame:
                    x_entry = df_entry_exit['x'].iloc[0] 
                    y_entry = df_entry_exit['y'].iloc[0] 
                    x_exit = df_entry_exit['x'].iloc[-1]            
                    y_exit = df_entry_exit['y'].iloc[-1]
                    
                    if abs(x_entry-x_exit)>10 or abs(y_entry-y_exit)>10:
                        data.append([i, x_entry, y_entry, x_exit, y_exit])
            else:
                if entry_frame != min_frame and exit_frame != max_frame:
                    x_entry = df_entry_exit['x'].iloc[0] 
                    y_entry = df_entry_exit['y'].iloc[0] 
                    x_exit = df_entry_exit['x'].iloc[-1]            
                    y_exit = df_entry_exit['y'].iloc[-1]
                
                    if abs(x_entry-x_exit)>10 or abs(y_entry-y_exit)>10:
                        data.append([i, x_entry, y_entry, x_exit, y_exit])
                
    df_point = pd.DataFrame(data, columns=['track_id', 'x_entry', 'y_entry', 'x_exit', 'y_exit'])
    X_entry = np.column_stack((df_point['x_entry'], df_point['y_entry']))
    X_exit = np.column_stack((df_point['x_exit'], df_point['y_exit']))
       
    return df_point, X_entry, X_exit

def dbscan_model(X_entry, X_exit):
    # this function create the model which cluster the entry point wich are close with each other
    # and the exit point  wich are closed with each other.
    # We consider the model suitable when 95% of the point are clustered

    k = 15
    model_entry = DBSCAN(eps = 5, min_samples = k).fit(X_entry)
    noise_rate_entry = (list(model_entry.labels_).count(-1))/len(model_entry.labels_)
    
    while noise_rate_entry > 0.05:
        k -= 1
        model_entry = DBSCAN(eps = 5, min_samples = k).fit(X_entry)
        noise_rate_entry = (list(model_entry.labels_).count(-1))/len(model_entry.labels_)
        
    k = 15
    model_exit = DBSCAN(eps = 10, min_samples = 10).fit(X_exit)
    noise_rate_exit = (list(model_exit.labels_).count(-1))/len(model_exit.labels_)
    
    while noise_rate_exit > 0.05:
        k -= 1
        model_exit = DBSCAN(eps = 10, min_samples = k).fit(X_exit)
        noise_rate_exit = (list(model_exit.labels_).count(-1))/len(model_exit.labels_)  
        
    return model_entry, model_exit

def get_direction(df_point):
    # This function defines the different existing direction.
    # A direction start at an entry point and end at an exit point
    # We considere that two vehicle have the same direction if there entry point had been clustered in the same groupe 
    # And the exit point had been clustered in the same groupe 

    # We add to df_point: - the direction by wich the vehicle passed (direction)

    directions = []

    for ind in df_point.index:
        entry_point = str(df_point['entry_label'][ind])
        exit_point = str(df_point['exit_label'][ind])  

        # Vehicles that have a direction with an unclusterd entry or exit point are not retained
        if entry_point == '-1' or exit_point == '-1':
            directions.append('noise')
            
        else:
            directions.append(entry_point+exit_point)
            
    df_point['direction'] = directions
    df_point.head()
    existing_directions = list(set(directions))
    
    if 'noise' in existing_directions:
        existing_directions.remove('noise')
    
    # Rename the direction 1 to n
    for i, dire in enumerate (existing_directions):
            df_point = df_point.replace(dire, i)
            
    nb_dire = len(existing_directions)
    return(df_point, nb_dire)
        
def mean_dir(direction, df, df_point):
    # This function get all the vehicle that passed through the same direction and defines tan average direction.
    # X, Y: the vectors containing the mean position of the direction
    
    # We compute the distance betewen the postion of a same vehicle on two succesive fram e
    df_dist = []

    vehicle_1 = df['track_id']
    vehicle_1 = list(dict.fromkeys(vehicle_1))
    nb_vehicle = len(vehicle_1)
    
    
    for i in range(1, nb_vehicle+1):
        df_i = df[(df['track_id'] == i)]
        df_i = df_i.reset_index()
        dist = [None]

        for ind in df_i.index:
            if ind == len(df_i.index)-1:
                    break
                    
            dist.append(((df_i['x'][ind+1] - df_i['x'][ind])**2+(df_i['y'][ind+1] - df_i['y'][ind])**2)**(1/2))

        df_i['dist'] = dist
        df_dist.append(df_i)

    df_dist = pd.concat(df_dist)
    df_dist = df_dist.set_index('index')
    df = df_dist

    df_i = df_point[df_point['direction'] == direction]
    df_i = df_i.reset_index()
    df_i = df_i.drop(['index'], axis=1)
    len_dire = []

    for i in range(len(df_i)):
        track_id = df_i['track_id'][i]
        df_track_i = df[df['track_id'] == track_id]
        len_dire.append(len(list(df_track_i['x'])))

    # We get all the trajectories of the vehicules that have the same direction.
    # We make these paths the same length by removing the closest positions from a same vehicle 

    X = np.zeros(min(len_dire))
    Y = np.zeros(min(len_dire))
    past_len = 0
    for i in range(len(df_i)):
        track_id = df_i['track_id'][i]
        df_track_i = df[df['track_id'] == track_id]

        df_track_i = df_track_i.reset_index()
        df_track_i = df_track_i.drop(['index'], axis=1)   

        while len(df_track_i) > len(X) :
            min_dist = df_track_i['dist'].min()
            df_track_i = df_track_i[df_track_i['dist']!=df_track_i['dist'].min()]
            past_ind = -1
            
            indexs = []
            for ind in df_track_i.index:
                if past_ind - ind != -1 :
                    indexs.append([ind,past_ind])
                past_ind = ind

            # we recompute the distance by taking into acompte the deleted points
            df_track_i = df_track_i.reset_index()
            df_track_i = df_track_i.drop(['index'], axis=1)
            for inds in indexs:
                
                if inds[0]-1 < len(df_track_i):                    
                    df_track_i['dist'][inds[0]-1] = (((df_track_i['x'][inds[0]-1] - df_track_i['x'][inds[1]])**2+
                                             (df_track_i['y'][inds[0]-1] - df_track_i['y'][inds[1]])**2)**(1/2))

            past_len = len(df_track_i)
        
        #if during the last iteration more than 1 line as been deleted, the last line of X are deleted.
        diff = 0
        if len(X) !=  len(np.asarray(df_track_i['x'])):
            diff = len(X) - len(np.asarray(df_track_i['x']))
            X = X[0:-diff]
            Y = Y[0:-diff]
        X = np.add(X,np.asarray(df_track_i['x']))
        Y = np.add(Y,np.asarray(df_track_i['y']))

    X /= len(df_i)
    Y /= len(df_i)
    
    return(X, Y)

def first_dir(direction, df, df_point):

    # This function defines the positions of the trajectory of the first vehicle that passes through the direction as well as the positions of the direction system.
    df_i = df_point[df_point['direction'] == direction]
    df_i = df_i.reset_index()
    df_i = df_i.drop(['index'], axis=1)
    track_id = df_i['track_id'][0]
    df_track_i = df[df['track_id'] == track_id]
    X = np.asarray([df_track_i['x']]).reshape((-1,))
    Y = np.asarray([df_track_i['y']]).reshape((-1,))
    
    return X,Y

def get_dir(df, df_point, nb_dire, dire_type):
    # This function defines the positions of all direction.
    # df_dire contains : - the number of the direction which is the index (direction)
    #                    - all positions X (X)
    #                    - all Y-positions


    for ind in df_point.index:
        track_id = df_point['track_id'][ind]
        dire = df_point['direction'][ind]
        df.loc[df.track_id == track_id, 'direction'] = (dire)
    
    data = []
    if dire_type == 'mean_dir':
        for i in range(nb_dire):

            X, Y = mean_dir(i, df, df_point)
            data.append([i, X, Y])
        
    elif dire_type == 'first_dir':
        for i in range(nb_dire):
            X, Y = first_dir(i, df, df_point)
            data.append([i, X, Y])

    else:
        print(dire_type+'n est pas reconnue')

    df_dire = pd.DataFrame(data, columns=['direction', 'X', 'Y'])
    df_dire = df_dire.set_index('direction')
        
    return df_dire

def get_intersection(df, i, j, tol_rate, rmax):
    # This function classifies the interactions between two directions. If the directions are close for a short time  
    # the interaction is considered a crossover, if they are close for a long time the interaction is considered a junction

    # tol_rate represents the acceptable distance between two directions to consider an interaction. 
    # The greater tol_rate, the more interactions will be detected. (which may be false)

    #rmax represents the radius at which an interaction is considered to be a junction.
    #if rmax is very large, all interactions will be classified as crossover.

    #it may be interesting to vary tol_rate and rmax according to each case to be better adapted to the situation.

    # c: position of the intersection/junction
    # r: radius around which the crossover takes place.
    # inter_type: type of interaction between two direction (junction, intersection or None)

    X1 = df['X'][i]
    Y1 = df['Y'][i]
    X2 = df['X'][j]
    Y2 = df['Y'][j]
    
    max_x = max(max(X1), max(X2))
    max_y = max(max(Y1), max(Y2))
    min_x = min(min(X1), min(X2))
    min_y = min(min(Y1), min(Y2))

    ''' maxs_x = []
    mins_x = []
    maxs_y = []
    mins_y = []
    
    for ind in df.index:
        maxs_x.append(max(df['X'][ind]))
        mins_x.append(min(df['X'][ind]))
        maxs_y.append(max(df['Y'][ind]))
        mins_y.append(min(df['Y'][ind]))
        
    max_x = max(maxs_x)
    min_x = min(mins_x)
    max_y = max(maxs_y)
    min_y = min(mins_y)'''

    #We calculate the average density, which can be a good estimator in the maximum distance that there can be between two points.
    dens_X1 = (max(X1) - min(X1))/len(X1)
    dens_X2 = (max(X2) - min(X2))/len(X2)
    dist_min_X = max(dens_X1, dens_X2)
    
    dens_Y1 = (max(Y1) - min(Y1))/len(Y1)
    dens_Y2 = (max(Y2) - min(Y2))/len(Y2)
    dist_min_Y = max(dens_Y1, dens_Y2)

    pos_crois = [[],[]]
    pos_X = []
    pos_Y = []
    for i in range(len(X1)):
        for j in range(len(X2)):
            diff_x = abs(X1[i] - X2[j])
            diff_y = abs(Y1[i] - Y2[j])
            if diff_x < dist_min_X*tol_rate and diff_y < dist_min_Y*tol_rate:
                pos_crois[0].append(i)
                pos_crois[1].append(j)
                pos_X.append((X1[i] + X2[j])/2)
                pos_Y.append((Y1[i]+ Y2[j])/2)
    if pos_X:
        c = ((min(pos_X) + max(pos_X))/2, (min(pos_Y) + max(pos_Y))/2)
        r = (max((max(pos_X) - min(pos_X))/2, (max(pos_Y) - min(pos_Y))/2))
        inter_type = 'crossover'
        if r>rmax:
            r = None
            extr0 = [abs(pos_X[0]-min_x), abs(pos_X[0]-max_x), abs(pos_Y[0]-min_y), abs(pos_Y[0]-max_y)]
            extr1 = [abs(pos_X[-1]-min_x), abs(pos_X[-1]-max_x), abs(pos_Y[-1]-min_y), abs(pos_Y[-1]-max_y)]
            smallest_dist = min(min(extr0), min(extr1))
            if smallest_dist in extr0:
                c = (pos_X[-1], pos_Y[-1])
            else:
                c = (pos_X[0], pos_Y[0])
            inter_type = 'jonction'

    else:
        c = None
        r = None
        inter_type = None

    return c, r, inter_type
            
def directions(df, dire_type = 'mean_dir', max_frame=None, min_frame=None):

    # This function allows us to compute df_dire and update df by defining which direction 
    # belongs to each vehicle and retrieve df_dire 

    # We add to df_point: - the direction by which the vehicle passed (direction)

    # df_dire contains: - the number of the direction which is the index (direction)
    #                   - all positions X (X)
    #                   - all positions Y (Y)


    min_x = (df.min(axis = 0)['x'])
    min_y = (df.min(axis = 0)['y'])
    if max_frame is None or min_frame is None:
        max_frame = (df.max(axis = 0)['frame_id'])
        min_frame = (df.min(axis = 0)['frame_id'])

    vehicle_1 = df['track_id']
    vehicle_1 = list(dict.fromkeys(vehicle_1))
    nb_vehicle = len(vehicle_1)
     
    df_point, X_entry, X_exit = point(df, nb_vehicle, min_frame, max_frame)
    
    vehicle_2 = df_point['track_id']
    vehicle_2 = list(dict.fromkeys(vehicle_2))
    lost_vehicle1 = nb_vehicle - len(vehicle_2)

    model_entry, model_exit = dbscan_model(X_entry, X_exit)
    
    # We add to df_point the points at which each vehicle enters and exits    
    df_point['entry_label'] = model_entry.labels_.astype(int)
    df_point['exit_label'] = model_exit.labels_.astype(int)
    
    df_point, nb_dire = get_direction(df_point)
    
    # We add, on the main df, the direction to which each vehicle at
    # Every moment belongs.
    

    lost_vehicle2 = len(df_point[(df_point['direction'] == 'noise')])
    lost_rate = (lost_vehicle2 + lost_vehicle1)*100/nb_vehicle
    print(str((lost_vehicle2 + lost_vehicle1)*100/nb_vehicle) + ' % of lost vehciule including ' + str((lost_vehicle2/nb_vehicle)*100) + '% in clustering')


    df['direction'] = [None]*len(df)
    
    
    for ind in df_point.index:
        track_id = df_point['track_id'][ind]
        dire = df_point['direction'][ind]
        df.loc[df.track_id == track_id, 'direction'] = (dire)
       
        
    df_dire = get_dir(df, df_point, nb_dire, dire_type=dire_type)
    
    return df, df_dire, lost_rate

def intersections(df_dire, tol_rate=1.5, rmax=2):
    # This function calculates all possible intersections for a df_dire
    # df_inter contains: - the first direction (direction 1) 
    #                    - the second direction (direction 2) 
    #                    - the point of interaction, if applicable (center)
    #                    - the radius in case of an intersection (radius)
    #                    - the type of intersection: crossover, junction or None (type)
    # Please note that to avoid repetition, direction 1 < direction 2 
    data = []
    for i in range(len(df_dire)):
        for j in range(len(df_dire)):
            if i < j:
                c, r, inter_type = get_intersection(df_dire, i, j, tol_rate=tol_rate, rmax=rmax)
                data.append([i, j, c, r, inter_type])
    df_inter = pd.DataFrame(data, columns=['direction 1', 'direction 2', 'center', 'radius', 'type'])
    
    return df_inter

def crossovers(df_inter, df, dist_cross = 10):
    # This function allows us to identify when, two vehicles belonging to two directions that cross each other, are close to the intersection of these two directions.
    # dist_cross is the maximum distance from which a vehicle is considered to be close to the intersection. 
  

    # df_crossover contains: - identification of the 1st vehicle (track_id_1)
                           # - the direction to which the first vehicle belongs (direction 1)
                           # - the average speed at which the vehicle is close to the intersection (v_mean_1)
                           # - identification of the 2nd vehicle (track_id_2)
                           # - the direction to which the 2nd vehicle belongs (direction 2)
                           # - the average speed at which the vehicle is close to the intersection (v_mean_2)
                           # - the position of the crossover (position)
                           # - the beginning of the moment they cross (t_max)
                           # - the end of the moment they cross (t_min)
    data = []
    
    df_croisement = df_inter[df_inter['type'] == 'crossover']
    df_croisement = df_croisement.reset_index()
    df_croisement = df_croisement.drop(['index'], axis=1)
    
    for croisement in df_croisement.index:
    
        crois = df_croisement['center'][croisement]
        dire_1 = df_croisement['direction 1'][croisement]
        dire_2 = df_croisement['direction 2'][croisement]
    
        df_dire_1 = df[df['direction'] == dire_1]
        df_dire_2 = df[df['direction'] == dire_2]
        # all vehicles from direction 1,2 that are near the intersection in question 
        df_dire_1 = df_dire_1[(abs(df_dire_1['x']-crois[0]) < dist_cross) & (abs(df_dire_1['y']-crois[1]) < dist_cross)]
        df_dire_2 = df_dire_2[(abs(df_dire_2['x']-crois[0]) < dist_cross) & (abs(df_dire_2['y']-crois[1]) < dist_cross)]
    
    
        track_ids_1 = df_dire_1['track_id']
        track_ids_1 = list(dict.fromkeys(track_ids_1))
        for track_id_1 in track_ids_1:
    
            # for each direction 1 vehicle we look at the max and min time at which it approaches the intersection.    
            df_dire_1_i = df_dire_1[df_dire_1['track_id'] == track_id_1]
            t_min = (df_dire_1_i.min(axis = 0)['timestamp_ms'])
            t_max = (df_dire_1_i.max(axis = 0)['timestamp_ms'])
    
            df_dire_2_cross = df_dire_2[(df_dire_2['timestamp_ms'] < t_max) & (df_dire_2['timestamp_ms'] > t_min)]    
            track_ids_2 = df_dire_2_cross['track_id']
            track_ids_2 = list(dict.fromkeys(track_ids_2))
            for track_id_2 in track_ids_2:
    
                df_dire_2_crois_i = df_dire_2_cross[df_dire_2_cross['track_id'] == track_id_2]
                                
                df_dire_1_cross_i = df[df['direction'] == dire_1]
                df_dire_1_cross_i = df_dire_1_cross_i[(abs(df_dire_1_cross_i['x']-crois[0]) < dist_cross) & (abs(df_dire_1_cross_i['y']-crois[1]) < dist_cross)]
                df_dire_1_cross_i = df_dire_1_cross_i[(df_dire_1_cross_i['timestamp_ms'] < t_max) & (df_dire_1_cross_i['timestamp_ms'] > t_min)]
                df_dire_1_cross_i = df_dire_1_cross_i[df_dire_1_cross_i['track_id'] == track_id_1]

                t = []
                for ind1 in df_dire_1_cross_i.index:
                    for ind2 in df_dire_2_crois_i.index:
                        t1 = df_dire_1_cross_i['timestamp_ms'][ind1]
                        t2 = df_dire_2_crois_i['timestamp_ms'][ind2]
                        x1 = df_dire_1_cross_i['x'][ind1]
                        x2 = df_dire_2_crois_i['x'][ind2]
                        y1 = df_dire_1_cross_i['y'][ind1]
                        y2 = df_dire_2_crois_i['y'][ind2]
                        if (((x1-x2)**2 + (y1-y2)**2)**(1/2))<dist_cross and t1==t2:
                            t.append(t1)
                if t:
                    t_max = max(t)
                    t_min = min(t)
                    df_dire_1_cross_i = df[df['direction'] == dire_1]
                    df_dire_1_cross_i = df_dire_1_cross_i[(abs(df_dire_1_cross_i['x']-crois[0]) < dist_cross) & (abs(df_dire_1_cross_i['y']-crois[1]) < dist_cross)]
                    df_dire_1_cross_i = df_dire_1_cross_i[(df_dire_1_cross_i['timestamp_ms'] < t_max) & (df_dire_1_cross_i['timestamp_ms'] > t_min)]
                    df_dire_1_cross_i = df_dire_1_cross_i[df_dire_1_cross_i['track_id'] == track_id_1]
                    v_mean_1 = ((df_dire_1_cross_i.mean(axis = 0)['vx'])**2 + (df_dire_1_cross_i.mean(axis = 0)['vy'])**2)**(1/2)

                    df_dire_2_crois_i = df[df['direction'] == dire_2]
                    df_dire_2_crois_i = df_dire_2_crois_i[(abs(df_dire_2_crois_i['x']-crois[0]) < dist_cross) & (abs(df_dire_2_crois_i['y']-crois[1]) < dist_cross)]
                    df_dire_2_crois_i = df_dire_2_crois_i[(df_dire_2_crois_i['timestamp_ms'] < t_max) & (df_dire_2_crois_i['timestamp_ms'] > t_min)]
                    df_dire_2_crois_i = df_dire_2_crois_i[df_dire_2_crois_i['track_id'] == track_id_1]
                    v_mean_2 = ((df_dire_1_cross_i.mean(axis = 0)['vx'])**2 + (df_dire_1_cross_i.mean(axis = 0)['vy'])**2)**(1/2)

                    data.append([track_id_1, dire_1, v_mean_1, track_id_2, dire_2, v_mean_2, crois, t_min, t_max])

        
    
        
    df_crossover = pd.DataFrame(data, columns=['track_id_1', 'direction 1','v_mean_1', 'track_id_2', 
                                                'direction 2', 'v_mean_2', 'position', 't_min', 't_max'])
        
    return df_crossover
   
def slowdowns(df, direction, discr=20, slow_rate):
    #This function allows us to detect slowdowns in a direction

    #it calculates the average speed over n=discr^2 zones spread over all the trajectories that have the same direction.
    #For each vehicle at any moment we look at which zone it belongs to and compare its speed with the average speed (v_mean) of the zone it belongs to.
    #if v<slow_rate*v_mean we consider this vehicle to be unusually slow and classify it as slowdowns

    df_dir = df[df['direction'] == direction]


    min_x = df_dir.min(axis = 0)['x']
    max_x = df_dir.max(axis = 0)['x']
    dx = (max_x-min_x)/discr

    min_y = df_dir.min(axis = 0)['y']
    max_y = df_dir.max(axis = 0)['y']
    dy = (max_y-min_y)/discr

    mean_speed = np.zeros((discr,discr))
    for i in range(discr-1):
        for j in range(discr-1):
            df_dx_dy = df_dir[(df_dir['x'] >= min_x + i*dx) & (df_dir['x'] <= min_x + (i+1)*dx) 
                            & (df_dir['y'] >= min_y + j*dy) & (df_dir['y'] <= min_y + (j+1)*dy)]

            v_mean = (df_dx_dy.mean(axis = 0)['vx']**2 + df_dx_dy.mean(axis = 0)['vy']**2)**(1/2)

            mean_speed[i, j] = v_mean


    data = []
    for ind in df_dir.index:
        x = df_dir['x'][ind]
        y = df_dir['y'][ind]
        t = df_dir['timestamp_ms'][ind]
        v = (df_dir['vx'][ind]**2 + df_dir['vy'][ind]**2)**(1/2)
        track_id = df_dir['track_id'][ind]
        i = int((x-min_x)/dx)
        j = int((y-min_y)/dy)
        if i != discr and j != discr:
            v_mean = mean_speed[i, j]
            if v<v_mean*slow_rate:
                data.append([track_id, v, v_mean, t])


    df_slowdown = pd.DataFrame(data, columns=['track_id', 'v', 'v_mean', 't'])
    tracks = df_slowdown['track_id']
    tracks = list(dict.fromkeys(tracks))
    data = []
    for track in tracks:
        df_track = df_slowdown[df_slowdown['track_id'] == track]
        t_min = df_track.min(axis = 0)['t']
        t_max = df_track.max(axis = 0)['t']
        v = df_track.mean(axis = 0)['v']
        mean_v = df_track.mean(axis = 0)['v_mean']
        data.append([track, v, mean_v, t_min, t_max])
    df_slowdown = pd.DataFrame(data, columns=['track_id', 'v','v_mean', 't_min', 't_max'])
    
    return df_slowdown
    
    
    
    
    
    
    
    
        
        
        
        