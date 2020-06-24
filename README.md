# Traffic-Analysis

:warning: [Work in progress]

**The purpose of the trajdect package is to analyze the data from the data set INTERACTION-Dataset-TC-v1_0**

In this document I use the term 'trajectory' to define all the positions taken by a single vehicle.
The term 'direction' denotes a set of trajectories that have the same entry and exit point.

four main function has been defined: 

## directions:

`df, df_dire, lost_rate = trajdect.directions(df, dire_type = 'mean_dir', max_frame=None, min_frame=None)`

Direction takes as input df, df is the data frame from the .csv of the data_set INTERACTION-Dataset-TC-v1_0.
It is possible to use more than one .csv at the same time. In this case, you have to merge them and renumber the `track_id` column so that there is no repetition within the df.
If you want to use a df composed of several .csv, you will have to give as input `max_frame`, a list with the maximum frame of each .csv and `min_frame`, composed of the minimum frame of each csv.


This function will: 
1. assign to each vehicle the direction it belongs to according to its trajectory and add it to df.
2. define the positions for each direction, and store them in df_dire. Two functions are available:
    - mean_traj will compute an average of all trajectories (more representative of all trajectories but slower to compute)
    - first_traj will take the first trajectory (fast to compute but not very representative of all trajectories)
3. lost_rate: the rate of vehicles present in the initial df but which were not taken into account. there are two reasons why they were not taken into account:
    - their entry point was on the first frame or their exit point was on the last frame.
    - the clustering algorithm failed to classify their entry or exit point.

exemple with DR_USA_Intersection_GL/vehicle_tracks_004.csv, all the direction plot, all the trajectories scatter:

![alt text](IMAGE/drection_trajectories.png =250x250)

## directions:


