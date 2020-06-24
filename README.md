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

exemple with DR_USA_Intersection_GL/vehicle_tracks_022.csv, mean_traj, all the direction plot, all the trajectories scatter:

<img src="IMAGE/traj_dire.png" width=800 height=400>

## intersections:

`df_inter = intersections(df_dire, tol_rate=1.5, rmax=2)`

The purpose of this function is to find the interactions between the different directions.
The interactions are classified into three types:
- `None`: no interaction between the two trajectories.
- `junction`: the trajectories merge at a certain point
- `crossing`: trajectories cross each other 

intersections take as input : 
- `df_dire`: compute with `directions`.
- `tol_rate`: represents the acceptable distance between two directions to consider an interaction. The greater tol_rate, the more interactions will be detected. (which may be false)
- `rmax` represents the radius at which an interaction is considered to be a junction. If rmax is very large, all interactions will be classified as crossing.

and will compute `df_inter`
df_inter contains: - the first direction `direction 1`
                   - the second direction `direction 2` 
                   - the point of interaction, if applicable `center`
                   - the radius in case of an intersection `radius`
                   - the type of intersection: `crossing`, `junction` or `None`
Note that to avoid repetition, `direction 1` < `direction 2`

