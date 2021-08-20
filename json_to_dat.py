import json
from os import write
import numpy as np


def main():
    global path
    path = '/home/hussein/Desktop/test_maps_WPN/training_maps/64x64_10k_urf/'
    save_path = '/home/hussein/Desktop/test_maps_WPN/training_maps/dat_files/test/test5.dat'
    inputs = [] # Should be a list with each line as the map obstacles (maps are sequential row 1 row 2 row 3 all sequential )
    g_maps = [] # should be a list with each row, where 1 is the start location
    s_maps = []

    nbr_maps = 10

    for mp in range(nbr_maps):
        grid,goal,start = open_map(mp,path)
        size =int(len(grid))
        grid = grid_cleanup(grid)

        goal_grid = point_to_grid(goal,size)
        goal_grid = grid_cleanup(goal_grid)

        start_grid = point_to_grid(start,size)
        start_grid = grid_cleanup(start_grid)

        inputs.append(grid)
        g_maps.append(goal_grid)
        s_maps.append(start_grid)

        write_to_dat(grid,save_path)

    print("Finished Processing")
        

def open_map(dom,path):
    '''
    Used to open a map json given dom and path, returns grid, goal and agent
    '''
    with open(str(path) + str(dom) +'.json') as json_file:
        data = json.load(json_file)
        print('Opening file: ' + str(path) + str(dom) + '.json' )
        return data['grid'], data['goal'], data['agent']

def grid_cleanup(grid):
    '''
    Clean up the grid objects. 
    Grid should be returned as a single list, not a list of lists
    '''

    grid_clean = [item for sublist in grid for item in sublist]
    
    return grid_clean

def point_to_grid(point,size):
    '''
    Converts a coordinate to a grid-based location
    Ex. [3,2]
    [0 0 0 0 0 0 1 0 0 0] (like [0 0 0, 0 0 1, 0 0 0]) but flattened  
    '''
    points_as_grid = np.zeros((size,size))

    points_as_grid[point[0],point[1]] = 1


    return points_as_grid

def write_to_dat(grid,path):

    '''
    writes objects to a dat file
    '''

    with open(str(path), 'a') as dat_file:
        dat_file.write(str(grid).strip('[]').replace('\'', '').replace(',', ''))
        dat_file.write("\n")

    print("Written to file", path)
    return

def write_cleanup(data,size):
    '''
    Removes brackets and commas, prepares for writing to file
    '''

main()
