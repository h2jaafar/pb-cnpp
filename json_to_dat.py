import json
from os import write
import numpy as np


def main():
    
    map_size = str(8)

    path_urf = '/home/hussein/Desktop/pb-cnpp/pb-cnpp/resources/jsons/test_maps/' + map_size + '/urf/'
    path_house = '/home/hussein/Desktop/pb-cnpp/pb-cnpp/resources/jsons/test_maps/' + map_size + '/house/'

    astar_paths_urf = '/home/hussein/Desktop/pb-cnpp/pb-cnpp/resources/jsons/test_maps/' + map_size + '/paths_urf.json'
    astar_paths_house = '/home/hussein/Desktop/pb-cnpp/pb-cnpp/resources/jsons/test_maps/' + map_size + '/paths_urf.json'

    save_path = './resources/dat_files/test_maps/' + map_size + '/'


    inputs = [] # Should be a list with each line as the map obstacles (maps are sequential row 1 row 2 row 3 all sequential )
    g_maps = [] # should be a list with each row, where 1 is the start location
    s_maps = []
    outputs = []
    nbr_maps = 1500

    outputs_urf_untouched = open_astar_paths(astar_paths_urf)
    outputs_house_untouched = open_astar_paths(astar_paths_house)
    for mp in range(nbr_maps-1):
        grid,goal,start = open_map(mp,path_urf)
        size =int(len(grid))
       

        grid = grid_cleanup(grid)

        goal_grid = point_to_grid(goal,size)
        goal_grid = grid_cleanup(goal_grid)

        start_grid = point_to_grid(start,size)
        start_grid = grid_cleanup(start_grid)

        trace = outputs_urf_untouched[mp]
        trace_grid = trace_cleanup(trace,size)
        trace_grid_clean = grid_cleanup(trace_grid)
        

        inputs.append(grid)
        g_maps.append(goal_grid)
        s_maps.append(start_grid)
        outputs.append(trace_grid_clean)

        write_to_dat(grid,str(save_path + 'inputs.dat'))
        write_to_dat(goal_grid,str(save_path + 'g_maps.dat'))
        write_to_dat(start_grid,str(save_path + 's_maps.dat'))
        write_to_dat(trace_grid_clean,str(save_path + 'outputs.dat'))

        ########################################################
        # House

        grid,goal,start = open_map(mp,path_house)
        size =int(len(grid))
       

        grid = grid_cleanup(grid)

        goal_grid = point_to_grid(goal,size)
        goal_grid = grid_cleanup(goal_grid)

        start_grid = point_to_grid(start,size)
        start_grid = grid_cleanup(start_grid)

        trace = outputs_house_untouched[mp]
        trace_grid = trace_cleanup(trace,size)
        trace_grid_clean = grid_cleanup(trace_grid)
        

        inputs.append(grid)
        g_maps.append(goal_grid)
        s_maps.append(start_grid)
        outputs.append(trace_grid_clean)

        write_to_dat(grid,str(save_path + 'inputs.dat'))
        write_to_dat(goal_grid,str(save_path + 'g_maps.dat'))
        write_to_dat(start_grid,str(save_path + 's_maps.dat'))
        write_to_dat(trace_grid_clean,str(save_path + 'outputs.dat'))


    print("Finished Processing")
        

def open_map(dom,path):
    '''
    Used to open a map json given dom and path, returns grid, goal and agent
    '''
    with open(str(path) + str(dom) +'.json') as json_file:
        data = json.load(json_file)
        print('Opening file: ' + str(path) + str(dom) + '.json' )
        return data['grid'], data['goal'], data['agent']

def open_astar_paths(path):
    '''
    Used to open a single json for astar paths
    '''
    with open(str(path)) as json_file:
        data = json.load(json_file)
        print('Opening file: ' + str(path))
        return data

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
    (like [0 0 0, 0 0 1, 0 0 0])  
    '''
    points_as_grid = np.zeros((size,size))

    points_as_grid[point[0],point[1]] = int(1)


    return points_as_grid.astype('int8')

def trace_cleanup(trace,size):
    '''
    Converts points in a trace to grid paths
    '''

    trace_grid = np.zeros((size,size))

    for point in trace:
        trace_grid[point[0],point[1]] = int(1)
    
    return trace_grid.astype('int8')



def write_to_dat(grid,path):

    '''
    writes objects to a dat file
    '''

    with open(str(path), 'a') as dat_file:
        dat_file.write(str(grid).strip('[]').replace('\'', '').replace(',', ''))
        dat_file.write("\n")

    print("Written to file", path)
    return


main()
