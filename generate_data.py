import sys
import numpy as np
import json
import maketrack
import racetrack
import h3
from OutputConversionFunctions import *


def transform_data(state, f_line, walls):
    x_s,y_s = state[0]
    u_s,v_s = state[1]

    #calculate heuristic values for grid
    h_grid = np.array(h3.h_grid(state, f_line, walls))

    #calculate wall grid
    w_grid = np.zeros([29,29])
    for point in points_on_walls(walls):
        x,y = point
        w_grid[x][y] = 1.

    #calculate finish line grid
    f_grid = np.zeros([29,29])
    for point in points_on_edge(f_line):
        x,y = point
        f_grid[x][y] = 1.

    #transform input data
    state_input = np.array([])

    #calculate player position
    p_grid = np.zeros([29,29])
    p_grid[x_s][y_s] = 1.

    #player u-velocity
    u_grid = np.zeros([29,29])
    u_grid[x_s][y_s] = float(u_s)

    #player v-velocity
    v_grid = np.zeros([29,29])
    v_grid[x_s][y_s] = float(v_s)

    #add all input features to list
    input_grids = [w_grid, f_grid, p_grid, u_grid, v_grid]
    for grd in input_grids:
        state_input = np.append(state_input, grd.flatten())

    return state_input.tolist()


def generate_data(n=10, file_prefix='', draw=0, doprint=0):
    """
    For n number of times, create racetrack problems and generate data.
    Data: racetrack grid, current state: {position, velocity}
    Output data to file: "DATA/data.json"
    """
    input_data = []
    label_data = []

    #clear save files
    f = open("DATA/{}_data.json".format(file_prefix), 'w')
    f.close()
    f = open("DATA/{}_label.json".format(file_prefix), 'w')
    f.close()

    #for n randomly generated racetrack problem
    for i in range(n):
        title = 'problem {}'.format(i)
        problem = maketrack.main(doprint=doprint, draw=draw, title=title)
        state = (problem[0], (0,0))    # initial state
        f_line = problem[1]
        walls = problem[2]

        #calculate heuristic values for grid
        h_grid = np.array(h3.h_grid(state, f_line, walls))

        #calculate wall grid
        w_grid = np.zeros([29,29])
        for point in points_on_walls(walls):
            x,y = point
            w_grid[x][y] = 1.

        #calculate finish line grid
        f_grid = np.zeros([29,29])
        for point in points_on_edge(f_line):
            x,y = point
            f_grid[x][y] = 1.

        #generate list of states to finish line
        states = racetrack.main(problem, 'a*', h3.h_h2, verbose=0)

        #generate input data and labels
        for index, state in enumerate(states):
            x_s,y_s = state[0]
            u_s,v_s = state[1]

            """generate state input"""
            #transform input data
            state_input = np.array([])

            #calculate player position
            p_grid = np.zeros([29,29])
            p_grid[x_s][y_s] = 1.

            #player u-velocity
            u_grid = np.zeros([29,29])
            u_grid[x_s][y_s] = float(u_s)

            #player v-velocity
            v_grid = np.zeros([29,29])
            v_grid[x_s][y_s] = float(v_s)

            #add all input features to list
            input_grids = [w_grid, f_grid, p_grid, u_grid, v_grid]
            for grd in input_grids:
                state_input = np.append(state_input, grd.flatten())

            #generate state label
            state_label = np.zeros(9, dtype=bool)
            if (index+1) < len(states):
                #set state label to velocity as one-hot vector
                next_u, next_v = states[index+1][1]
                j = convertChangeInVelocityToIndex((next_u-u_s, next_v-v_s))
                state_label[j] = True
            else:
                #already at end state
                state_label[convertChangeInVelocityToIndex((0,0))] = True

            #append state input and label to lists
            input_data.append(state_input.tolist())
            label_data.append(state_label.tolist())

        #print for debugging purposes
        if draw:
            print(states)
            print(input_data)
            print(label_data)
            print("\n*** maketrack: finished drawing {}.".format(title), end=' ')
            print("Hit carriage return to continue.\n")
            sys.stdin.readline()

    #save input data and labels to file
    with open("DATA/{}_data.json".format(file_prefix), 'a') as f:
        json.dump(input_data, f)
        f.close()
    with open("DATA/{}_label.json".format(file_prefix), 'a') as f:
        json.dump(label_data, f)
        f.close()


def load_data(file_prefix=''):
    """return loaded input data and labels"""
    with open("DATA/{}_data.json".format(file_prefix), 'r') as f:
        input_data = json.load(f)
        f.close()
    with open("DATA/{}_label.json".format(file_prefix), 'r') as f:
        label_data = json.load(f)
        f.close()
    return (input_data, label_data)


def points_on_edge(edge):
    """return list of points on edge (assume horizontal or vertical)"""
    points = []
    pt1, pt2 = edge
    xmax = max(pt1[0], pt2[0])
    xmin = min(pt1[0], pt2[0])
    ymax = max(pt1[1], pt2[1])
    ymin = min(pt1[1], pt2[1])
    for x in range(xmin, xmax+1):
        for y in range(ymin, ymax+1):
            points.append((x,y))
    return points


def points_on_walls(walls):
    """return list of points on walls"""
    points = []
    for wall in walls:
        points.extend(points_on_edge(wall))
    return points


def goal_test(point,velocity,f_line):
	"""Test whether point is on the finish line and velocity is (0,0)"""
	return velocity == (0,0) and intersect((point,point), f_line)


def test():
    #data = generate_data(n=3, draw=0, doprint=0)
    data = load_data()
