import numpy as np
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
from PVBM.helpers.tortuosity import compute_tortuosity
from PVBM.helpers.perimeter import compute_perimeter_
from PVBM.helpers.branching_angle import compute_angles_dictionary
#from PVBM.helpers.far import far
from PVBM.GraphRegularisation.GraphRegularisation import TreeReg
import matplotlib.pyplot as plt
from scipy.ndimage import label

import sys
sys.setrecursionlimit(5000)

class GeometricalVBMs:
    """A class that can perform geometrical biomarker computation for a fundus image.
    """

    def extract_subgraphs(self,graphs, x_c, y_c, iterative_or_recursive='iterative'):
        """
        Extract B, the a graph where each of the disconnected subgraph is labeled differently and D which contains the euclidian distance graph between each
        point in the graph and the optic disc.

        :param graphs: Original blood vessel segmentation graph
        :type graphs: array
        :param x_c: x axis of the optic disc center
        :type x_c: int
        :param y_c: y axis of the optic disc center
        :type y_c: int
        :param iterative_or_recursive: Choose between 'iterative' or 'recursive' implementation of the subgraph extraction
        :type iterative_or_recursive: str
        :return: B,D
        :rtype: tuple
        """
        B = np.zeros_like(graphs, dtype=np.float32)
        D = np.zeros_like(graphs, dtype=np.float32)
        n = 1
        for i in range(graphs.shape[0]):
            for j in range(graphs.shape[1]):
                if B[i, j] == 0 and graphs[i, j] == 1:
                    self.recursive_subgraph(graphs, B, D, i, j, n, x_c, y_c)
                    n += 1
        return B, D
    

    def recursive_subgraph(self,A, B, D, i, j, n, x_c, y_c):
        """
        Recursively extract the value within B and D.

        :param A: Original blood vessel segmentation graph
        :type A: array
        :param B: A graph where each of the disconnected subgraph is labeled differentl, which is initialized by a zeros matrix and recursively built
        :type B: array
        :param D: Euclidian Distance graph between each point in A and the optic disc center (x_c,y_c), which is initialized by a zeros matrix and recursively built
        :type D: array
        :param i: Current x axis location within the graph
        :type i: int
        :param j: Current y axis location within the graph
        :type j: int
        :param n: Current number of point distance since the optic disc
        :type n: int
        :param x_c: x axis of the optic disc center
        :type x_c: int
        :param y_c: y axis of the optic disc center
        :type y_c: int

        :return: B,D
        :rtype: tuple
        """
        up = (i - 1, j)
        down = (i + 1, j)
        left = (i, j - 1)
        right = (i, j + 1)

        up_left = (i - 1, j - 1)
        up_right = (i - 1, j + 1)
        down_left = (i + 1, j - 1)
        down_right = (i + 1, j + 1)
        points = [up, down, left, right, up_left, up_right, down_left, down_right]
        for point in points:
            if point[0] >= 0 and point[0] < B.shape[0] and point[1] < B.shape[1] and point[1] >= 0:
                if A[point] == 1:
                    B[point] = n
                    A[point] = 0
                    D[point] = ((y_c - point[0]) ** 2 + (x_c - point[1]) ** 2) ** 0.5
                    self.recursive_subgraph(A, B, D, point[0], point[1], n, x_c, y_c)

    #Moved to iterative because recursive leads to stack overflow in c
    def iterative_topology(
            self, A, B, i, j, n, max_radius, x_c, y_c, endpoints, interpoints,
            i_or, j_or, dico, bacount, bapos, dist=0
    ):
        """
        Iteratively compute and analyze the topology of a segmented image using a stack.
        """
        # Initialize the stack with the initial node's state
        stack = []
        initial_frame = {
            'i': i,
            'j': j,
            'n': n,
            'i_or': i_or,
            'j_or': j_or,
            'bacount': bacount,
            'bapos': bapos,
            'dist': dist,
            'state': 'process_node'  # Possible states: 'process_node', 'process_children'
        }
        stack.append(initial_frame)
        A[i, j] = 0  # Mark the starting node as visited

        while stack:
            # Peek at the last frame on the stack
            frame = stack[-1]

            current_i = frame['i']
            current_j = frame['j']
            current_n = frame['n']
            current_i_or = frame['i_or']
            current_j_or = frame['j_or']
            current_bacount = frame['bacount']
            current_bapos = frame['bapos']
            current_dist = frame['dist']
            state = frame['state']

            if state == 'process_node':
                # print(
                #     f"Processing node: i={current_i}, j={current_j}, i_or={current_i_or}, j_or={current_j_or}, dist={current_dist},n={current_n} \n")

                # Calculate the Euclidean distance from the center
                distance_from_center = ((y_c - current_i) ** 2 + (x_c - current_j) ** 2) ** 0.5

                # Base Case 1: If beyond the allowed radius
                if distance_from_center > max_radius:
                    endpoints[current_i, current_j] = 1
                    true_distance = ((current_i_or - current_i) ** 2 + (current_j_or - current_j) ** 2) ** 0.5
                    dico[(current_i_or, current_j_or, current_i, current_j)] = (
                        current_dist,
                        true_distance,
                        true_distance / current_dist if current_dist != 0 else float('inf'),
                        current_bapos
                    )
                    stack.pop()  # Remove frame from stack
                    continue  # Proceed to next frame

                # Define all 8 neighbors and their corresponding distances
                up = (current_i - 1, current_j)
                down = (current_i + 1, current_j)
                left = (current_i, current_j - 1)
                right = (current_i, current_j + 1)
                up_left = (current_i - 1, current_j - 1)
                up_right = (current_i - 1, current_j + 1)
                down_left = (current_i + 1, current_j - 1)
                down_right = (current_i + 1, current_j + 1)
                points = [up, down, left, right, up_left, up_right, down_left, down_right]
                distances = [1, 1, 1, 1, 2 ** 0.5, 2 ** 0.5, 2 ** 0.5, 2 ** 0.5]

                # Compute the number of children
                children = 0
                valid_children = []
                child_distances = []
                for point, distance in zip(points, distances):
                    pi, pj = point
                    if 0 <= pi < A.shape[0] and 0 <= pj < A.shape[1]:
                        if A[pi, pj] == 1:
                            children += 1
                            valid_children.append(point)
                            child_distances.append(distance)

                # Store valid children and distances in the frame
                frame['valid_children'] = valid_children
                frame['child_distances'] = child_distances
                frame['child_index'] = 0  # Index of next child to process

                # Base Case 2: No children and sufficient depth
                if children == 0 and current_n >= 10:
                    endpoints[current_i, current_j] = 1
                    true_distance = ((current_i_or - current_i) ** 2 + (current_j_or - current_j) ** 2) ** 0.5
                    dico[(current_i_or, current_j_or, current_i, current_j)] = (
                        current_dist,
                        true_distance,
                        true_distance / current_dist if current_dist != 0 else float('inf'),
                        current_bapos
                    )
                    stack.pop()  # Remove frame from stack
                    continue

                # Base Case 3: More than one child and sufficient depth
                if children > 1 and current_n >= 10:
                    interpoints[current_i, current_j] = 1
                    true_distance = ((current_i_or - current_i) ** 2 + (current_j_or - current_j) ** 2) ** 0.5
                    dico[(current_i_or, current_j_or, current_i, current_j)] = (
                        current_dist,
                        true_distance,
                        true_distance / current_dist if current_dist != 0 else float('inf'),
                        current_bapos
                    )
                    # Reset variables for this frame (affects only its children)
                    frame['i_or'] = current_i
                    frame['j_or'] = current_j
                    frame['dist'] = 0
                    frame['n'] = 0
                    frame['bacount'] = 0
                    frame['bapos'] = None

                # Set state to 'process_children' to begin processing children
                frame['state'] = 'process_children'

            elif state == 'process_children':
                # Get the list of valid children and current child index
                valid_children = frame['valid_children']
                child_distances = frame['child_distances']
                child_index = frame['child_index']

                if child_index >= len(valid_children):
                    # All children have been processed, pop the frame
                    stack.pop()
                    continue

                # Get the next child to process
                point = valid_children[child_index]
                distance = child_distances[child_index]
                pi, pj = point

                # Increment child index in the parent frame
                frame['child_index'] += 1

                # Mark child as visited
                A[pi, pj] = 0

                # Update backup position if bacount reaches 30
                child_bacount = frame['bacount'] + 1
                child_bapos = frame['bapos']
                if child_bacount == 30:
                    child_bapos = (pi, pj)

                # Update cumulative distance
                child_dist = frame['dist'] + distance

                # Create a new frame for the child node
                child_frame = {
                    'i': pi,
                    'j': pj,
                    'n': frame['n'] + 1,
                    'i_or': frame['i_or'],
                    'j_or': frame['j_or'],
                    'bacount': child_bacount,
                    'bapos': child_bapos,
                    'dist': child_dist,
                    'state': 'process_node'
                }

                # Push the child frame onto the stack
                stack.append(child_frame)

        return


    def apply_roi(self, segmentation, skeleton, zones_ABC, roi):
        """
        Apply a region of interest (ROI) mask to the segmentation and skeleton images.

        :param segmentation: The segmentation image containing binary values within {0, 1}.
        :type segmentation: np.array
        :param skeleton: The skeleton image containing binary values within {0, 1}.
        :type skeleton: np.array
        :param zones_ABC: A mask image used to exclude specific zones, where the second channel defines the exclusion areas.
        :type zones_ABC: np.array
        :param roi: The region of interest mask, where the second channel defines the ROI areas.
        :type roi: np.array

        :return: A tuple containing:
            - The modified segmentation image with the ROI applied.
            - The modified skeleton image with the ROI applied.
        :rtype: Tuple[np.array, np.array]
        """
        segmentation_roi = segmentation * (1 - zones_ABC[:, :, 1] / 255)
        segmentation_roi = segmentation_roi * roi[:, :, 1] / 255

        skeleton_roi = skeleton * (1 - zones_ABC[:, :, 1] / 255)
        skeleton_roi = skeleton_roi * roi[:, :, 1] / 255

        return segmentation_roi, skeleton_roi


    def compute_geomVBMs(self,
                         blood_vessel: np.ndarray, 
                         skeleton: np.ndarray,
                         xc: int,
                         yc: int,
                         radius: int,
                         iterative_or_recursive: str = 'recursive',
                         interpoints_only: bool = False):
        """
        Compute various geometrical vascular biomarkers (VBMs) for a given blood vessel graph.

        This function analyzes the blood vessel segmentation and skeleton to extract several biomarkers such as area,
        tortuosity index, median tortuosity, overall length, median branching angle, and counts of start, end, and
        intersection points. It also provides visualizations of specific points on the graph.

        :param blood_vessel: Blood vessel segmentation containing binary values within {0,1}.
        :type blood_vessel: np.array
        :param skeleton: Blood vessel segmentation skeleton containing binary values within {0,1}.
        :type skeleton: np.array
        :param xc: X-axis coordinate of the optic disc center.
        :type xc: int
        :param yc: Y-axis coordinate of the optic disc center.
        :type yc: int
        :param radius: Radius in pixels of the optic disc.
        :type radius: int
        :param iterative_or_recursive: Choose between 'iterative' or 'recursive' implementation of the subgraph extraction
        :type iterative_or_recursive: str
        :param interpoints_only: If True, only the interpoints visualization will be returned.
        :type interpoints_only: bool

        :return: A tuple containing:
            - A list of biomarkers [area, tortuosity index, median tortuosity, overall length, median branching angle, number of start points, number of end points, number of intersection points].
            - A tuple of visualizations (endpoints, interpoints, startpoints, angles_dico, dico).
        :rtype: Tuple[list, tuple]
        """

        assert iterative_or_recursive in ['iterative', 'recursive'], "iterative_or_recursive must be either 'iterative' or 'recursive'"

        ####Compute the area
        area = np.sum(blood_vessel)

        ## Extract the distances graphs
        try:
            # print(xc, yc)
            # #Extract all the info about skeleton
            # print(f"Skeleton shape: {skeleton.shape}")
            # print(f"Skeleton dtype: {skeleton.dtype}")
            # print(f"Skeleton unique values: {np.unique(skeleton)}")
            # print(f"Skeleton sum: {np.sum(skeleton)}")
            # print(f"Skeleton nonzero count: {np.count_nonzero(skeleton)}")
            # labeled, n_components = label(skeleton)
            # print("Number of connected components:", n_components)
            B, D = self.extract_subgraphs(graphs=skeleton.copy(), x_c=xc, y_c=yc)
        
        except:
            return [area, None, None, None, None, None, None, None, None], (None, None, None, None, None)

        ## Extract the starting points list by navigating through the skeleton graph
        starting_points = np.zeros((skeleton.shape[0], skeleton.shape[1]), dtype=float)
        for i in set(list(B.reshape(-1))) - {0}:
            mask = B == i
            if mask.sum() >= 50:
                min_index = (D * mask + (1 - mask) * 1e10).argmin()
                min_coordinates = np.unravel_index(min_index, D.shape)
                # print(((min_coordinates[0] - yc)**2 + (min_coordinates[1] - xc)**2)**0.5)
                if ((min_coordinates[0] - yc) ** 2 + (min_coordinates[1] - xc) ** 2) ** 0.5 < 100 + 1 * radius:
                    starting_points[min_coordinates[0], min_coordinates[1]] = 1
        starting_point_list = np.argwhere(starting_points == 1)


        ### Cleaning the skeleton graph irregularities
        B = np.zeros((blood_vessel.shape[0], blood_vessel.shape[1]))
        tree_reg_list = []
        plot = np.zeros((blood_vessel.shape[0], blood_vessel.shape[1]))
        #try:
        for idx_start in starting_point_list:
            tree = TreeReg(idx_start[0], idx_start[1])
            tree.recursive_reg(skeleton.copy(), idx_start[0], idx_start[1], 0, tree, plot)
            tree_reg_list.append(tree)

        #except:
        #    return [area, None, None, None, None, None, None, None, None], (None, None, None, None, None)
        
        plot_list = []
        for tree_reg in tree_reg_list:
            plot = np.zeros((blood_vessel.shape[0], blood_vessel.shape[1]))
            if iterative_or_recursive == 'recursive':
                tree_reg.print_reg(tree_reg, plot)
            elif iterative_or_recursive == 'iterative':
                print_reg_iterative(tree_reg, plot)
            plot_list.append(plot.copy())
        skoustideB_reg = np.sum(np.array(plot_list), axis=0)

        #####Initialise the endpoints, interpoints and startpoints array that will be used later for visualization
        endpoints = np.zeros((blood_vessel.shape[0], blood_vessel.shape[1]))
        interpoints = np.zeros((blood_vessel.shape[0], blood_vessel.shape[1]))
        startpoints = np.zeros((blood_vessel.shape[0], blood_vessel.shape[1]))

        #####Initialising a dictionary that will be used to store the topology of th graph
        dico = {}

        ####Navigating through the graph to fill the end,inter and startpoints array and the topology dico
        for idx_start in starting_point_list:
            # print("new index")
            # print(t)
            i, j = idx_start
            startpoints[i, j] = 1
            self.iterative_topology(skoustideB_reg.copy(), B, idx_start[0], idx_start[1], 1, np.inf, xc, yc, endpoints,
                            interpoints, i, j, dico, 0, None)
            
        if interpoints_only:
            return interpoints

        #### Extracting the tortuosity and length using the topology of the graph
        chord_list = np.array([val[1] for val in dico.values()])
        arc_list = np.array([val[0] for val in dico.values()])
        TI = arc_list.sum() / chord_list.sum()
        medTor = np.median(arc_list / chord_list)
        ovlen = np.sum(arc_list)
        
        #Length-weighted global tortuosity
        arc = arc_list.astype(np.float64)
        chord = chord_list.astype(np.float64)
        segment_tortuosity = arc / chord
        T_weighted = np.sum(arc * segment_tortuosity) / np.sum(arc)

        ####Filtering the potential double angles for the branching angles computations
        angles_dico = {}
        s = dico.keys()
        v = list(dico.values())
        for element, val in zip(s, v):
            angles_dico[(element[0], element[1])] = angles_dico.get((element[0], element[1]), []) + [val[3]]

        ####Storing all the branching angles value
        angles = []
        for key, value in angles_dico.items():
            b = key
            if len(value) == 2:
                a, c = value[0], value[1]
                if all(x is not None for x in (a, b, c)):
                    ba = np.array(a) - np.array(b)
                    bc = np.array(c) - np.array(b)
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.arccos(cosine_angle)
                    angles.append(np.degrees(angle))
            elif len(value) == 3:
                a, c, d = value
                if all(x is not None for x in (a, b, c, d)):
                    ba = np.array(a) - np.array(b)
                    bc = np.array(c) - np.array(b)
                    cosine_angle_ac = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle_ac = np.arccos(cosine_angle_ac)
                    angles.append(np.degrees(angle_ac))

                    bc = np.array(c) - np.array(b)
                    bd = np.array(d) - np.array(b)
                    cosine_angle_cd = np.dot(bc, bd) / (np.linalg.norm(bc) * np.linalg.norm(bd))
                    angle_cd = np.arccos(cosine_angle_cd)
                    angles.append(np.degrees(angle_cd))

        ####Computing the median branching angles, the number of start/inter/endpoints
        medianba = np.median(angles)
        startp = len(starting_point_list)
        endp = endpoints.sum()
        interp = interpoints.sum()

        #### Return the biomarkers as well as the particular points visualisations
        print('i got here somehow')
        return [area, TI, medTor, T_weighted, ovlen, medianba, startp, endp, interp], (endpoints,interpoints,startpoints, angles_dico, dico)
    

def print_reg_iterative(tree, plot, min_size=10):
        """
        Iterative version of TreeReg.print_reg to avoid recursion depth overflow.
        Preserves original logic exactly.
        """

        # Stack entries: (node, visited_flag)
        stack = [(tree, False)]
        subtree_sizes = {}

        while stack:
            node, visited = stack.pop()

            if not visited:
                # First time: push back as visited, then children
                stack.append((node, True))
                for child in node.children:
                    stack.append((child, False))
            else:
                # Post-order: children already processed
                if len(node.children) == 0:
                    n = 1
                else:
                    n = 1 + sum(subtree_sizes[child] for child in node.children)

                subtree_sizes[node] = n

                if n >= min_size:
                    plot[node.plot] = 1

        return subtree_sizes[tree]