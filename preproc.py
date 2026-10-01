import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import meshio

def input_data():
    """
    This function defines input parameters for the Boundary Element Method (BEM)
    simulation applied to the plane elasticity equation with radial integration for domain
    integrals.

    Returns:
        dict: A dictionary containing the following key-value pairs:
            * bound_cond (dict): Boundary conditions for each edge:
                *
            * E (float): Material elastic modulus
            * nu (float): Poisson's ratio
            * file_name (str): Name of the .msh file (Gmsh mesh)
            * qpoint (str): String representing the coordinates of the heat source point
    """
    # Directly define values here for scripting (modify as needed)
    # Boundary conditions are specified in local coordinate system (normal-tangent)
    # type_x = type of the boundary condition in x direction
    # type_y = type of the boundary condition in y direction
    #    (type_x and type_y: 0 = displacement is know while 1 = traction is known)

    bound_cond = {'botton': {'type_x': 1, 'value_x': [0,0], 'type_y': 0, 'value_y': [0,0]},
                  'right': {'type_x': 0, 'value_x': [0,0], 'type_y': 1, 'value_y': [0,0]},
        'top': {'type_x': 1, 'value_x': [0,0], 'type_y': 1, 'value_y': [100,0]},
        'left': {'type_x': 0, 'value_x': [0,0], 'type_y': 1, 'value_y': [0,0]}
    }
    E = 200.0e9 # # Elastic modulus
    nu = 0.3 # Poisson ration
    file_name = 'placa'
    rho = 7850 # density
    damp = 0.05 # damping

    return {'bound_cond': bound_cond, 'E': E, 'nu': nu, 'file_name': file_name,\
            'rho' : rho, 'damp' : damp}


def compute_inodes(file_name, bound_cond):
    """
    This function reads a mesh file and extracts information about nodes, elements,
    and boundary conditions.

    Args:
      file_name: Name of the mesh file.
      bound_cond: Dictionary containing boundary conditions for each edge.

    Returns:
      A dictionary containing information about nodes, elements, and boundary
      conditions.
    """
    mesh = meshio.read(file_name + '.msh')
    coordinates = mesh.points
    line_elements = mesh.cells_dict['line3']
    segments = mesh.cell_data_dict['gmsh:physical']['line3']

    bc_info = {}
    for key, value in bound_cond.items():
        segment = mesh.field_data[key][0]
        bc_info[key] = {
            'type_x': value['type_x'],
            'value_x': value['value_x'],
            'segment': segment,
            'type_y': value['type_y'],
            'value_y': value['value_y']
        }

    boundary_nodes = np.unique(line_elements)

    return {
        'nodes': boundary_nodes,
        'coordinates': coordinates,
        'elements': line_elements,
        'segments': segments,
        'bc_info': bc_info
    }


def compute_n(xi, x1, y1, x2, y2, x3, y3):
    """
    Computes the Jacobian, normal vector, and tangent vector at a point on the quadratic element.

    Args:
        xi (float): Gauss point in the local coordinate system.
        x1, y1, x2, y2, x3, y3: Coordinates of the nodes of the quadratic element.

    Returns:
        nx, ny (float): Components of the unit normal vector.
        dgamadxi (float): Magnitude of the Jacobian.
    """
    # Shape function derivatives
    dN1dxi = -1 / 2 + xi
    dN2dxi = -2 * xi
    dN3dxi = 1 / 2 + xi

    # Derivatives of x and y with respect to xi
    dxdxi = dN1dxi * x1 + dN2dxi * x2 + dN3dxi * x3
    dydxi = dN1dxi * y1 + dN2dxi * y2 + dN3dxi * y3

    # Jacobian magnitude
    dgamadxi = np.sqrt(dxdxi**2 + dydxi**2)

    # Tangent and normal vectors
    sx, sy = dxdxi / dgamadxi, dydxi / dgamadxi
    nx, ny = sy, -sx  # Rotate tangent by 90 degrees to get normal

    return nx, ny

def mount_bcs(segments, bc_info):
    """
    Constructs a boundary condition matrix from segment information and boundary condition data.

    This function takes a list of segment indices (`segments`) and a dictionary of boundary condition
    information (`bc_info`) to create a matrix (`bcs`) that stores the type and value of the boundary
    condition for each element.

    Args:
        segments (list or np.ndarray): A list or array of integers representing the indices of the boundary
                                       elements (segments) in the problem.
        bc_info (dict): A dictionary containing boundary condition information. The keys are typically
                        strings representing boundary condition names, and the values are dictionaries
                        with the following structure:
                        {
                            'type_x': bc_type,     # The type of boundary condition in x (0 for Dirichlet, 1 for Neumann)
                            'value_x': bc_value    # The value of the boundary condition in x (displacement or traction)
                            'segment': segment_index,  # The index of the segment this BC applies to
                            'type_y': bc_type,          # The type of boundary condition in y (0 for Dirichlet, 1 for Neumann)
                            'value_y': bc_value         # The value of the boundary condition in y (displacement or traction)
                        }

    Returns:
        bcs (np.ndarray): A 2D NumPy array where each row corresponds to a boundary element. The first
                         and third column contains the boundary condition type (0 or 1), and the second
                         and fourth column contains the boundary condition value.
    """

    num_elements = len(segments)

    # Initialize boundary condition matrix
    type_bcs = np.zeros((num_elements, 6), dtype=np.int64)
    val_bcs = np.zeros((num_elements, 6), dtype=np.complex64)

    # Populate boundary condition matrix (bcs)
    for i, segment_index in enumerate(segments):
        # Find the boundary condition data corresponding to the current segment
        for bc_name, bc_data in bc_info.items():
            if bc_data['segment'] == segment_index:
                tp_x=bc_data['type_x']
                vl_xr= bc_data['value_x'][0]
                vl_xi= bc_data['value_x'][1]
                tp_y=bc_data['type_y']
                vl_yr= bc_data['value_y'][0]
                vl_yi= bc_data['value_y'][1]
                vx = vl_xr + 1j * vl_xi
                vy = vl_yr + 1j * vl_yi
                type_bcs[i,:] = np.array([tp_x,tp_y,tp_x,tp_y,tp_x,tp_y])
                val_bcs[i,:] = np.array([vx,vy,vx,vy,vx,vy])
                break  # Move on to the next segment once the BC is found
    return type_bcs,val_bcs

def comp_normal(elem, nodes):
    """
    Calculates the nodes and outward normal vector for each boundary element.

    This function takes element connectivity information (`elem`) and node coordinates (`nodes`)
    to determine the geometric center of each boundary element and the unit normal vector
    pointing outward from that element.

    Args:
        elem (np.ndarray): A 2D array defining the connectivity of the boundary elements.
                          Each row contains the indices of the two nodes that form an element.
                          Shape: (num_elements, 2).
        nodes (np.ndarray): A 2D array containing the coordinates (x, y) of each node in the mesh.
                           Shape: (num_nodes, 2).

    Returns:
        node_des (np.ndarray): A 2D array containing the coordinates (x, y) of the discontinuous nodes
                              of each boundary element. Shape: (2*num_elements, 2).
        normal (np.ndarray): A 2D array containing the unit outward normal vectors for each
                             boundary element. Shape: (num_elements, 2).
    """
    npp = 8                              # Number of plot points
    eta = np.linspace(-1., 1., npp)        # Plot points

    num_elements = elem.shape[0]

    xplot = np.zeros((num_elements,npp))        # Store x-coordinates for plotting
    yplot = np.zeros((num_elements,npp))        # Store y-coordinates for plotting


    # Initialize arrays to store results

    normal = np.zeros((num_elements, 6))   # Outward normal vectors

    # Define the local coordinates (xi) for nodes within a quadratic element
    xi = np.array([-1., 0., 1.])  # Gauss-Lobatto points for quadratic elements

    # Iterate over each boundary element
    for i in range(num_elements):
        # Get node indices for the current element
        node1, node3, node2 = elem[i]

        # Get coordinates of the two nodes forming the element
        x1, y1 = nodes[node1][0:2]
        x2, y2 = nodes[node2][0:2]
        x3, y3 = nodes[node3][0:2]

        # Calculate and store the normal at the first node (xi = -2/3)
        nx1, ny1 = compute_n(-1., x1, y1, x2, y2, x3, y3)


        # Calculate and store the normal at the second node (xi = 0)
        nx2, ny2 = compute_n(0, x1, y1, x2, y2, x3, y3)


        # Calculate and store the normal at the third node (xi = 1)
        nx3, ny3 = compute_n(1, x1, y1, x2, y2, x3, y3)


        normal[i , :] = [nx1, ny1, nx2, ny2, nx3, ny3]

        # Iterate over Gauss points for numerical integration
        for k in range(npp):
            # Shape functions and their derivatives
            N1 = 0.5 * eta[k] * (eta[k] - 1)
            N2 = 1 - eta[k]**2
            N3 = 0.5 * eta[k] * (eta[k] + 1)

            # Calculate coordinates and normal vector at the Gauss point
            x, y = N1 * x1 + N2 * x2 + N3 * x3, N1 * y1 + N2 * y2 + N3 * y3
            ivet = npp * i + k  # Index for storing plot coordinates
            xplot[i,k], yplot[i,k] = x, y

    return normal,xplot,yplot

def show_geometry(nodes,xplot,yplot,elem):
    xnode=nodes[:,0]
    ynode=nodes[:,1]
    ax = plt.axes()
    plt.plot(xnode[elem[:,0]],ynode[elem[:,0]],"gd",markersize=8)	# Plot the node of the elements
    plt.plot(xnode,ynode,"ro",markersize=2)	# Plot the node of the elements
    plt.axis("equal")

    n_el = elem.shape[0]
    for i in range(n_el):
        x=xplot[i,:]
        y=yplot[i,:]
        line_segments = LineCollection([list(zip(x, y))],
                                        linestyles='solid')
        ax.add_collection(line_segments)

    plt.savefig("geometry.png")

def create_input_file(filename, computed_data, int_points, normal_vectors, type_bcs, val_bcs, input_params, frequencies):
    """
    Creates an input file for the Fortran BEM program based on computed mesh data
    and boundary conditions.
    """
    with open(filename, 'w') as f:
        # 1. Title Line
        f.write("SQUARE PLATE UNDER TIME-HARMONIC TRACTION (4 QUADRATIC ELEMENTS)\n")

        # Extract data from computed_data and input_params
        nodes = computed_data['coordinates']
        elements = computed_data['elements']
        num_elements = elements.shape[0]
        num_internal_points = int_points.shape[0]

        # Frequencies (hardcoded as per example squa4.dat)
        num_frequencies = len(frequencies)

        shear_modulus_real = input_params['E'] / (2 * (1 + input_params['nu']))
        damping_ratio = input_params['damp']
        density = input_params['rho']
        poisson_ratio = input_params['nu']

        # 2. Basic Parameter Line (formatted to match squa4.dat)
        f.write(f" {num_elements},{num_internal_points},{num_frequencies},{shear_modulus_real:.1f},{damping_ratio:.2f},{density:.1f},{poisson_ratio:.2f}\n")

        # 3. Frequencies Line
        f.write(" ")
        f.write(" ".join(f"{freq:.1f}" for freq in frequencies))
        f.write("\n")

        # 4. Boundary Nodes Coordinates Lines (dynamically writing all nodes)
        f.write(" ")
        node_coords_str = " ".join(f"{coord[0]:.1f}  {coord[1]:.1f}" for coord in nodes)
        f.write(node_coords_str + "\n")

        # 5. Boundary Conditions Lines
        for i in range(num_elements):
            # X direction BCs for 3 nodes
            f.write(f" {type_bcs[i, 0]} ({val_bcs[i, 0].real:.1f},{val_bcs[i, 0].imag:.1f})   ")
            f.write(f"{type_bcs[i, 2]} ({val_bcs[i, 2].real:.1f},{val_bcs[i, 2].imag:.1f})   ")
            f.write(f"{type_bcs[i, 4]} ({val_bcs[i, 4].real:.1f},{val_bcs[i, 4].imag:.1f})\n")

            # Y direction BCs for 3 nodes
            f.write(f" {type_bcs[i, 1]} ({val_bcs[i, 1].real:.1f},{val_bcs[i, 1].imag:.1f})   ")
            f.write(f"{type_bcs[i, 3]} ({val_bcs[i, 3].real:.1f},{val_bcs[i, 3].imag:.1f})   ")
            f.write(f"{type_bcs[i, 5]} ({val_bcs[i, 5].real:.1f},{val_bcs[i, 5].imag:.1f})\n")

        # 6. Internal Points Coordinates Lines (all on one line, formatted to match squa4.dat)
        int_points_str = " ".join(f"{p[0]:.1f}  {p[1]:.1f}" for p in int_points)
        f.write(" " + int_points_str + "\n")
        f.write("\n") # Add one more newline as seen in example










# Read information about input data
inp_data = input_data()


# Format input data
computed_data = compute_inodes(inp_data['file_name'], inp_data['bound_cond'])


normal,xplot,yplot = comp_normal(computed_data['elements'], computed_data['coordinates'])


# Generate bcs array with boundary conditions in each element
type_bcs,val_bcs = mount_bcs(computed_data['segments'], computed_data['bc_info'])


frequencies = [10., 40., 70.]

show_geometry(computed_data['coordinates'],xplot,yplot,computed_data['elements'],)

# Re-get the global `inp_data` dictionary to avoid shadowing
global_input_data = input_data()

inp_data_filename='squa4.dat' # This is the output file name
int_points=np.array([[3.,1.],[3.,2.],[3.,3.],[3.,4.],[3.,5.]])

frequencies = [10., 40., 70.]

import numpy as np

# Re-get the global `inp_data` dictionary to avoid shadowing
global_input_data = input_data()

inp_data_filename='squa4.dat' # This is the output file name
int_points=np.array([[3.,1.],[3.,2.],[3.,3.],[3.,4.],[3.,5.]])


# Call the function with the correct arguments
create_input_file(inp_data_filename, computed_data, int_points, normal, type_bcs, val_bcs, global_input_data, frequencies)
