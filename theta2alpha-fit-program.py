import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from matplotlib.widgets import TextBox, Button
from utils import array_to_header_file, map_theta_to_si
from LookUpTableGen import generateTable, N_SMP
import warnings

# Define the order of the polynomial
poly_order = 5

# Initialize data points list
coords = []
slice_out = []

# Outputs
fitted_coefs = []
theta_fit = []
alpha_fit = []

# Get lookip table data
frame, table = generateTable()

# Clear data for 3d plots
CLEAR = np.zeros((10,10))

# Polynomial function expecting normalized theta
def poly_func(theta_norm, coeffs):
    return np.polyval(coeffs[::-1], theta_norm)

# Function to calculate the derivative of the polynomial
def poly_deriv(theta_norm, coeffs):
    deriv_coeffs = np.polyder(coeffs[::-1]) 
    return np.polyval(deriv_coeffs, theta_norm)

# Function to calculate distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Callback function for mouse click event
def onclick(event):
    global coords, slice_out
    if event.inaxes == ax1:
        click_coords = (event.xdata, event.ydata)

        # Define the bounds of the graph
        x_bounds = ax1.get_xlim()
        y_bounds = ax1.get_ylim()

        # Check if the click is within the bounds
        if (x_bounds[0] <= click_coords[0] <= x_bounds[1]) and (y_bounds[0] <= click_coords[1] <= y_bounds[1]):
            threshold = 10  # Adjust threshold based on your plot scale
            close_points = [p for p in coords if distance(p, click_coords) < threshold]
        
            if close_points:
                # Remove the closest point
                coords.remove(close_points[0])
            else:
                # Add a new point
                coords.append(click_coords)

        # Clear slice points
        slice_out = []
        # Update the plot
        update_plot()

# Function to update the plot with current coords and fitted curve
def update_plot():
    #if len(coords) < 2:
    #    return
    
    # Suppress all warnings (use with caution!)
    warnings.filterwarnings("ignore")

    if len(coords) < 1:
        line.set_data([], [])
        fitted_line.set_data([], [])
        slice_points2d.set_data([],[])
        slice_points3d._offsets3d = ([],[],[])
        plt.draw()
        return

    x_data, y_data = zip(*coords)
    x_data = np.array(x_data) / 360  # Normalize the data
    y_data = np.array(y_data)

    # Constraints for continuity and tangent
    def constraints(coeffs):
        poly_start = poly_func(0, coeffs)
        poly_end = poly_func(1, coeffs)
        deriv_start = poly_deriv(0, coeffs)
        deriv_end = poly_deriv(1, coeffs)
        return [poly_start - poly_end, deriv_start - deriv_end]

    # Objective function (sum of squared errors)
    def objective(coeffs):
        return np.sum((poly_func(x_data, coeffs) - y_data) ** 2)

    # Initial guess for coefficients
    coeffs_guess = np.random.rand(poly_order + 1)

    # Perform the optimization
    result = minimize(objective, coeffs_guess, constraints={'type': 'eq', 'fun': constraints})

    global theta_fit, alpha_fit, fitted_coeffs, slice_out

    # Update plot only if optimization is successful
    if result.success:
        fitted_coeffs = result.x
        theta_fit = np.linspace(0, 360, 400)
        alpha_fit = poly_func(theta_fit / 360, fitted_coeffs)

        line.set_data(x_data * 360, y_data)  # Plot data points
        fitted_line.set_data(theta_fit, alpha_fit)  # Plot fitted curve
        #print(fitted_coeffs)

        # Calculate and print the required values
        alpha_0 = poly_func(0, fitted_coeffs)
        alpha_360 = poly_func(1, fitted_coeffs)
        deriv_alpha_0 = poly_deriv(0, fitted_coeffs)
        deriv_alpha_360 = poly_deriv(1, fitted_coeffs)
        #print(f"alpha(0) = {alpha_0}, alpha(360) = {alpha_360}")
        #print(f"d_alpha/d_theta(0) = {deriv_alpha_0}, d_alpha/d_theta(360) = {deriv_alpha_360}")

    else:
        print("Optimization failed:", result.message)

    if slice_out:
        # Extract x and y data for the slice points
        slice_x, slice_y, slice_z = zip(*slice_out)
        slice_points2d.set_data(slice_x, slice_z)
        slice_points3d._offsets3d = (slice_x, slice_y, slice_z)
    else:
        slice_points2d.set_data([],[])
        slice_points3d._offsets3d = ([],[],[])
        
    plt.draw()

# Callback function for updating polynomial order
def submit_order(text):
    global poly_order
    try:
        poly_order = int(text)
    except ValueError:
        print("Please enter a valid integer for the polynomial order.")
    update_plot()

# Callback function for exporting data
def export(ev):
    global theta_fit, alpha_fit, fitted_coeffs, frame, table

    # Create a table of theta, alpha, and alpha values
    sliced_table = map_theta_to_si(fitted_coeffs, frame) # TODO: Reverse alpha (@pi rad)

    global slice_out

    slice_out = []
    for theta in frame.index:
        si = sliced_table.loc[theta]['si']
        alpha = frame.loc[theta, 'Alpha'][si]       # TODO: Reverse alpha (@pi rad)
        slice_out.append((theta, si, alpha))

    #print(slice_out)
    update_plot()

    # Save the data to a header file
    array_to_header_file(sliced_table, 'lookupTable.h')

def clearPoints(ev):
    global coords

    coords = []
    update_plot()

# Initialize the plot
fig = plt.figure(figsize=(24, 12))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

plt.subplots_adjust(bottom=0.25)

line, = ax1.plot([], [], 'ro', label='Data')  # Data points
fitted_line, = ax1.plot([], [], 'r-', label='Fitted Curve')  # Fitted polynomial curve
slice_points2d, = ax1.plot([], [], 'bo', label='Slice')

ax1.set_xlim(0, 360)
ax1.set_ylim(-35, 35)
ax1.set_xlabel('Theta (degrees)')
ax1.set_ylabel('Alpha')
ax1.legend()

# Extract theta, si and alpha values for plotting
theta, si, alpha = table[:,:,0], table[:,:,1], table[:,:,3]
surface = ax2.plot_surface(theta, si, alpha, cmap='viridis', alpha=0.8)
slice_points3d = ax2.scatter([],[],[], color='r', s=50)

ax2.set_xlabel('Theta')
ax2.set_ylabel('Si')
ax2.set_zlabel('Alpha')
#ax2.legend()

# Create a TextBox widget for polynomial order input
axbox = fig.add_axes([0.4, 0.05, 0.1, 0.05])
text_box = TextBox(axbox, 'Poly Order:', initial=str(poly_order))
text_box.on_submit(submit_order)

# Create header export button
ax_export = fig.add_axes([0.75, 0.05, 0.1, 0.05])
btn_export = Button(ax_export, 'Export')
btn_export.on_clicked(export)

# Create header export button
ax_clear = fig.add_axes([0.15, 0.05, 0.1, 0.05])
btn_clear = Button(ax_clear, 'Clear')
btn_clear.on_clicked(clearPoints)

# Connect the onclick event
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
