import numpy as np
import math
from scipy.optimize import fsolve
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import pandas as pd
import warnings

# Constants
dc = 180/np.pi
N_SMP = 180

R1 = 11.5 #-
R2 = 24 #-
L = 37.4 #-
Sx = -20.5 #-
Sy = 29.6 #-
Sz = 18.9 #-
HCD = 43.558 #20 #-
AoA = 17.675 #10 #-
SA = 33 #-
SL = 36 #-
ED = 36 #-
EC = 36 #-
CD = 52 #-
Spz = 40.92
Cx = 31.7 #-
Cy = 2.4 #-
Cz = 0
CR = 32 #-
DPx = 15.675 #-

AoA_o = 20.13 # phase offset
#AoA_o = 0 # phase offset

# Equation for Phi and Theta
def PhiEq(phi, theta):
    val = (R2 * np.cos(phi) + Sx)**2 + (R2 * np.sin(phi) + Sy - R1 * np.sin(theta))**2 + (Sz - R1 * np.cos(theta))**2 - L**2
    return val

# Functions for Px, Py, Qx, and Qy
def P(theta):
    return (R1 * np.cos(theta), R1 * np.sin(theta))

def Q(si):
    return (Cx + SA * np.cos(si), Cy + SA * np.sin(si))

# Function to solve for Beta
def BetaEq(beta, theta, si):
    val = (Q(si)[0] + SL * np.cos(beta) - P(theta)[0])**2 + (Q(si)[1] + SL * np.sin(beta) - P(theta)[1])**2 - ED**2
    return val

def Beta(theta, si):
    # Solve for all possible beta values (should only be 2)
    beta_range = np.linspace(-np.pi, np.pi, 8)  # Adjust range and density as needed
    betas = set()
    for start in beta_range:
        sol, _, ier, _ = fsolve(BetaEq, start, args=(theta, si), full_output=True)
        if ier == 1:  # ier=1 indicates a successful solution
            sol = sol[0]
            # Check if the solution falls within the desired range before adding
            if -np.pi <= sol <= np.pi:
                betas.add(np.round(sol, decimals=5))
    
    betas = np.array(list(betas))

    if len(betas)<2:
        return None

    return betas

def CP(theta, si):

    betas = Beta(theta, si)
    if not isinstance(betas, np.ndarray):
        return None, None

    Q_loc = Q(si)

    # filter betas to narrow to 1 solution (if no usable solution, return none)
    try:
        
        # filter for angle between SA and SL to stay within proper bounds
        betas = betas[(2*np.pi > (betas - si) % (2*np.pi)) & ((betas - si) % (2*np.pi) > np.pi)]

        # filter for ED angle to stay within proper bounds
        CP_loc = (Q_loc[0] + SL * np.cos(betas), Q_loc[1] + SL * np.sin(betas))
        p = P(theta)

        if len(betas) == 1:
            edA = np.array([math.atan2(p[1] - CP_loc[1][0], p[0] - CP_loc[0][0])])
        else:
            edA = np.array([math.atan2(p[1] - CP_loc[1][0], p[0] - CP_loc[0][0]), math.atan2(p[1] - CP_loc[1][1], p[0] - CP_loc[0][1])])
            pass
        A_bound = math.atan2(Cy, Cx)

        #edA - A_bound

        #if si < -155/dc & si > -165/dc & theta > 235/dc & theta < 245/dc:
        #    pass

        if len(betas) > 1:
            beta = betas[1] # Need to actually solve for correct beta (maybe?)
        else:
            beta = betas[0]

        #if len(betas) > 1:
            #print(theta*dc, si*dc, betas*dc)
    except:
        return None, None
    CP_loc = (Q_loc[0] + SL * np.cos(beta), Q_loc[1] + SL * np.sin(beta))

    return beta % (2*np.pi), CP_loc

def getDifPlate_CP_Angle():
    return math.acos((CD**2 + ED**2 - EC**2) / (2 * CD * ED))

# Equation for Alpha
def equation_for_alpha(alpha, phi, dp):
    #eq1 = (Sx + DPx + HCD * np.cos(phi) + AoA * (-np.sin(phi) * np.sin(alpha)))**2 
    #eq2 = (Sy + HCD * np.sin(phi) + AoA * (np.cos(phi) * np.cos(alpha)) - dp[1])**2
    #eq3 = (Sz + Spz + AoA * np.cos(alpha) - dp[0])**2

    eq1 = (Sx + HCD * np.cos(phi) + AoA * (-np.sin(phi) * np.sin(alpha - AoA_o/dc)) - DPx)**2 
    eq2 = (Sy + HCD * np.sin(phi) + AoA * (np.cos(phi) * np.sin(alpha - AoA_o/dc)) - dp[1])**2
    eq3 = (Sz + Spz + AoA * np.cos(alpha - AoA_o/dc) - dp[0])**2

    return eq1 + eq2 + eq3 - CR**2

# Parallel computation for alpha values
def compute_alpha(theta, si):

    # Suppress all warnings (use with caution!)
    warnings.filterwarnings("ignore")

    # solve for ecentric driver point
    p = P(theta)

    # solve for control point
    beta, cp = CP(theta, si)
    if cp == None:
        return (theta, si, np.nan, np.nan)

    # get angle of ED
    ed_a = math.atan2(p[1] - cp[1], p[0] - cp[0])

    # subtract angle CP diffplate angle
    cd_a = ed_a - getDifPlate_CP_Angle()

    # solve for DP pos
    dp = cp + CD * np.array([np.cos(cd_a), np.sin(cd_a)])

    phi = fsolve(PhiEq, 0, args=(theta))
    alpha_sol = fsolve(equation_for_alpha, np.pi, args=(phi, dp)) #--------------------
    alpha = np.pi - alpha_sol[0]

    if alpha > 200: # TODO: Figure out what is causing spike
        alpha = 0
    
    return theta, si, beta, alpha

# Function to generate the table of values using parallel processing
def generateTable():
    theta_values = np.linspace(0, 2 * np.pi, N_SMP+1)
    si_bounds = np.arctan2(Cy, Cx)
    si_values = np.linspace(si_bounds - np.pi, si_bounds, N_SMP+1)

    # Prepare the tasks
    tasks = [(theta, si) for theta in theta_values for si in si_values]

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results = np.array(list(executor.map(lambda p: compute_alpha(*p), tasks)))

    alpha_table = np.degrees(results)#np.column_stack((results[:,0], results[:,1], results[:,3])))
    frame = pd.pivot(pd.DataFrame(alpha_table, columns=['Theta', 'Si', 'Beta', 'Alpha']), index='Theta', columns='Si', values=['Beta','Alpha'])
    return frame, np.degrees(results).reshape(N_SMP+1, N_SMP+1, 4)


if __name__ == '__main__':
    # Call the function and print results
    frame, alpha_table = generateTable()

    # Extract theta, si and alpha values for plotting
    theta, si, alpha = alpha_table[:,:,0], alpha_table[:,:,1], alpha_table[:,:,3]

    #gradient = np.gradient(alpha)
    #threshold = 15
    #mask = np.max(np.abs(gradient), axis=0) < threshold
    #masked_alpha = np.ma.masked_where(~mask,alpha)


    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    ax.plot_surface(theta, si, alpha, cmap='viridis')

    # Set labels
    ax.set_xlabel('Theta')
    ax.set_ylabel('Si')
    ax.set_zlabel('Alpha')

    # Show plot
    plt.show()

