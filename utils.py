import os
import numpy as np

def array_to_header_file(frame, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    
    with open(filename, 'w') as file:
        file.write("#ifndef LOOKUP_TABLE_H\n")
        file.write("#define LOOKUP_TABLE_H\n\n")
        file.write("const float lookupTable[][2] = {\n")
        for theta in frame.index:
            file.write(f"  {{{theta}, {frame.loc[theta]['si']}}},\n")
        file.write("};\n\n")
        file.write("#endif // LOOKUP_TABLE_H")

from numpy.polynomial.polynomial import Polynomial
import pandas as pd

def map_theta_to_si(alpha_coeffs, df):
    
    poly = Polynomial(alpha_coeffs)

    theta_values = df.index
    si_values = df.iloc[0].index.get_level_values('Si').unique()

    expected_alpha = poly(theta_values/360)

    theta_to_si_df = pd.DataFrame(index=theta_values, columns=['si'])

    # start end, si error
    si_se_error = pd.DataFrame(index=si_values, columns=['si_error', 'alpha_error', 'si_0', 'tot_mov'])

    for ii, s in enumerate(si_values):

        total_alpha_error = 0
        tot_mov = 0
        # Find the si value for each theta that has the alpha value closest to the expected alpha
        for i, t in enumerate(theta_values):
            # The column (si) in the original DataFrame that has the closest alpha value to the expected alpha
            alpha_error = df.loc[t]['Alpha'].sub(expected_alpha[i]).abs()
            
            if i>0:
                prev_si = theta_to_si_df.iloc[i-1]['si']
            else:
                prev_si = s

            si_error = (alpha_error*0+alpha_error.index-prev_si).abs()
            #si_factor = si_error.apply(lambda x: x**1.01 if x < 15 else 1000)
            si_factor = si_error.apply(weighted_distance_function)

            try:
                best_si = (alpha_error*si_factor).idxmin()
            except:
                pass

            theta_to_si_df.loc[t] = best_si

            total_alpha_error += alpha_error.loc[best_si]
            tot_mov += si_error.loc[best_si]

        si_se_error.loc[s, 'si_error'] = np.abs(theta_to_si_df.loc[theta_values[-1]] - theta_to_si_df.loc[theta_values[0]])['si']
        si_se_error.loc[s, 'alpha_error'] = total_alpha_error
        si_se_error.loc[s, 'si_0'] = theta_to_si_df.loc[0, 'si']
        si_se_error.loc[s, 'tot_mov'] = tot_mov

    min_si_error = si_se_error['si_error'].min()
    filtered_si_se_error = si_se_error[si_se_error['si_error'] == min_si_error]

    min_alpha_error = filtered_si_se_error['alpha_error'].min()
    filtered_alpha_error = filtered_si_se_error[filtered_si_se_error['alpha_error'] == min_alpha_error]

    tot_mov_df = filtered_alpha_error['tot_mov']

    best_i = 0
    min = 1000
    for i in range(len(tot_mov_df.index)):
        val = tot_mov_df.iloc[i]
        if val < min:
            min = val
            best_i = i

    best_si_start = tot_mov_df.index[best_i]

    for i, t in enumerate(theta_values):
        # The column (si) in the original DataFrame that has the closest alpha value to the expected alpha
        alpha_error = df.loc[t, 'Alpha'].sub(expected_alpha[i]).abs()
        
        if i>0:
            prev_si = theta_to_si_df.iloc[i-1]['si']
        else:
            prev_si = best_si_start

        try:
            si_error = (alpha_error*0+alpha_error.index-prev_si).abs()
        except: 
            pass
        #si_factor = si_error.apply(lambda x: x**1.01 if x < 15 else 1000)
        si_factor = si_error.apply(weighted_distance_function)

        try:
            best_si = (alpha_error*si_factor).idxmin()
        except:
            pass

        theta_to_si_df.loc[t] = best_si

    #print(theta_to_si_df)
    return theta_to_si_df


def weighted_distance_function(x, threshold1=4, threshold2=8, decay_rate1=0.01, decay_rate2=0.15):
    """
    Weights the distances by applying a different exponential decay based on the thresholds.
    
    :param x: The distance value to apply the weighting function.
    :param threshold1: The first threshold where the decay starts to happen.
    :param threshold2: The second threshold after which the decay happens more rapidly.
    :param decay_rate1: The decay rate before the second threshold.
    :param decay_rate2: The decay rate after the second threshold.
    :return: The weighted distance.
    """
    if x < threshold1:
        return 1
    elif threshold1 <= x < threshold2:
        return np.exp(decay_rate1 * (x))
    else:
        return np.exp(decay_rate2 * (x)) * np.exp(decay_rate1)
