############################ CREDIT:###################################################################
    #Metadata-Version: 2.1
    #Name: mlrose-hiive
    #Version: 2.2.4
    #Summary: MLROSe: Machine Learning, Randomized Optimization and Search (hiive extended remix)
    #Home-page: https://github.com/hiive/mlrose
    #Author: Genevieve Hayes (modified by Andrew Rollings)
    #License: BSD
    #Download-URL: https://github.com/hiive/mlrose/archive/2.2.4.tar.gz
    #Description: # mlrose: Machine Learning, Randomized Optimization and SEarch
########################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
import random
from mlrose_hiive.runners import GARunner, MIMICRunner, RHCRunner, SARunner
from mlrose_hiive.generators import QueensGenerator
import os

import time

seed = 10
random.seed(seed)
OUTPUT_DIRECTORY = './output'
problem_size = 8


#########################################################################
#      BELOW IS THE CREATION OF MAIN, WHICH IMPLEMENTS MY 4 RUNNERS    #
#########################################################################

def main():
    # Creating the N-Queens problem here BUT also ensure change the problem size above (per FAQ requirement)
    problem = QueensGenerator.generate(size=problem_size, seed=seed)

    # GA Algo Runner Creation for N-Queens
    experiment_name_ga = 'N-Queens_GA'
    ga = GARunner(problem=problem, experiment_name=experiment_name_ga, output_directory=OUTPUT_DIRECTORY, seed=seed,
                  iteration_list=2 ** np.arange(10), max_attempts=100, population_sizes=[150, 200, 250, 300],
                  mutation_rates=[0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4])
    ga.run()

    # SA Algo Runner Creation for N-Queens
    experiment_name_sa = 'N-Queens_SA'
    sa = SARunner(problem=problem, experiment_name=experiment_name_sa, output_directory=OUTPUT_DIRECTORY, seed=seed,
                  iteration_list=2 ** np.arange(10), max_attempts=50,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000])
    sa.run()

    # MIMIC Algo Runner Creation for N-Queens
    experiment_name_mimic = 'N-Queens_MIMIC'
    mmc = MIMICRunner(problem=problem, experiment_name=experiment_name_mimic, output_directory=OUTPUT_DIRECTORY,
                      seed=seed,
                      iteration_list=2 ** np.arange(10),
                      max_attempts=50, population_sizes=[200, 250], keep_percent_list=[0.10, 0.20])
    mmc.run()

    # Random Hill Algo Runner Creation for N-Queens
    experiment_name_rhc = 'N-Queens_RHC'
    rhc = RHCRunner(problem=problem, experiment_name=experiment_name_rhc, output_directory=OUTPUT_DIRECTORY, seed=seed,
                    iteration_list=2 ** np.arange(10), max_attempts=100, restart_list=[25, 75])
    rhc.run()

#########################################################################
#                             END OF MAIN                               #
#########################################################################


#########################################################################
#           BELOW IS THE CREATION OF THE MAIN GRAPHS REQUIRED           #
#########################################################################

def graph_nqueens_sa():
    # READING SA SAVED CSV DATA
    df = pd.read_csv("./output/N-Queens_SA/sa__N-Queens_SA__run_stats_df.csv")

    # Establising temp values to be used and renaming fitness column below to correspond to the various temps
    temperatures = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000]
    data_frames = []

    for temp in temperatures:
        df_temp = df.loc[(df['Temperature'] == temp)].copy()
        df_temp = df_temp[['Iteration', 'Fitness']]
        max_fitness = df_temp['Fitness'].max()
        df_temp['Fitness'] = max_fitness - df_temp['Fitness']
        df_temp.rename(columns={'Fitness': f'Temperature {temp}'}, inplace=True)
        data_frames.append(df_temp)

    # merging iteration column
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'], how='outer'), data_frames)

    # plotting - note to self: check/scan outfile files and adjust y and x axis as needed
    ax = df_merged.plot(x='Iteration', colormap='gist_rainbow')
    plt.ylim(0, 25)
    plt.xlim(0, 1400)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    title = f"N-Queens for SA (Problem Size {problem_size})"
    plt.title(title, fontsize=14, y=1.03)
    plt.savefig('./graphs/' + 'N-Queens_SA' + '_' + '.png')
    plt.close()

def graph_nqueens_ga():
    df = pd.read_csv("./output/N-Queens_GA/ga__N-Queens_GA__run_stats_df.csv")

    # setting parameters for GA and creating dataframes
    pop_sizes = [150, 200, 250, 300]
    mutation_rates = [0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4]
    data_frames = []

    for pop_size in pop_sizes:
        for mutation_rate in mutation_rates:
            df_temp = df.loc[(df['Population Size'] == pop_size) & (df['Mutation Rate'] == mutation_rate)].copy()
            df_temp = df_temp[['Iteration', 'Fitness']]
            max_fitness = df_temp['Fitness'].max()
            df_temp['Fitness'] = max_fitness - df_temp['Fitness']
            df_temp.rename(columns={'Fitness': f'Pop {pop_size}, Mutation {mutation_rate}'}, inplace=True)
            data_frames.append(df_temp)

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'], how='outer'), data_frames)
    ax = df_merged.plot(x='Iteration', colormap='gist_rainbow')
    plt.ylim(0, 16)
    plt.xlim(0, 100)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    title = f"N-Queens for GA (Problem Size {problem_size})"
    plt.title(title, fontsize=14, y=1.03)
    plt.savefig('./graphs/' + 'N-Queens_GA' + '_' + '.png')
    plt.close()

def graph_mimic():
    df = pd.read_csv("./output/N-Queens_MIMIC/mimic__N-Queens_MIMIC__run_stats_df.csv")

    # Establishing graphing parameters - note to self: ensure these match the parameters set above in main for each algo
    pop_sizes = [200, 250]
    keep_percents = [0.10, 0.20]
    data_frames = []

    for pop_size in pop_sizes:
        for keep_percent in keep_percents:
            df_temp = df.loc[(df['Population Size'] == pop_size) & (df['Keep Percent'] == keep_percent)].copy()
            df_temp = df_temp[['Iteration', 'Fitness']]
            max_fitness = df_temp['Fitness'].max()
            df_temp['Fitness'] = max_fitness - df_temp['Fitness']
            df_temp.rename(columns={'Fitness': f'Pop {pop_size}, Keep Percent {keep_percent}'}, inplace=True)
            data_frames.append(df_temp)

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'], how='outer'), data_frames)
    ax = df_merged.plot(x='Iteration', colormap='gist_rainbow')
    plt.ylim(0, 16)
    plt.xlim(0, 15)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    title = f"N-Queens for MIMIC (Problem Size {problem_size})"
    plt.title(title, fontsize=14, y=1.03)
    plt.savefig('./graphs/' + 'N-Queens_MIMIC' + '_' + '.png')
    plt.close()

def graph_rhc():
    df = pd.read_csv("./output/N-Queens_RHC/rhc__N-Queens_RHC__run_stats_df.csv")

    # Establishing graphing parameters - note to self: ensure these match the parameters set above in main for each algo
    restarts = [25, 75]
    data_frames = []

    for restart in restarts:
        df_temp = df[df['current_restart'] == 0].loc[df['Restarts'] == restart].copy()
        df_temp = df_temp[['Iteration', 'Fitness']]
        max_fitness = df_temp['Fitness'].max()
        df_temp['Fitness'] = max_fitness - df_temp['Fitness']
        df_temp.rename(columns={'Fitness': f'Restarts {restart}'}, inplace=True)
        data_frames.append(df_temp)

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'], how='outer'), data_frames)

    # plotting - note to self: check/scan outfile files and adjust y and x axis as needed
    ax = df_merged.plot(x='Iteration', colormap='gist_rainbow')
    plt.ylim(0, 16)
    plt.xlim(0, 100)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    title = f"N-Queens for RHC (Problem Size {problem_size})"
    plt.title(title, fontsize=14, y=1.03)
    plt.savefig('./graphs/' + 'N-Queens_RHC' + '_' + '.png')
    plt.close()



def load_and_filter_data(file_path, **filters):
    df = pd.read_csv(file_path)
    for key, value in filters.items():
        df = df[df[key] == value]
    return df

def graph_timings():
    configs = {
        'GA': {
            'path': "./output/N-Queens_GA/ga__N-Queens_GA__run_stats_df.csv",
            'filters': {'Population Size': 250, 'Mutation Rate': 0.03}
        },
        'MIMIC': {
            'path': "./output/N-Queens_MIMIC/mimic__N-Queens_MIMIC__run_stats_df.csv",
            'filters': {'Population Size': 150, 'Keep Percent': 0.2}
        },
        'SA': {
            'path': "./output/N-Queens_SA/sa__N-Queens_SA__run_stats_df.csv",
            'filters': {'Temperature': 10}
        },
        'RHC': {
            'path': "./output/N-Queens_RHC/rhc__N-Queens_RHC__run_stats_df.csv",
            'filters': {'Restarts': 25, 'current_restart': 14}
        }
    }

    data_frames = []
    for algo, config in configs.items():
        df_temp = load_and_filter_data(config['path'], **config['filters'])
        df_temp = df_temp[['Iteration', 'Time']].rename(columns={'Time': algo})
        data_frames.append(df_temp)

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'], how='outer'), data_frames)

    # plotting - note to self: check/scan outfile files and adjust y and x axis as needed
    ax = df_merged.plot(x='Iteration', colormap='gist_rainbow')
    plt.ylim(0, 2)
    plt.xlim(0, 100)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.05, 1.1), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Timings")
    title = f"N-Queens Timings for All 4 Algorithms (Problem Size {problem_size})"
    plt.title(title, fontsize=14, y=1.03)
    plt.savefig('./graphs/' + 'N-Queens_Timings' + '_' + '.png')
    plt.close()

def graph_fevals_per_iteration():
    configs = {
        'GA': "./output/N-Queens_GA/ga__N-Queens_GA__run_stats_df.csv",
        'MIMIC': "./output/N-Queens_MIMIC/mimic__N-Queens_MIMIC__run_stats_df.csv",
        'SA': "./output/N-Queens_SA/sa__N-Queens_SA__run_stats_df.csv",
        'RHC': "./output/N-Queens_RHC/rhc__N-Queens_RHC__run_stats_df.csv"
    }

    data_frames = []
    for algo, file_path in configs.items():
        df_temp = load_and_filter_data(file_path)
        df_temp['FEvals_per_Iteration'] = df_temp['FEvals'] / (df_temp['Iteration'] + 1)  # +1 to avoid division by zero
        df_temp = df_temp[['Iteration', 'FEvals_per_Iteration']].rename(columns={'FEvals_per_Iteration': algo})
        data_frames.append(df_temp)

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'], how='outer'), data_frames)

    # plotting - note to self: check/scan outfile files and adjust y and x axis as needed
    ax = df_merged.plot(x='Iteration', colormap='gist_rainbow')
    plt.ylim(0, 500)
    plt.xlim(0, 500)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.00, 1.00), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("FEvals per Iteration for nqueens")
    title = f"FEvals per Iteration for All 4 Algorithms (Problem Size {problem_size})"
    plt.title(title, fontsize=14, y=1.03)
    plt.savefig('./graphs/' + 'N-Queens_FEvals_per_Iteration' + '_' + '.png')
    plt.close()


#########################################################################
#                             END OF GRAPHS                             #
#########################################################################

if __name__ == "__main__":
    main()
    graph_nqueens_sa()
    graph_nqueens_ga()
    graph_mimic()
    graph_rhc()
    graph_timings()
    graph_fevals_per_iteration()



