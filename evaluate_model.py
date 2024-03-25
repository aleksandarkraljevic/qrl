from helper import *

def plot_experiment(parameter_name, repetitions, show, save):
    '''
    Plots an experiment. Both raw plots and smoothed plots are performed.

    Parameters
    ----------
    parameter_name (str):
        The name of the experiment files, excluding "-repeition_" onwards.
    repetitions (int):
        The number of repetitions that the experiment contains.
    show (boolean):
        Whether the plots will be shown to the user.
    save (boolean):
        Whether the plots will be saved using the same name as parameter_name.
    '''
    data_names = []

    for rep in range(repetitions):
        data_names.append(parameter_name + '-repetition_' + str(rep + 1))

    if save:
        plot_averaged(data_names=data_names, show=show, savename=parameter_name, smooth=False)
        plot_averaged(data_names=data_names, show=show, savename=parameter_name+'-smooth', smooth=True)
    else:
        plot_averaged(data_names=data_names, show=show, savename=False, smooth=False)
        plot_averaged(data_names=data_names, show=show, savename=False, smooth=True)

def plot_comparison(parameter_names, repetitions, show, savename, label_names):
    '''
    Plots a comparison between multiple experiments. Both raw plots and smoothed plots are performed.

    Parameters
    ----------
    parameter_names (str):
        The name of the experiments' files, excluding "-repeition_" onwards.
    repetitions (int):
        The number of repetitions that the experiment contains.
    show (boolean):
        Whether the plots will be shown to the user.
    savename (str):
        The name that the comparison plots will be saved as.
    label_names (str):
        The names that the labels in the legend will be, of which each one represents one of the experiments.
    '''
    compare_models(parameter_names=parameter_names, repetitions=repetitions, show=show, savename=savename, label_names=label_names, smooth=False)
    compare_models(parameter_names=parameter_names, repetitions=repetitions, show=show, savename=savename+'-smooth ', label_names=label_names, smooth=True)

def main():
    '''
    This function evalutes what the user is interested in evaluating. Each of the following lines can be commented or uncommented depending on what the user exactly wants to evaluate
    '''
    parameter_names = ['lr_in_0.1-lr_var_0.01-lr_out_0.1', 'lr_in_0.1-lr_var_0.01-lr_out_0.01', 'lr_in_0.1-lr_var_0.01-lr_out_0.001']
    label_names = ['w=0.1', 'w=0.01', 'w=0.001']
    parameter_name = 'lr_in_0.1-lr_var_0.01-lr_out_0.01'

    #plot(data_name='test-repetition_9', show=True, savename='test', smooth=False)

    plot_comparison(parameter_names=parameter_names, repetitions=20, show=True, savename='lr_in_0.1-lr_var_0.01', label_names=label_names)

    #plot_experiment(parameter_name, 20, True, True)

    #compare_training_steps(parameter_names=parameter_names, repetitions=20, convergence_points=[500, 1500])


if __name__ == '__main__':
    main()