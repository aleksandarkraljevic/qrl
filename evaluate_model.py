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

def main():
    '''
    This function evalutes what the user is interested in evaluating. Each of the following lines can be commented or uncommented depending on what the user exactly wants to evaluate
    '''
    parameter_names = ['full_comb', 'reduced_layer']
    label_names = ['XYZ', 'YZ']
    parameter_name = 'non_flipped'

    #plot(data_name='test', show=True, savename='test', smooth=False)

    #compare_models(parameter_names=parameter_names, repetitions=20, show=True, savename='check_reduced_layer_smooth', label_names=label_names, smooth=True)

    plot_experiment(parameter_name, 20, True, True)


if __name__ == '__main__':
    main()