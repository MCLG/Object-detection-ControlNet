import matplotlib.pyplot as plt
import csv
import sys

def sample(loc=None,temp_file=None) :
    
    if loc is None :
        loc = "epsplot.png"
    if temp_file is None :
        #temp_file = 'guided_sampling_bounds.csv'
        msg = 'No file "temp_file" passed as input. sample() requires a csv. file created during a cldm.cldm.ControlNet.count_guided_sampling(progress_track : str ) call. \n'
         'Usage : 1) cldm.cldm.ControlNet.count_guided_sampling(progress_track : str = temp_file) \n'
         '        2) "python CGsampling_plotter.py temp_file" or "sample(temp_file=temp_file)" '
        raise TypeError(msg)
    content = dict()

    with open(temp_file, 'r') as data:
        reader = csv.DictReader(data)
        
        # Initialize the content dictionary with empty lists for each header
        for key in reader.fieldnames:
            content[key] = []

        # Read each row and append values to the corresponding lists in content
        for row in reader:
            for key in content.keys():
                content[key].append(float(row[key]))

    from matplotlib import scale as mscale
    from matplotlib import transforms as mtransforms
    from matplotlib.ticker import FixedLocator, FuncFormatter

    fig, ax = plt.subplots(figsize=(20,12))
    time = [t for t in reversed(range(len(content[list(content.keys())[0]])))]
    ax2 = ax.twinx()

    ax2.plot(time,content['eps_min'],c='b',linestyle='dashed', label = r'range of predicted noise $\varepsilon_t$')
    ax2.plot(time,content['eps_max'],c='b',linestyle='dashed')

    ax2.plot(time,content['teps_min'],c='r', linestyle='-.', label = r'range of predicted guided noise $\tilde{\varepsilon}_t$')
    ax2.plot(time,content['teps_max'],c='r',linestyle='-.')

    ax.plot(time,content['x_min'],c='g',linestyle ='-', label = r'range of x_t')
    ax.plot(time,content['x_max'],c='g',linestyle ='-')

    ax2.plot(time,content['score_min'], c = 'k', linestyle = '-', label = r'range of score $\nabla_{x_t}p(y_{gt}|x_t)$')
    ax2.plot(time,content['score_max'], c = 'k', linestyle = '-')
    
    ax2.plot(time,content['alpha_'], c = 'm', linestyle = ':', label = r'guidance scale $\alpha$ (log scaled)')
    ax2.set_yscale('log')

    ax.set(xlabel='timestep t',
        title='Value range (min and max) of tensors implicated in the guided sampling procedure')
    ax.grid()
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    ax.set_yscale('linear')
    fig.savefig(loc)
    plt.show()

def main() :
    if len(sys.argv) < 2:
        print("Error: Please provide 'temp_file' .")
        sys.exit(1)  # Exit the program with an error code

    temp_file = sys.argv[1]

    sample(loc= None, temp_file= temp_file)

if __name__ == '__main__' :
    main()
