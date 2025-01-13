import csv
import matplotlib.pyplot as plt
from bisect import insort
from statistics import median, mean

def plot_grad(file_path,name) :
        
    data = []
    min_mean, max_mean, min_median, max_median = [], [], [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        prev_itr = 0
        l_min = list()
        l_max = list()
        for row in reader:
            
            itr, minimum, maximum = row[0], row[-2], row[-1]#[float(value) for value in row][0,-2,-1]
            
            if int(itr) != prev_itr :
                
                min_median.append(median(l_min))
                max_median.append(median(l_max))
                min_mean.append(mean(l_min))
                max_mean.append(mean(l_max))

                l_min = list()
                l_max = list()
                prev_itr = int(itr) 

            insort(l_min,float(minimum))
            insort(l_max,float(maximum))
    f, ax = plt.subplots(1,2,figsize=(12,6))
    indices = [i for i in range(0,len(min_mean))]
    if len(indices) > 262 :
        indices = indices[:262]
        min_mean = min_mean[:262]
        max_mean = max_mean[:262]
        min_median = min_median[:262]
        max_median = max_median[:262]

    ax[0].plot(indices, min_mean, label='Min Mean', linestyle='-', color='blue')  # Line style and color
    ax[0].plot(indices, max_mean, label='Max Mean', linestyle='--', color='red')  # Line style and color
    ax[0].set_yscale('log')
    ax[0].set_title('Min/Max Mean')  # Title for the first subplot
    ax[0].legend()  # Display the legend for the first subplot

    ax[1].plot(indices, min_median, label='Min Median', linestyle='-.', color='green')  # Line style and color
    ax[1].plot(indices, max_median, label='Max Median', linestyle=':', color='orange')  # Line style and color
    ax[1].set_yscale('log')
    ax[1].set_title('Min/Max Median')  # Title for the second subplot
    ax[1].legend()  # Display the legend for the second subplot

    plt.savefig(f'./{name}.png')
    print(f'Saved file at "./{name}.png" ')

def main() :
    
    file_path = './gradients_clip.txt'
    plot_grad(file_path,'mean-median-grad-clip')
    file_path = './gradients_no_clip.txt'
    plot_grad(file_path,'mean-median-grad-no-clip')

if __name__ == '__main__' :
    main()



