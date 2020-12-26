import torch
import model
from tqdm import tqdm

LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}

def plot_samples(data_set, dataPlot):

    for i in range(len(data_set)):
        data = data_set[i]
        dataPlot.show_data(data)
        print(i, data['image'].size(), LABELS[data['label']])
        if i == 3:
            break

    for i_batch, sample_batch in enumerate(data_set):
        print(i_batch, sample_batch['image'].size(),
              sample_batch['label'].size())

        if i_batch == 3:
            dataPlot.show_batch(sample_batch, 3)
            break

def save_net(network, PATH):
    torch.save(network.state_dict(), PATH)

def load_net(PATH):
    net = model.Classifier()
    net.load_state_dict(torch.load(PATH))
    net.eval()
    return net

def get_mean_and_std(data_set):
    colour_sum = 0
    channel_squared = 0

    print("\nCalculating mean and std:\n")


    for i_batch, sample_batch in enumerate(tqdm(data_set)):
        colour_sum += torch.mean(sample_batch['image'], dim=[0,2,3])
        channel_squared += torch.mean(sample_batch['image']**2, dim=[0,2,3])
    mean = colour_sum / len(data_set)
    std  = (channel_squared/len(data_set) - mean**2)**0.5

    print(f"\nMean: {mean}")
    print(f"Standard Deviation: {std}")

    return mean, std
