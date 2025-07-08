from iqtools import tools
import numpy as np
import toml


# Load the config
with open("config.toml", "r") as f:
    config = toml.load(f)

# Load naming settings from config
t_start = config["settings"]["t_start"]
t_end = config["settings"]["t_end"]
experiment_name = config["settings"]["experiment_name"]

# Load program settings from config
file_list = config["settings"]["file_list"]
file_path = config["settings"]["file_path"]
n_samples = config["settings"]["n_samples"]
output_location = config["settings"]["output_location"]


def parse_dataset(dataset):
    data_arr = []
    with open(dataset, "r") as f:
        for line in f:
            val = line.splitlines()
            data_arr.append(val)

    return data_arr


def f_tracker(dataset, path, output_location, n_samples):
    
    max_frequencies = np.array([])

    for filename in dataset:
        fullpath = path + filename[0]
        print(f"Processing {filename}")

        iq = tools.get_iq_object(fullpath)
        iq.read_samples(n_samples)

        iq.window = "hamming"
        iq.method = "mtm"

        ff, tt, pp = iq.get_power_spectrogram(nframes=1,
                                      lframes=n_samples)
        
        ff += iq.center

        f_max_index = np.argmax(pp)
        f_max = ff[0][f_max_index]
        max_frequencies.append(f_max)


    output_name = f"{experiment_name}_{t_start}-{t_end}_FMAX"

    print(f"Saving array containing maximum frequencies between {t_start} and {t_end} to file {output_name}.npz in location {output_location}")
    
    np.savez(output_location + output_name + ".npz", max_frequencies=max_frequencies)


def main():

    dataset = parse_dataset(file_list)

    f_tracker(dataset=dataset, path=file_path, output_location=output_location, n_samples=n_samples)


if __name__ == "__main__":
    main()


        