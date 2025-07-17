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
n_samples = int(config["settings"]["n_samples"])
output_location = config["settings"]["output_location"]


def parse_dataset(dataset):
    data_arr = []
    with open(dataset, "r") as f:
        for line in f:
            val = line.splitlines()
            data_arr.append(val)

    return data_arr

# reads the filename and returns a dictionary containing the time elements
def get_file_time(filename):

    # remove the RSA tag
    parts = filename.split("-")
    parts = parts[1]

    # split by decimal point
    parts = parts.split(".")

    # add each time unit as an integer to a list, except for the .tiq file extension
    time_units = []
    for unit in parts:
        try:
            time_units.append(int(unit))
        except ValueError:
            pass


    # initialise the dictionary and keys
    time_dict = {}

    dict_keys = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond"
    ]

    # add time units with keys to the dictionary
    for unit in range(len(time_units)):
        time_dict.update({dict_keys[unit] : time_units[unit]})

    return time_dict


def f_tracker(dataset, path, output_location, n_samples):
    
    max_frequencies = []
    time_elements = []

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

        time_elements.append(get_file_time(filename[0]))


    output_name = f"{experiment_name}_{t_start}-{t_end}_FMAX-{str(n_samples)}samples"

    print(f"Saving array containing maximum frequencies between {t_start} and {t_end} to file {output_name}.npz in location {output_location}")
    
    np.savez(output_location + output_name + ".npz", max_frequencies=max_frequencies, time_elements=time_elements)


def main():

    dataset = parse_dataset(file_list)

    f_tracker(dataset=dataset, path=file_path, output_location=output_location, n_samples=n_samples)


if __name__ == "__main__":
    main()


        