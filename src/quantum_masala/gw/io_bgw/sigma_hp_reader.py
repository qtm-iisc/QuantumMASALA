"""`sigma_hp.log` reader
Read and parse `sigma_hp.log`, for the purpose of testing.
"""


def read_sigma_hp(filename):
    """Read `sigma_hp.log` file

    :param filename: `sigma_hp.log` filename
    :type filename: str
    """
    data_dict = {}
    with open(filename, "r") as f:
        for line in f:
            # For each k-point
            # If header row is encountered, 
            # start reading data
            # Stop when empty line is encountered.
            if line.strip().startswith("k ="):
                split_line = line.strip().split()
                # Save k-point info
                ik = int(split_line[7])
                # init dict for ik
                data_dict[ik] = {}
                data_dict[ik]["kvec"] = [float(split_line[2]), float(split_line[3]), float(split_line[4])]
                # next line empty
                line = next(f)
                # skip empty line
                line = next(f)
                
                if not "header" in data_dict:
                    data_dict["header"] = line.strip().split()
                
                # init ik lists, one per each header
                for header in data_dict["header"]:
                    data_dict[ik][header] = []

                # start reading data
                line = next(f)
                
                while line.strip():
                    split_line = line.strip().split()
                    data_dict[ik]["n"].append([int(split_line[0])])
                    for i in range(1,len(split_line)):
                        data_dict[ik][data_dict["header"][i]].append(float(split_line[i]))
                    line = next(f)
                    
            elif line.startswith("====="):
                break
            elif line.strip():
                print(line.strip())
                split_line = line.strip().split()
                # if "=" not in split_line:
                data_dict[split_line[0]] = [split_line[i] for i in range(1,len(split_line)) if split_line[i] != "="]
    
    return data_dict

if __name__=="__main__":
    data_dict =read_sigma_hp("sigma_hp.log")
    import pprint
    pp = pprint.PrettyPrinter(compact=True, width=120)
    pp.pprint(data_dict)