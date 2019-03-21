


valid_id = dict()

for category in ("beauty", "fashion", "mobile"):
    with open("final/%s_data_info_val_competition.csv" % (category), "r") as infile:
        next(infile)
        for line in infile:
            curr_id = line.strip().split(',')[0]
            valid_id[curr_id] = True

# This is the new output submission file containing 977987 rows
with open("submission-leak-leakremoved.csv", "w") as outfile:
    outfile.write("id,tagging\n")
    
    # Please change the file below to your current submission filename containing 1174802 rows
    # with open("submission-in.csv", "r")  as infile:
    with open("submission8_leak.csv", "r") as infile:
        next(infile)
        for line in infile:
            curr_id = line.strip().split('_')[0]
            if curr_id in valid_id:
                outfile.write(line.strip() + '\n')
