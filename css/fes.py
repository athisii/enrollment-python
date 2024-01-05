import re

pattern = re.compile(r'^\d')

with open("fes_1600x1200.css", 'w') as out_file:
    with open("fes_base.css") as in_file:
        for line in in_file.readlines():
            tokens = line.split(':')
            if len(tokens) > 1:
                stripped = tokens[1].strip()
                if pattern.match(stripped):
                    num_str = re.findall(r'\d+', stripped)[0]
                    # num = int(int(num_str) * 1.3) # for 1400x1050
                    num = int(int(num_str) * 1.5)  # for 1600x1200
                    out_file.write(tokens[0] + ": " + str(num) + "px;")
                else:
                    out_file.write(line)
            else:
                out_file.write(line)

print("Done.")
