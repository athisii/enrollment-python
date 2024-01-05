import re

pattern = re.compile(r'^\d')

with open("pes_fhd.css", 'w') as out_file:
    with open("pes_base.css") as in_file:
        for line in in_file.readlines():
            tokens = line.split(':')
            if len(tokens) > 1:
                stripped = tokens[1].strip()
                if pattern.match(stripped):
                    num_str = re.findall(r'\d+', stripped)[0]  # returns list; get first element
                    num = int(int(num_str) * 1.4)
                    out_file.write(tokens[0] + ": " + str(num) + "px;")
                else:
                    out_file.write(line)
            else:
                out_file.write(line)

print("Done.")
