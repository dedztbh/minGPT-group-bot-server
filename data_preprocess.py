import re

with open('qbltjl.txt', 'r') as f:
    lines = f.readlines()


# remove date, @, [xxx] stuff
def filt(line):
    return re.match(r'\d{4}-\d{2}-\d{2} ', line) is None \
           and re.search(r'\[.+]', line) is None \
           and re.search(r'@', line) is None \
           and re.search(r'\d{4}/\d{2}/\d{2} ', line) is None


# write to new file merging newlines
with open('input.txt', 'w') as f:
    prev_newline = False
    for line in filter(filt, lines):
        if line == '\n':
            if prev_newline:
                continue
            prev_newline = True
        else:
            prev_newline = False
        f.write(line)
