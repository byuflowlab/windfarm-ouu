
import subprocess

# Will need the x y locations.
# Use these to overwrite the the dakota input file
# I have done this in the cluster
dakotaInput = 'dakotaInput.in'
#dakotaInput = '--version'
subprocess.check_call(['dakota', dakotaInput], stdout=sys.stdout,
                             stderr=sys.stderr)

print 'finished calling dakota.'
# read the dakota tabular file

dakotaTabular = 'dakota_tabular.dat'
f = open(dakotaTabular,'r')
line = f.readline()
x = []
for line in f:
    x.append(float(line.split()[2]))




# I need to check the return of call and raise error if something happened.


# return or promote wind directions

