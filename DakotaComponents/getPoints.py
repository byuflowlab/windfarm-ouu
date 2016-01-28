
import subprocess
import sys
#import SU2

# -------------------------------------------------------------------
#  Output Redirection
# -------------------------------------------------------------------
# original source: http://stackoverflow.com/questions/6796492/python-temporarily-redirect-stdout-stderr
class RedirectOutput(object):
    """ with SU2.io.redirect_output(stdout,stderr)

        Temporarily redirects sys.stdout and sys.stderr when used in
        a 'with' contextmanager

        Example:
        with SU2.io.redirect_output('stdout.txt','stderr.txt'):
            sys.stdout.write("standard out")
            sys.stderr.write("stanrard error")
            # code
        #: with output redirection

        Inputs:
            stdout - None, a filename, or a file stream
            stderr - None, a filename, or a file stream
        None will not redirect outptu

    """
    def __init__(self, stdout=None, stderr=None):
        print stdout
        _newout = False
        _newerr = False

        if isinstance(stdout, str):
            stdout = open(stdout, 'w')
            _newout = True
        if isinstance(stderr, str):
            stderr = open(stderr, 'a')
            _newerr = True

        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr
        self._newout = _newout
        self._newerr = _newerr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        if self._newout:
            self._stdout.close()
        if self._newerr:
            self._stderr.close()

#: class output()

# Pipe the output
# Call dakota
print 'Calling dakota...'
log = 'logDakota.out'
err = 'logDakota.err'
with RedirectOutput(log,err):
#with SU2.io.redirect_output(log):
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

