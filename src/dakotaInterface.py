
import re
import sys
import shutil


def parseDakotaParametersFile(paramsfilename):
    """Return parameters for application."""

    # setup regular expressions for parameter/label matching
    e = r'-?(?:\d+\.?\d*|\.\d+)[eEdD](?:\+|-)?\d+'  # exponential notation
    f = r'-?\d+\.\d*|-?\.\d+'                       # floating point
    i = r'-?\d+'                                    # integer
    value = e + '|' + f + '|' + i                   # numeric field
    tag = r'\w+(?::\w+)*'                           # text tag field

    # regular expression for standard parameters format
    standard_regex = re.compile('^\s*(' + value + ')\s+(' + tag + ')$')

    # open DAKOTA parameters file for reading
    paramsfile = open(paramsfilename, 'r')

    # extract the parameters from the file and store in a dictionary
    paramsdict = {}
    for line in paramsfile:
        m = standard_regex.match(line)
        if m:
            # print m.group()
            paramsdict[m.group(2)] = m.group(1)

    paramsfile.close()

    return paramsdict


def checknVar(nvar, paramsdict):
    """Check to make sure we have the right number of uncertain variables."""
    num_vars = 0
    if 'variables' in paramsdict:
        num_vars = int(paramsdict['variables'])
    if num_vars != nvar:
        message = 'Error: Simulation expected ' + str(nvar) + ' variables, found ' \
            + str(num_vars) + ' variables.'
        raise Exception(message)


def writeDakotaResultsFile(
        resultfilename, resultsdict, paramsdict, active_set_vector):
    """Write results of application for Dakota."""

    # Make sure number of functions is as expected.
    num_fns = 0
    if 'functions' in paramsdict:
        num_fns = int(paramsdict['functions'])
    if num_fns != len(resultsdict['fns']):
        raise Exception('Number of functions not as expected.')

    # write outputfile
    outfile = open(resultfilename, 'w')

    for func_ind in range(0, num_fns):
        # write functions
        if active_set_vector[func_ind] & 1:
            functions = resultsdict['fns']
            outfile.write(str(functions[func_ind]) +
                          ' f' + str(func_ind) + '\n')

    # write gradients
    for func_ind in range(0, num_fns):
        if active_set_vector[func_ind] & 2:
            grad = resultsdict['fnGrads'][func_ind]
            outfile.write('[ ')
            for deriv in grad:
                outfile.write(str(deriv) + ' ')
            outfile.write(']\n')

    outfile.close()


# -------------------------------------------------------------------
#  Output Redirection
# -------------------------------------------------------------------
# original source: http://stackoverflow.com/questions/6796492/python-temporarily-redirect-stdout-stderr
class RedirectOutput(object):
    """ with RedirectOutput(stdout,stderr)

        Temporarily redirects sys.stdout and sys.stderr when used in
        a 'with' contextmanager

        Example:
        with Redirect_output('stdout.txt','stderr.txt'):
            sys.stdout.write("standard out")
            sys.stderr.write("standard error")
            # code
        #: with output redirection

        Inputs:
            stdout - None, a filename, or a file stream
            stderr - None, a filename, or a file stream
        None will not redirect output

    """
    def __init__(self, stdout=None, stderr=None):

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
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        if self._newout:
            self._stdout.close()
        if self._newerr:
            self._stderr.close()

#: class output()


def updateDakotaFile(dakotaFilename, quadrature_points, x, f):
    """Rewrite number of quadrature points in Dakota file."""

    filein = dakotaFilename
    fileout = dakotaFilename + '.tmp'
    fr = open(filein, 'r')
    fw = open(fileout, 'w')
    for line in fr:
        if 'quadrature_order' in line:
            towrite = 'quadrature_order  ' + str(quadrature_points) + '\n'
            fw.write(towrite)
        elif 'abscissas' in line:
            towrite = 'abscissas = '
            for xi in x:
                towrite = towrite + str(xi) + ' '
            towrite += '\n'
            fw.write(towrite)
        elif 'ordinates' in line:
            towrite = 'ordinates = '
            for fi in f:
                towrite = towrite + str(fi) + ' '
            towrite += '0.0\n'
            fw.write(towrite)
        else:
            fw.write(line)
    fr.close()
    fw.close()
    shutil.move(fileout, filein)


def updateDakotaFile5(dakotaFilename, quadrature_points, bounds):
    """Rewrite number of quadrature points in Dakota file."""

    filein = dakotaFilename
    fileout = dakotaFilename + '.tmp'
    fr = open(filein, 'r')
    fw = open(fileout, 'w')
    for line in fr:
        if 'quadrature_order' in line:
            towrite = 'quadrature_order  ' + str(quadrature_points) + '\n'
            fw.write(towrite)
        elif 'lower_bounds' in line:
            towrite = 'lower_bounds = ' + str(bounds[0]) + '\n'
            fw.write(towrite)
        elif 'upper_bounds' in line:
            towrite = 'upper_bounds = ' + str(bounds[1]) + '\n'
            fw.write(towrite)
        else:
            fw.write(line)
    fr.close()
    fw.close()
    shutil.move(fileout, filein)


def updateDakotaFile_before_introducing_bounds(dakotaFilename, quadrature_points):
    """Rewrite number of quadrature points in Dakota file."""

    filein = dakotaFilename
    fileout = dakotaFilename + '.tmp'
    fr = open(filein, 'r')
    fw = open(fileout, 'w')
    for line in fr:
        if 'quadrature_order' in line:
            towrite = 'quadrature_order  ' + str(quadrature_points) + '\n'
            fw.write(towrite)
        else:
            fw.write(line)
    fr.close()
    fw.close()
    shutil.move(fileout, filein)


def updateDakotaFile4(dakotaFilename, quadrature_points):
    """Rewrite number of quadrature points in Dakota file."""

    filein = dakotaFilename
    fileout = dakotaFilename + '.tmp'
    fr = open(filein, 'r')
    fw = open(fileout, 'w')
    for line in fr:
        if 'sparse_grid_level' in line:
            towrite = 'sparse_grid_level  ' + str(quadrature_points) + '\n'
            fw.write(towrite)
        else:
            fw.write(line)
    fr.close()
    fw.close()
    shutil.move(fileout, filein)


def updateDakotaFile2(dakotaFilename, quadrature_points):
    """Rewrite number of quadrature points in Dakota file."""

    filein = dakotaFilename
    fileout = dakotaFilename + '.tmp'
    fr = open(filein, 'r')
    fw = open(fileout, 'w')
    for line in fr:
        if 'collocation_points' in line:
            towrite = 'collocation_points  ' + str(quadrature_points) + '\n'
            fw.write(towrite)
        else:
            fw.write(line)
    fr.close()
    fw.close()
    shutil.move(fileout, filein)

def updateDakotaFile3(dakotaFilename, quadrature_points):
    """Rewrite number of quadrature points in Dakota file."""

    filein = dakotaFilename
    fileout = dakotaFilename + '.tmp'
    fr = open(filein, 'r')
    fw = open(fileout, 'w')
    for line in fr:
        if 'samples' in line:
            towrite = 'samples  ' + str(quadrature_points) + '\n'
            fw.write(towrite)
        else:
            fw.write(line)
    fr.close()
    fw.close()
    shutil.move(fileout, filein)