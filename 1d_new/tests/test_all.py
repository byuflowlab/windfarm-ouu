# content of test_rectangle.py
from statistics_convergence_fortest import run
import json

def assertions(test, baseline):
    assert test['samples'] == baseline['samples']
    assert test['method'] == baseline['method']
    assert test['uncertain_variable'] == baseline['uncertain_variable']
    assert test['mean'] == baseline['mean']
    assert test['std'] == baseline['std'] 
    assert test['power'] == baseline['power']
    assert test['winddirections'] == baseline['winddirections']
    assert test['windspeeds'] == baseline['windspeeds']
    assert test == baseline

def test_rectangle_speed():
    n = 20
    method_dict = {'method': 'rect', 'uncertain_var': 'speed'}
    jsonfile = open('tests/record_test_rectangle_speed.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    run(method_dict, n)
    jsonfile = open('record.json','r')
    test = json.load(jsonfile)
    jsonfile.close()
    assertions(test, baseline)

def test_rectangle_direction():
    n = 20
    method_dict = {'method': 'rect', 'uncertain_var': 'direction'}
    jsonfile = open('tests/record_test_rectangle_direction.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    run(method_dict, n)
    jsonfile = open('record.json','r')
    test = json.load(jsonfile)
    jsonfile.close()
    assertions(test, baseline)

def test_dakota_speed():
    n = 5
    method_dict = {'method': 'dakota', 'uncertain_var': 'speed',
                   'dakota_filename': 'dakotaAEPspeed.in'}
    jsonfile = open('tests/record_test_dakota_speed.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    run(method_dict, n)
    jsonfile = open('record.json','r')
    test = json.load(jsonfile)
    jsonfile.close()
    assertions(test, baseline)

