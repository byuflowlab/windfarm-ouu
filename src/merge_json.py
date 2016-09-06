import json
from collections import OrderedDict
import argparse


def merge_combined_record(args):

    jsonfile = open(args.jsonfile1, 'r')
    a = json.load(jsonfile)
    jsonfile.close()
    jsonfile = open(args.jsonfile2, 'r')
    b = json.load(jsonfile)
    jsonfile.close()
    # print a.keys()
    # print b.keys()
    c = {}
    assert a.keys() == b.keys(), 'The json files should have the same entries'
    for k in a:
        assert a[k].keys() == b[k].keys(), 'The json files should have the same entries'
        # print a[k].keys()
        c[k] = a[k]
        for kk in a[k]:
            assert a[k][kk].keys() == b[k][kk].keys(), 'The json files should have the same entries'
            # print a[k][kk].keys()
            c[k][kk] = a[k][kk]
            for kkk in a[k][kk]:
                # print a[k][kk][kkk]
                # Here combine the records
                c[k][kk][kkk] = a[k][kk][kkk] + b[k][kk][kkk]
                # Remove duplicates and preserve order
                # http://stackoverflow.com/questions/7961363/removing-duplicates-in-lists
                c[k][kk][kkk] = list(OrderedDict.fromkeys(c[k][kk][kkk]))

    jsonfileout = '2d_dakota_merged.json'
    jsonfile = open(jsonfileout, 'w')
    json.dump(c, jsonfile, indent=2)
    jsonfile.close()
    print jsonfileout + ' written'


def merge_simple_record(args):

    jsonfile = open(args.jsonfile1, 'r')
    a = json.load(jsonfile)
    jsonfile.close()
    jsonfile = open(args.jsonfile2, 'r')
    b = json.load(jsonfile)
    jsonfile.close()

    c = {}
    assert a.keys() == b.keys(), 'The json files should have the same entries'
    for k in a:
        # For these keys combine the entries
        # If you didn't run verbose, this doesn't combine properly for the winddirections, windspeeds, power and power_approx
        if k in ['std', 'samples', 'mean', 'winddirections', 'windspeeds', 'power', 'power_approx']:
            c[k] = a[k] + b[k]
            # Remove duplicates and preserve order
            # Can't use the above solution because for the power_approx I have a list of lists.
            d = []
            for entry in c[k]:
                if entry not in d:
                    d.append(entry)
            c[k] = d
            
        # For these keys make sure they are the same
        if k in ['layout', 'uncertain_variable', 'method', 'Noffset', 'offset', 'windspeeds_approx', 'winddirections_approx']:
            assert a[k] == b[k], 'These entries should be the same'
            c[k] = a[k]

    jsonfileout = 'record_merged.json'
    jsonfile = open(jsonfileout, 'w')
    json.dump(c, jsonfile, indent=2)
    jsonfile.close()
    print jsonfileout + ' written'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine json files')
    # Positional argument averiguar.
    parser.add_argument('jsonfile1')
    parser.add_argument('jsonfile2')
    parser.add_argument('option', help='1 or 2, 1-merge an already combined record, 2-merge the simple record')
    parser.add_argument('--version', action='version', version='Combine json files 0.0')
    args = parser.parse_args()
    if args.option == "1":
        merge_combined_record(args)
    elif args.option == "2":
        merge_simple_record(args)
    else:
        raise ValueError('unknown option %s, valid options 1 or 2.' % args.option)

