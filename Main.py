import argparse
import random
import torchvision.transforms as transforms

from framework_application import *
#from model_evaluation import *
print("Main")

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('--train', type=str, default='False')


    
args = parser.parse_args()
print(args)

    
if args.apply_framework == 'True':
    print("test")
    test_framework()
    print("done")
    