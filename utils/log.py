import sys

"""
ref: https://github.com/A2Zadeh/CMU-MultimodalSDK
"""

__all__ = ['status', 'error', 'success']

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def status(msgstring):
	print (bcolors.OKBLUE +bcolors.BOLD+"<Status>: "+bcolors.ENDC + msgstring)

def error(msgstring,error=False,errorType=RuntimeError):
	action,msgstart=(lambda x:(_ for _ in ()).throw(errorType(x)),"<Error>: ") if error else (lambda x:sys.stdout.write(str(x)+'\n'),bcolors.WARNING+bcolors.BOLD+"<Warning>: "+bcolors.ENDC)
	action("%s%s"%(msgstart,msgstring))

def success(msgstring):
	print(bcolors.OKGREEN+bcolors.BOLD+"<Success>: "+ bcolors.ENDC+msgstring)

	
