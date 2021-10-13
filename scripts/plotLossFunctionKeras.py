#!/usr/bin/python

# given input train log file, extract loss function on validation and training. Feed into python script for plotting

import sys, getopt, os
import matplotlib.pyplot as plt
from subprocess import PIPE, Popen
import numpy as np



def movingAverageConv(a, window_size=1) :
    if window_size == 1:
        return a
    #if not a : return a
    window = np.ones(int(window_size))
    result = np.convolve(a, window, 'full')[ : len(a)] # Convolve full returns array of shape ( M + N - 1 ).
    slotsWithIncompleteConvolution = min(len(a), window_size-1)
    result[slotsWithIncompleteConvolution:] = result[slotsWithIncompleteConvolution:]/float(window_size)
    if slotsWithIncompleteConvolution > 1 :
        divisorArr = np.asarray(range(1, slotsWithIncompleteConvolution+1, 1), dtype=float)
        result[ : slotsWithIncompleteConvolution] = result[ : slotsWithIncompleteConvolution] / divisorArr
    return result

def makeFiles(argv):
    log_file = ''
    w = 1
    v = 1
    t = 1
    #session = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/UNet_v0_TumorSegmenter_breastMask_UNet_v0_TumorSegmenter_DGNS_2020-01-23_1658/'
    save = False
    try:
		opts, args = getopt.getopt(argv,"hsf:m:v:t:",["file=", "movingAverage=","movingAverageVal=","movingAverageTest="])
    except getopt.GetoptError:
		print('plotLossFunctionKeras.py -f <session address : /home/...> -m <moving average> -v <movingAverageVal> -t <movingAverageTest> -s <save>')
		sys.exit(2)
    for opt, arg in opts:
		if opt == '-h':
			print('plotLossFunctionKeras.py -f <train log full file address : /home/...> -m <moving average> -v <movingAverageVal> -t <movingAverageTest> -s <save>')
			sys.exit()
		elif opt in ("-f","--file"):
			session = str(arg)
		elif opt in ("-m","--movingAverage"):
			w = int(arg)   
		elif opt in ("-v","--movingAverageVal"):
			v = int(arg)   
		elif opt in ("-t","--movingAverageTest"):
			t = int(arg)               
		elif opt in ("-s"):
			save = True   

    log_file = [session + '/' + x for x in os.listdir(session) if x.endswith('log') and 'segmentations' not in x][0]


#    bashCommand_getTrain = "grep 'Train cost and metrics' " +  log_file + " | awk '{print $5}' | grep -o '[0-9].*' | sed 's/,//' "
#    bashCommand_getDICE1Train = "grep 'Train cost and metrics' " + log_file +  " | awk '{print $6}' | grep -o '[0-9].*' | sed 's/]//' | sed 's/,//' "
#    bashCommand_getDICE2Train = "grep 'Train cost and metrics' " + log_file +  " | awk '{print $7}' | grep -o '[0-9].*' | sed 's/]//'| sed 's/,//' "
#    p = Popen(bashCommand_getTrain, stdout=PIPE, shell=True)
#    output = p.communicate()
#    train = output[0].split()
#
#    p = Popen(bashCommand_getDICE1Train, stdout=PIPE, shell=True)
#    output = p.communicate()
#    Tdice1 = output[0].split()
#    	
#    p = Popen(bashCommand_getDICE2Train, stdout=PIPE, shell=True)
#    output = p.communicate()
#    Tdice2 = output[0].split()

    bashCommand_getVal = "grep 'Validation cost and accuracy' " +  log_file + " | awk '{print $5}' | grep -o '[0-9].*' | sed 's/,//' "
    bashCommand_getDICE1Val = "grep 'Validation cost and accuracy' " + log_file +  " | awk '{print $7}' | grep -o '[0-9].*' | sed 's/]//' | sed 's/,//'"
    bashCommand_getDICE2Val = "grep 'Validation cost and accuracy' " + log_file +  " | awk '{print $8}' | grep -o '[0-9].*' | sed 's/]//' | sed 's/,//' "


    bashCommand_getDSC = "grep 'Overall DCS' " + log_file +  "  | awk '{print $3}'  | sed 's/]//' "
    bashCommand_getSMOOTH_DSC = "grep 'Overall SMOOTH_DCS' " + log_file +  "  | awk '{print $3}'  | sed 's/]//' "
    bashCommand_get_foreground = "grep 'Epoch_foreground_percent' " + log_file +  "  | awk '{print $2}'  | sed 's/]//' "
    
    
#    bashCommand_indDICE = "grep -A3 'Full segmentation evaluation of' " + log_file +  "  | awk '{print $2}'  | sed 's/]//' "
#    p = Popen(bashCommand_indDICE, stdout=PIPE, shell=True)
#    output = p.communicate()
#    indDice = output[0].split('subject0')
#    [x.split('DCS ')[-1] for x in indDice]

    bashCommand_indDICE = "cat " + log_file 
    p = Popen(bashCommand_indDICE, stdout=PIPE, shell=True)
    output = p.communicate()
    epochs = output[0].split('FULL HEAD SEGMENTATION')
    segmentation_results = [x.split('Overall DCS')[0] for x in epochs ][1:]
    len(segmentation_results)
    epoch = []
    dice_scan_epoch = []
    for i in range(len(segmentation_results)):
        epoch.append([x for x in segmentation_results[i].replace('SMOOTH_DCS','SMOOTH').split('DCS ')])
    for i in range(len(epoch)):    
        dice_scan_epoch.append([float(x[:6]) for x in epoch[i][1:]][:50])
       
    if len(dice_scan_epoch) > 1:
        if len(dice_scan_epoch[-1]) < len(dice_scan_epoch[-2]):
            dice_scan_epoch = dice_scan_epoch[:-1]
                


    train = np.load(session + '/LOSS.npy', allow_pickle=True)
    metric = np.load(session + '/METRICS.npy', allow_pickle=True)
    Tdice1 = metric[:,0]
    Tdice2 = metric[:,1]   
        
    
    
    p = Popen(bashCommand_getVal, stdout=PIPE, shell=True)
    output = p.communicate()
    Val = output[0].split()

    p = Popen(bashCommand_getDICE1Val, stdout=PIPE, shell=True)
    output = p.communicate()
    Valdice1 = output[0].split()

    p = Popen(bashCommand_getDICE2Val, stdout=PIPE, shell=True)
    output = p.communicate()
    Valdice2 = output[0].split()

    p = Popen(bashCommand_getDSC, stdout=PIPE, shell=True)
    output = p.communicate()
    DSC = output[0].split()

    p = Popen(bashCommand_getSMOOTH_DSC, stdout=PIPE, shell=True)
    output = p.communicate()
    SMOOTH_DSC = output[0].split()
    


    p = Popen(bashCommand_get_foreground, stdout=PIPE, shell=True)
    output = p.communicate()
    FG = output[0].split()

    for i in range(0,len(train)-1):
        train[i] = float(train[i])
    train = train[:-1]

    for i in range(0,len(Tdice1)-1):
        Tdice1[i] = float(Tdice1[i])
    Tdice1 = Tdice1[:-1]
    
    for i in range(0, len(Tdice2)-1):
        Tdice2[i] = float(Tdice2[i])
    Tdice2 = Tdice2[:-1]



    for i in range(0, len(Val)-1):
    	Val[i] = float(Val[i])
    Val = Val[:-1]

    for i in range(0, len(Valdice1)):
    	Valdice1[i] = float(Valdice1[i])
    Valdice1 = Valdice1[:-1]
    for i in range(0, len(Valdice2)):
    	Valdice2[i] = float(Valdice2[i])
    Valdice2 = Valdice2[:-1]



    for i in range(0, len(DSC)):
    	DSC[i] = float(DSC[i])
        
    for i in range(0, len(SMOOTH_DSC)):
    	SMOOTH_DSC[i] = float(SMOOTH_DSC[i])

    for i in range(0, len(FG)):
    	FG[i] = float(FG[i])


    train = movingAverageConv(train, window_size = w)
    Tdice1 = movingAverageConv(Tdice1, window_size = w)
    Tdice2 = movingAverageConv(Tdice2, window_size = w)

    Val = movingAverageConv(Val, window_size = v)
    Valdice1 = movingAverageConv(Valdice1, window_size = v)
    Valdice2 = movingAverageConv(Valdice2, window_size = v)
    DSC = movingAverageConv(DSC, window_size = t)
    SMOOTH_DSC = movingAverageConv(SMOOTH_DSC, window_size = t)
    FG = movingAverageConv(FG, window_size = t)

    plt.figure(figsize=(12,12))
    
    ax1 = plt.subplot(321)
    plt.plot(range(len(train)),train,'k-')
    plt.title('Training Data - moving average {}'.format(w),)
    plt.axis('tight')
    plt.legend(('Training Loss',))
    plt.grid(b=True, which='major',zorder=1)
    plt.minorticks_on()
    plt.grid(b=True, which='minor',alpha=0.35, zorder=1) 


    ax2 = plt.subplot(322, sharey = ax1)


    plt.plot(np.linspace(0,len(Val),len(train),endpoint=True), train,'r--')
    plt.plot(range(0,len(Val)),Val,'k-')
    plt.legend(('Training Loss','Validation Loss',))
    plt.title('Validation Data',)
    
    #plt.ylim([0.1,0.5])
    
    plt.grid(b=True, which='major',zorder=1)
    plt.minorticks_on()
    plt.grid(b=True, which='minor',alpha=0.35, zorder=1) 

    ax3 = plt.subplot(323)
    plt.plot(range(len(Tdice1)),Tdice1,'k-')
    plt.plot(range(len(Tdice2)),Tdice2,'b-')
    plt.title('Dice per class')
    plt.legend(('Background','Tumor'), loc='lower center')
    plt.grid(b=True, which='major',zorder=1)
    plt.minorticks_on()
    plt.grid(b=True, which='minor',alpha=0.35, zorder=1) 

    ax4 = plt.subplot(324, sharey = ax3)
    plt.title('Dice per class')
    plt.plot(range(0,len(Valdice1)),Valdice1,'k-')
    plt.plot(range(0,len(Valdice2)),Valdice2,'b')
    plt.grid(b=True, which='major',zorder=1)
    plt.minorticks_on()
    plt.grid(b=True, which='minor',alpha=0.35, zorder=1) 
    plt.legend(('Background','Tumor'), loc='lower center')

    ax5 = plt.subplot(325)#, sharey=ax4)
    plt.plot(range(len(FG)),FG,'k-o')
    
    plt.legend(('Foreground percent',), loc='upper right')
    plt.xlabel('Epochs')
    ax5.set_yscale('log')
    plt.grid(b=True, which='major',zorder=1)
    plt.minorticks_on()
    plt.grid(b=True, which='minor',alpha=0.35, zorder=1) 
    
    ax6 = plt.subplot(326)#, sharey=ax4) 

    #plt.plot(range(len(SMOOTH_DSC)),SMOOTH_DSC,'b-o',alpha=0.8)   
    #DSC = [x*2-0.99 for x in DSC]
    
    plt.plot(range(len(DSC)),DSC,'k-o',alpha=0.8)
    plt.legend(('Dice average both classes',), loc='lower center')
#    if len(dice_scan_epoch) == len(SMOOTH_DSC):
#        plt.boxplot(positions=range(len(SMOOTH_DSC)),x=dice_scan_epoch, showmeans=False,bootstrap=100)
    plt.plot(dice_scan_epoch, alpha=0.7)
    plt.xlabel('Epochs')
    #plt.ylim([0,0.5])

    plt.grid(b=True, which='major',zorder=1)
    #plt.minorticks_on()
    plt.grid(b=True, which='minor',alpha=0.35, zorder=1)     
    
    
    if save:
        #out = '/'.join(file.split('/')[:-1])
        print('saved in {}'.format(session))
        plt.savefig(session+'/Training_session.png')
        plt.clf()

    else:
	#print('bye')
        plt.show()
    
    
if __name__ == "__main__":
	makeFiles(sys.argv[1:])
    
    
    
