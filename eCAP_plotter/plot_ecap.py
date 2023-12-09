# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:08:52 2023

@author: SPARC_PSOCT_MGH
"""

def plot_ecap(**kwargs):
    # import pdb; pdb.set_trace()
    for key, value in kwargs.items():
    # Assign variables dynamically using exec()
        locals()[key] = value
    
    #import pylustrator
    #pylustrator.start()
    if PlotAggreConds:
        fig, ax = plt.subplots(figsize=figsz)
    fileCnt = -1
    for fileind in filelist1:
        fileCnt = fileCnt+1
        data_CAP = pyCAP.Ephys(Dir[ExpNo-1,fileind], stores=['CAPc'],sample_delay=delay);
        if isinstance(common_reference,str) & (not common_reference == 0):
            data_CAP = data_CAP.common_reference(method=common_reference)
        stim_data = pyCAP.Stim(Dir[ExpNo-1,fileind])
        if Subtract_SA:
            data_SA = pyCAP.Ephys(Dir[ExpNo-1,SA_fileind], stores=['CAPc'],sample_delay=delay);
            stim_data_SA = pyCAP.Stim(Dir[ExpNo-1,SA_fileind])
        # else:
        #     data_SA = data_CAP
        #     stim_data_SA = stim_data
        if pl_on:
            data_CAP = data_CAP.filter_powerline()   #60 default
            if Subtract_SA:
                data_SA = data_SA.filter_powerline()   #60 default
        if hp_on:
            data_CAP = data_CAP.filter_median(btype='highpass')   #300 default
            if Subtract_SA:
                data_SA = data_SA.filter_median(btype='highpass')   #300 default
        if lp_on:
            data_CAP = data_CAP.filter_gaussian(lpff, btype='lowpass')    #5000 default
            if Subtract_SA:
                data_SA = data_SA.filter_gaussian(lpff, btype='lowpass')    #5000 default
        ecap_data_CAP_filt = pyCAP.ECAP(data_CAP,stim_data)
        if Subtract_SA:
            ecap_data_SA_filt = pyCAP.ECAP(data_SA,stim_data_SA)
    
        
    #     data_EMG = pyCAP.Ephys(Dir[ExpNo-1,fileind], stores=['EMGG'],sample_delay=delay);
    #     if hp_on:
    #         data_EMG = data_EMG.filter_iir(Wn = hpff, btype='highpass', order=hpfo)   #300 default
    #     if pl_on:
    #         data_EMG = data_EMG.filter_powerline(frequencies=[60])   #60 default
    #     if lp_on:
    #         data_EMG = data_EMG.filter_gaussian(lpff, btype='lowpass', order=0, truncate=4.0)    #5000 default
    #     ecap_data_EMG_filt = pyCAP.ECAP(data_EMG,stim_data)
    
        Amp = pd.DataFrame(pyCAP.Stim(Dir[ExpNo-1,fileind]).parameters, columns=['pulse amplitude (μA)']).to_numpy()
        Amp = Amp.tolist()
        Pulses = pd.DataFrame(pyCAP.Stim(Dir[ExpNo-1,fileind]).parameters, columns=['pulse count']).to_numpy()
        Pulses = Pulses.tolist()
        if not PlotAggreConds:
            fig, ax = plt.subplots(figsize=figsz)
        Cnt_par = -1
        Amp_arr_sz = np.shape(Amp); 
        # par_list = range(0, Amp_arr_sz[0])
        if not PlotAggrePara:
            par_list1 = [par_list1_[fileCnt]]
        else:
            par_list1 = par_list1_
        legend_added = [False] * len(ch_eCAP)
        for par_ind in par_list1:
            Cnt_par = Cnt_par+1
            Cnt_ref_ch = -1
            for ch in ch_eCAP:
                ch=ch-1
                color = next(color_cycle)
                Cnt_ref_ch = Cnt_ref_ch+1
                index_ar = (0,par_ind)
                for polarity_ in polarity:     
                    # Pulse_ind=np.arange(polarity_,Pulses[par_ind][0],seperate)
                    bipo_posi_ch=np.median(ecap_data_CAP_filt.array(index_ar)[Pulse_Ind,ch],axis=0)
                    if gain[Cnt_ref_ch]!=0:
                        bipo_nega_ch=np.median(ecap_data_CAP_filt.array(index_ar)[Pulse_Ind,ch_eCAP_ref[Cnt_ref_ch]-1],axis=0)    
                        trace=flipgain[Cnt_ref_ch]*(bipo_posi_ch-(bipo_nega_ch*gain[Cnt_ref_ch]))
                    else:
                        trace=bipo_posi_ch
                    if Subtract_SA:
                        if Subtract_SA & (not PlotAggrePara):
                            par_ind_SA = par_list1_[SA_fileind]
                            index_ar_SA = (0,par_list1_[SA_fileind])
                        else:
                            par_ind_SA = par_ind
                            index_ar_SA = index_ar
                        # Pulse_ind = np.arange(polarity_,Pulses[par_ind_SA][0],seperate)
                        bipo_posi_ch_SA=np.median(ecap_data_SA_filt.array(index_ar_SA)[Pulse_Ind,ch],axis=0)
                        bipo_nega_ch_SA=np.median(ecap_data_SA_filt.array(index_ar_SA)[Pulse_Ind,ch_eCAP_ref[Cnt_ref_ch]-1],axis=0)         
                        trace_SA=flipgain[Cnt_ref_ch]*(bipo_posi_ch_SA-(bipo_nega_ch_SA*gain[Cnt_ref_ch]))
                        trace=trace-trace_SA     
                    if PlotAggreConds:
                        plt.plot(ecap_data_CAP_filt.time(index_ar)*1e3,trace*1e6-eCAPspacing*(fileCnt-len(filelist1)/2), label=Status[ExpNo-1,fileind] + str(Amp[par_ind][0]) + "uA, (ch" + str(ch+1) + "-ch" + str(ch_eCAP_ref[Cnt_ref_ch]) + "*" + str(gain[Cnt_ref_ch]) + ")*" + str(flipgain[Cnt_ref_ch]) + " p=" +str(polarity_), alpha=0.7)
                        plt.plot(trace*0 -eCAPspacing*(fileCnt-len(filelist1)/2),'--',color='gray', alpha=0.25);
                    elif PlotAggrePara:
                        if not legend_added[ch]:
                            label_text = "ch" + str(ch + 1)
                            legend_added[ch] = True
                        else:
                            label_text = None
                        plt.plot(ecap_data_CAP_filt.time(index_ar)*1e3,trace*1e6-eCAPspacing*(par_ind-Amp_arr_sz[0]/2), label= label_text, color=color, alpha=0.7)
                        plt.plot(trace*0 -eCAPspacing*(par_ind-Amp_arr_sz[0]/2),'--',color='gray', alpha=0.25);
                    else:
                        plt.plot(ecap_data_CAP_filt.time(index_ar)*1e3,trace*1e6-eCAPspacing*(ch-len(ch_eCAP)/2), label=  "(ch" + str(ch+1) + "-ch" + str(ch_eCAP_ref[Cnt_ref_ch]) + "*" + str(gain[Cnt_ref_ch]) + ")*" + str(flipgain[Cnt_ref_ch]) + " p=" +str(polarity_), alpha=0.7)
                        plt.plot(trace*0 -eCAPspacing*(ch-len(ch_eCAP)/2),'--',color='gray', alpha=0.25);
                    # Add text to the right side of each parameter's dashed line
                    text_x = eCAP_x_lim_range[1]  # Right end of the x-axis range
                    text_y = -eCAPspacing * (par_ind - Amp_arr_sz[0] / 2)
                    plt.text(text_x*1.01, text_y, str(Amp[par_ind][0]) + "uA", fontsize=lgfnt, va='center', ha='left')
    
            # for ch in ch_EMG:
            #     ch=ch-1;
            #     index_ar = (0,par_ind)
            #     trace=np.median(ecap_data_EMG_filt.array(index_ar)[np.arange(polarity_,Pulses[par_ind][0],seperate),ch],axis=0)
            #     plt.plot(ecap_data_EMG_filt.time(index_ar)*1e3,trace*EMGgain*1e6-EMGspacing*(par_ind-Amp_arr_sz[0]/2), label="EMG" + str(Amp[par_ind][0]) + "uA, ch" + str(ch+1) + "*" + str(EMGgain), alpha=0.7)
            # print("file & parameter = "+ str(index_ar))
        if not PlotAggreConds:
            plt.title(Status[ExpNo-1,fileind],fontsize=32)
        plt.xlabel('Time (ms)',fontsize=25)
        plt.ylabel('eCAP Voltage (uV)',fontsize=25)
        plt.xlim(eCAP_x_lim_range)
        plt.ylim(eCAP_y_lim_range)
        plt.legend(loc=lc,fontsize=lgfnt)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25 )
        if not PlotAggreConds:
            plt.show()
    if PlotAggreConds:    
        plt.show()