import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats as stat


#effect SS, grand mean - condition mean
#subejct SS, grand mean - subject mean
#error SS within, individual - condition mean
#residual, error SS within- subject SS

def rm_anova_2(df, str_con1, str_con2, str_subject, str_value):
    """df, pandas dataframe
    str_con1, variable1
    str_con2, variable2, where repeated measures happen
    str_subject, variable that identify subjects
    str_value, measurement
    """
    ss_1 = 0; ss_ef1 = 0; n1 = 0
    ss_2 = 0; ss_ef2 = 0; n2 = 0
    ss_3 = 0; ss_ef3 = 0; ns = 0
    ss_i = 0; ss_efi = 0
    ss_re = 0; ss_efre = 0; ss_efrei = 0
    
    g_1 = df.groupby(str_con1)
    g_2 = df.groupby(str_con2)
    g_3 = df.groupby(str_subject)
    
    ss = ((df[str_value] - df[str_value].mean())**2).sum()

    for x, y in g_1:
        ss_1 += y[str_value].sum()**2/len(y)
        n1 +=1

    for x, y in g_2:
        ss_2 += y[str_value].sum()**2/len(y)
        n2 +=1

    for x1, y1 in g_1: 
        for x2, y2 in y1.groupby(str_con2):
            ss_i += y2[str_value].sum()**2/len(y2)

    for x1, y1 in g_3:
        for x2, y2 in y1.groupby(str_con1):
            ss_re += y2[str_value].sum()**2/len(y2)

    ns = len(df)/n1/n2
    
    sst = df[str_value].sum()**2/(len(df))
    ss_ef1 = ss_1 - sst
    ss_ef2 = ss_2 - sst
    ss_efi = ss_i - ss_ef1 - ss_ef2 - sst      
    ss_efre = ss_re - ss_1
    ss_efrei = (df[str_value]**2).sum() - ss_i - ss_re + ss_1
    
    n1 = n1-1
    n2 = n2-1
    ni = n1*n2
    nr = ns*n1*n2
    
    ms1 = ss_ef1/n1
    ms2 = ss_ef2/n2
    msi = ss_efi/n1/n2
    msre = ss_efre/(n1+1)/(ns-1)
    msrei = ss_efrei/(n1+1)/(ns-1)/n2
    
    f1 = ms1/msre
    f2 = ms2/msrei
    fi = msi/msrei
    fre = msre/msrei
    
    columns = ['MS', 'df', 'F', 'PR(>F)']
    table = pd.DataFrame(np.zeros((5, 4)), columns = columns)
    table = table.rename(index={0:str_con1, 1:str_con2, 2:str_con1+':'+str_con2, 
                                3:'subject', 4:'subject/repeat'})              
    
    table.loc[str_con1,'MS'] = ms1
    table.loc[str_con2,'MS'] = ms2
    table.loc[str_con1+':'+str_con2,'MS'] = msi
    table.loc['subject','MS'] = msre
    table.loc['subject/repeat','MS'] = msrei
    
    table.loc[str_con1,'df'] = n1
    table.loc[str_con2,'df'] = n2
    table.loc[str_con1+':'+str_con2,'df'] = ni
    table.loc['subject','df'] = (n1+1)*(ns-1)
    table.loc['subject/repeat','df'] = (n1+1)*(ns-1)*n2
    
    table.loc[str_con1,'F'] = f1
    table.loc[str_con2,'F'] = f2
    table.loc[str_con1+':'+str_con2,'F'] = fi
    table.loc['subject','F'] = fre
    table.loc['subject/repeat','F'] = 'Nan'
    
    table.loc[str_con1,'PR(>F)'] = stats.f.sf(f1, n1, (n1+1)*(ns-1))
    table.loc[str_con2,'PR(>F)'] = stats.f.sf(f2, n2, (n1+1)*(ns-1)*n2)
    table.loc[str_con1+':'+str_con2,'PR(>F)'] = stats.f.sf(fi, ni, (n1+1)*(ns-1)*n2)
    table.loc['subject','PR(>F)'] = stats.f.sf(fre, (n1+1)*(ns-1), (n1+1)*(ns-1)*n2)
    table.loc['subject/repeat','PR(>F)'] = 'Nan'
       
    return table

def rm_anova_3(df, str_con1, str_con2, str_con3, str_subject, str_value):
    """df, pandas dataframe
    str_con1, variable1
    str_con2, variable2, where repeated measures happen
    str_con3, variable3, where repeated measures happen
    str_subject, variable that identify subjects
    str_value, measurement
    """
    ss_1 = 0; ss_ef1 = 0; n1 = 0
    ss_2 = 0; ss_ef2 = 0; n2 = 0
    ss_3 = 0; ss_ef3 = 0; n3 = 0
    ss_4 = 0; ss_efs = 0; ns = 0 
    ss_i1 = 0; ss_efi1 = 0
    ss_i2 = 0; ss_efi2 = 0
    ss_i3 = 0; ss_efi3 = 0
    ss_i = 0; ss_efi = 0
    ss_re = 0; ss_re2 = 0; ss_re3 = 0
    ss_efre = 0; ss_efre1 = 0; ss_efre2 = 0; ss_efrei = 0
    
    g_1 = df.groupby(str_con1)
    g_2 = df.groupby(str_con2)
    g_3 = df.groupby(str_con3)
    g_4 = df.groupby(str_subject)
    
    ss = ((df[str_value] - df[str_value].mean())**2).sum()
    
    for x, y in g_1:
        ss_1 += y[str_value].sum()**2/len(y)
        n1 +=1

    for x, y in g_2:
        ss_2 += y[str_value].sum()**2/len(y)
        n2 +=1

    for x, y in g_3:
        ss_3 += y[str_value].sum()**2/len(y)
        n3 +=1
        
    for x1, y1 in g_1: 
        for x2, y2 in y1.groupby(str_con2):
            ss_i1 += y2[str_value].sum()**2/len(y2)
            
    for x1, y1 in g_1: 
        for x2, y2 in y1.groupby(str_con3):
            ss_i2 += y2[str_value].sum()**2/len(y2)
            
    for x1, y1 in g_2: 
        for x2, y2 in y1.groupby(str_con3):
            ss_i3 += y2[str_value].sum()**2/len(y2)
            
    for x1, y1 in g_1: 
        for x2, y2 in y1.groupby(str_con2):
            for x3, y3 in y2.groupby(str_con3):
                ss_i += y3[str_value].sum()**2/len(y3)

    for x1, y1 in g_4:
        for x2, y2 in y1.groupby(str_con1):
            ss_re += y2[str_value].sum()**2/len(y2)
        for x3, y3 in y1.groupby(str_con2):
            ss_re2 += y3[str_value].sum()**2/len(y3)
        for x4, y4 in y1.groupby(str_con3):
            ss_re3 += y4[str_value].sum()**2/len(y4)
            
    ns = len(df)/n1/n2/n3
    
    sst = df[str_value].sum()**2/(len(df))
    ss_ef1 = ss_1 - sst
    ss_ef2 = ss_2 - sst
    ss_ef3 = ss_3 - sst
    ss_efi1 = ss_i1 - ss_ef1 - ss_ef2 - sst 
    ss_efi2 = ss_i2 - ss_ef1 - ss_ef3 - sst
    ss_efi3 = ss_i3 - ss_ef2 - ss_ef3 - sst
    ss_efi = ss_i - ss_i1 - ss_i2 - ss_i3 + ss_1 + ss_2 + ss_3 - sst
    ss_efre = ss_re - ss_1
    ss_efre1 = ss_re2 - ss_i1 - ss_re + ss_1
    ss_efre2 = ss_re3 - ss_i2 - ss_re + ss_1
    ss_efrei = (df[str_value]**2).sum() - ss_i - ss_re2 - ss_re3 - ss_1 + ss_re + ss_i1 + ss_i2

    n1 = n1-1
    n2 = n2-1
    n3 = n3-1
    ni1 = n1*n2
    ni2 = n1*n3
    ni3 = n2*n3
    ni = n1*n2*n3
    
    ms1 = ss_ef1/n1
    ms2 = ss_ef2/n2
    ms3 = ss_ef3/n3
    msi1 = ss_efi1/ni1
    msi2 = ss_efi2/ni2
    msi3 = ss_efi3/ni3
    msi = ss_efi/ni
    msr = ss_efre/(n1+1)/(ns-1)
    msre1 = ss_efre1/(n1+1)/(ns-1)/n2
    msre2 = ss_efre2/(n1+1)/(ns-1)/n3
    msrei = ss_efrei/(n1+1)/(ns-1)/ni3
    
    f1 = ms1/msr
    f2 = ms2/msre1
    f3 = ms3/msre2
    fi1 = msi1/msre1
    fi2 = msi2/msre2
    fi3 = msi3/msrei
    fi = msi/msrei
    
    columns = ['MS', 'df', 'F', 'PR(>F)']
    table = pd.DataFrame(np.zeros((11, 4)), columns = columns)
    table = table.rename(index={0:str_con1, 1:str_con2, 2:str_con3, 
                                3:str_con1+':'+str_con2, 4:str_con1+':'+str_con3,
                                5:str_con2+':'+str_con3, 6:str_con1+':'+str_con2+':'+str_con3,
                                7:'subject', 8:str_con2+':subject', 9:str_con3+':subject',
                                10:str_con2+':'+str_con3+':subject'})              
    
    table.loc[str_con1,'MS'] = ms1
    table.loc[str_con2,'MS'] = ms2
    table.loc[str_con3,'MS'] = ms3
    table.loc[str_con1+':'+str_con2,'MS'] = msi1
    table.loc[str_con1+':'+str_con3,'MS'] = msi2
    table.loc[str_con2+':'+str_con3,'MS'] = msi3
    table.loc[str_con1+':'+str_con2+':'+str_con3,'MS'] = msi
    table.loc['subject','MS'] = msr
    table.loc[str_con2+':subject','MS'] = msre1
    table.loc[str_con3+':subject','MS'] = msre2
    table.loc[str_con2+':'+str_con3+':subject','MS'] = msrei
    
    table.loc[str_con1,'df'] = str(n1)+','+str((n1+1)*(ns-1))
    table.loc[str_con2,'df'] = str(n2)+','+str((n1+1)*(ns-1)*n2)
    table.loc[str_con3,'df'] = str(n3)+','+str((n1+1)*(ns-1)*n3)
    table.loc[str_con1+':'+str_con2,'df'] = str(ni1)+','+str((n1+1)*(ns-1)*n2)
    table.loc[str_con1+':'+str_con3,'df'] = str(ni2)+','+str((n1+1)*(ns-1)*n3)
    table.loc[str_con2+':'+str_con3,'df'] = str(ni3)+','+str((n1+1)*(ns-1)*ni3)
    table.loc[str_con1+':'+str_con2+':'+str_con3,'df'] = str(ni)+','+str((n1+1)*(ns-1)*ni3)
    table.loc['subject','df'] = (n1+1)*(ns-1)
    table.loc[str_con2+':subject','df'] = (n1+1)*(ns-1)*n2
    table.loc[str_con3+':subject','df'] = (n1+1)*(ns-1)*n3
    table.loc[str_con2+':'+str_con3+':subject','df'] = (n1+1)*(ns-1)*ni3
    
    table.loc[str_con1,'F'] = f1
    table.loc[str_con2,'F'] = f2
    table.loc[str_con3,'F'] = f3
    table.loc[str_con1+':'+str_con2,'F'] = fi1
    table.loc[str_con1+':'+str_con3,'F'] = fi2
    table.loc[str_con2+':'+str_con3,'F'] = fi3
    table.loc[str_con1+':'+str_con2+':'+str_con3,'F'] = fi
    table.loc['subject','F'] = 'Nan'
    table.loc[str_con2+':subject','F'] = 'Nan'
    table.loc[str_con3+':subject','F'] = 'Nan'
    table.loc[str_con2+':'+str_con3+':subject','F'] = 'Nan'
    
    table.loc[str_con1,'PR(>F)'] = stats.f.sf(f1, n1, (n1+1)*(ns-1))
    table.loc[str_con2,'PR(>F)'] = stats.f.sf(f2, n2, (n1+1)*(ns-1)*n2)
    table.loc[str_con3,'PR(>F)'] = stats.f.sf(f3, n3, (n1+1)*(ns-1)*n3)
    table.loc[str_con1+':'+str_con2,'PR(>F)'] = stats.f.sf(fi1, ni1, (n1+1)*(ns-1)*n2)
    table.loc[str_con1+':'+str_con3,'PR(>F)'] = stats.f.sf(fi1, ni2, (n1+1)*(ns-1)*n3)
    table.loc[str_con2+':'+str_con3,'PR(>F)'] = stats.f.sf(fi1, ni3, (n1+1)*(ns-1)*ni3)
    table.loc[str_con1+':'+str_con2+':'+str_con3,'PR(>F)'] = stats.f.sf(fi, ni, (n1+1)*(ns-1)*ni3)
    table.loc['subject','PR(>F)'] = 'Nan'
    table.loc[str_con2+':subject','PR(>F)'] = 'Nan'
    table.loc[str_con3+':subject','PR(>F)'] = 'Nan'
    table.loc[str_con2+':'+str_con3+':subject','PR(>F)'] = 'Nan'
       
    return table
