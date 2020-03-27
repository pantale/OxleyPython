# -*- coding: utf-8 -*-

# Initialisation
import math
import numpy as np
import pylab

# Import LMFIT
import lmfit

# Définition des paramètres de coupe
V = 200/60
alpha = -7*math.pi/180
w = 1.6e-3
t1 = 0.15e-3

# définition des paramètres matériau
T0 = 273.15
Tw = T0 + 25
rho = 8000.0 ##valeur réelle 7862
eta = 0.9
psi = 0.9
A = 553.1e6
B = 600.8e6
n = 0.234
C = 0.0134
m = 1
Tm = 1460 + T0
Epsp0 = 1

nLoops_1 = 0
nLoops_2 = 0

def misesEquivalent(eps,epsp,T):
    return (A + B*eps**n) * (1.0 + C*math.log(epsp/Epsp0)) * (1.0 - ((T-Tw)/(Tm-Tw))**m)
    
def KLaw(T):
    return 52.61 - 0.0281*(T - T0)
    
def CpLaw(T):
    return 420 + 0.504*(T - T0)
       
# Fonction de calcul de TAB, renvoie 0 si temperature au dela de la TF
def computeAB(): 
    # TAB égale à Tw
    TAB = Tw
    #Initialisation du test de convergence 
    maxLoops = 1000
    # Evaluation de la température issue de la dissipation plastique 
    while (maxLoops>0) :
        # Calcul de Cp et K pour température TAB 
        Cp = CpLaw(TAB)
        K = KLaw(TAB)
        # Calcul de la loi contrainte d'écoulement du matériau 
        kAB = (1.0/math.sqrt(3))*misesEquivalent(EpsAB,EpspAB,TAB)
        # Calcul de Fs 
        Fs = kAB*lAB*w
         # Calcul de RT 
        RTtanPhi = math.tan(phi)*(rho*Cp*V*t1)/K
        # Calcul de beta 
        if (RTtanPhi>10) :
            beta = 0.3 - 0.15*math.log10(RTtanPhi)
        else :
            beta = 0.5 - 0.35*math.log10(RTtanPhi)
        # Calcul de l'écart de température 
        deltaTsz = ((1.0-beta)*Fs*Vs)/(mchip*Cp)
        # Calcul de la nouvelle TAB 
        NewTAB = Tw + eta*deltaTsz
        # Test de limite sur TAB si TAB est supérieure à Tm 
        if (NewTAB>Tm) : return 0,0,0,0
        # Test de la condition de sortie 
        if (abs(NewTAB - TAB) <= 1e-3) :
            return NewTAB, Fs, deltaTsz, kAB
        # Affectation nouvelle TAB 
        TAB = NewTAB
        maxLoops -= 1
    # Pas de convergence ? 
    return 0,0,0,0
           
# Fonction de calcul de TC
def computeTc(deltaTsz): 
#    global deltaTc
    Tc = Tw + deltaTsz
    while (True) :
        # Calcul de Cp et K pour température Tc 
        Cp = CpLaw(Tc)
#        K = KLaw(Tc) Not used here
        # Delta de température à l'interface 
        deltaTc = Ff*Vc/(mchip*Cp)
        # Nouvelle température d'interface 
        NewTc = Tw + deltaTsz + deltaTc
        # Test de la condition de sortie 
        if (abs(NewTc - Tc) <= 1e-3) :
            return Tc, deltaTc
        # Affectation nouvelle Tc 
        Tc = NewTc
    
def compute_Toint_Kchip():
    global lc, neq, EpsAB, EpspAB, TAB, lAB, Vs, Ff, Fn, Fc, Ft, Vc, t2, Epsint, Epspint, Tint, R, kAB, Lambda, theta
    # Longueur de la bande de cisaillement primaire lAB
    lAB = t1/math.sin(phi)   
    # Vitesse le long de la bande de cisaillement primaire
    Vs = V*math.cos(alpha)/math.cos(phi-alpha)   
    # Epaisseur du copeau
    t2 = t1*math.cos(phi-alpha)/math.sin(phi)
    # Vitesse du copeau
    Vc = V*math.sin(phi)/math.cos(phi-alpha)
    # Déformation plastique dans la zone AB
    gammaAB = 1/2*math.cos(alpha)/(math.sin(phi)*math.cos(phi-alpha))
    EpsAB = gammaAB/math.sqrt(3)
    # Vitesses de déformation plastique dans la zone AB
    gammapAB = C0*Vs/lAB # not here
    EpspAB = gammapAB/math.sqrt(3)
    # Calcul des inconnues dans la bande primaire
    TAB, Fs, deltaTsz, kAB = computeAB()
    # Si la température est supérieure à la température de fusion du matériau on renvoie 0 et +inf
    if (TAB==0): return 0, 1e10, 0, 0
    # Calcul de neq formulation de Lalwani
    neq = B*EpsAB**n*n/(A + B*EpsAB**n)
    # Calcul de l'angle theta 
    theta = math.atan(1.0 + math.pi/2.0 - 2.0*phi - C0*neq)
    # Calcul de la résultante R fonction de Fs et theta
    R = Fs/math.cos(theta)
    # Calcul de Lambda 
    Lambda = theta + alpha - phi
    # Calcul des efforts
    Ff = R*math.sin(Lambda)
    Fn = R*math.cos(Lambda)
    Fc = R*math.cos(theta - phi)
    Ft = R*math.sin(theta - phi)
    # Calcul de SigmaNp 
    sigmaNmax = kAB*(1.0 + math.pi/2.0 - 2.0*alpha - 2.0*C0*neq)
    # Longueur de contact outil/copeau 
    lc = t1*math.sin(theta)/(math.cos(Lambda)*math.sin(phi))*(1+C0*neq/(3.0*(1.0+2.0*(math.pi/4.0-phi)-C0*neq)))
    # Contrainte de cisaillement à l'interface 
    Toint = Ff/(lc*w)
    # Déformation équivalente à l'interface 
    gammaM = lc/(delta*t2)
    gammaint = 2.0*gammaAB + 0.5*gammaM
    Epsint = gammaint/math.sqrt(3)
    # Taux de déformation équivalente à l'interface 
    gammapint = Vc/(delta*t2)
    Epspint = gammapint/math.sqrt(3)
    # Temperature de contact 
    Tc, deltaTc = computeTc(deltaTsz)
    K = KLaw(Tc)
    Cp = CpLaw(Tc)
    # Calcul de RT (La question est de savoir réellement quelle valeur de K et de Cp on prend ???)
    RT = (rho*Cp*V*t1)/K
    # Calcul de delta Tm 
#    deltaTm = deltaTc*10.0**(0.06-0.195*delta*math.sqrt(RT*t2/lc)+0.5*math.log10(RT*t2/lc))
    deltaTm = deltaTc*10.0**(0.06-0.195*delta*math.sqrt(RT*t2/lc)) * math.sqrt(RT*t2/lc)
    # Température moyenne à l'interface outil/copeau 
    Tint = Tw+deltaTsz+psi*deltaTm
    # Contrainte d'écoulement plastique dans le copeau 
    kchip = (1.0/math.sqrt(3))*misesEquivalent(Epsint,Epspint,Tint)
    # Calcul de SigmaN 
    sigmaN = Fn/(lc*w)
    return Toint, kchip, sigmaN, sigmaNmax
    
# Réinit des valeurs
def reinitParams(paramsOpt1,paramsOpt2):
    paramsOpt2['C0'].value=(paramsOpt2['C0'].max+paramsOpt2['C0'].min)/2
    paramsOpt2['phi'].value=(paramsOpt2['phi'].max+paramsOpt2['phi'].min)/2
    paramsOpt1['delta'].value=(paramsOpt1['delta'].max+paramsOpt1['delta'].min)/2

# fonction optimum
def fittingFunctionInternal(paramsOpt2):
    global C0, phi
    global Toint, kchip, sigmaN, sigmaNmax, nLoops_1
    C0 = paramsOpt2['C0'].value
    phi = paramsOpt2['phi'].value
    # Calcul des paramètres internes pour l'optimisation
    Toint, kchip, sigmaN, sigmaNmax = compute_Toint_Kchip()
    # Test de bug total
    if (Toint==0): print(V*60, t1, 180/math.pi*phi, C0, delta, ' FAILED\n')
    # Calcul du vecteur de Gap pour l'optimisation
    Gap = [(Toint-kchip),(sigmaN-sigmaNmax)]
    nLoops_1 += 1
    return Gap

# fonction optimum
def fittingFunction(paramsOpt1):
    global delta
    global paramsOpt2, nLoops_2
    delta = paramsOpt1['delta'].value
    fitOpt1 = lmfit.minimize(fittingFunctionInternal, paramsOpt2)
    paramsOpt2 = fitOpt1.params
    nLoops_2 += 1
    return [Fc]

def Optimize(fittingFunction):
    global paramsOpt1, nLoops_1, nLoops_2
    nLoops_1 = 0
    nLoops_2 = 0
    myfit = lmfit.minimize(fittingFunction, paramsOpt1)
    paramsOpt1 = myfit.params
    return myfit

def analyseError(data,slp,var):
    global dat
    dat = data[:,3]
    slope=(dat.max()-dat.min())/slp+dat.min()
    st = 0
    sp = len(dat)-1
    for d in dat:
        if (d<=slope): break
        st+=1
    for d in reversed(dat):
        if (d<=slope): break
        sp-=1
    rng0,rng1 = data[:,0][0],data[:,0][-1]
    sst = st*((rng1-rng0)/len(dat))+rng0
    ssp = sp*((rng1-rng0)/len(dat))+rng0
    print('Frame of values ',var,'>',slope,'[',sst,'-',ssp,']',dat.min())

def getParameter(par):
    if (par in paramsOpt1.valuesdict()) :
        return paramsOpt1[par]
    if (par in paramsOpt2.valuesdict()):
        return paramsOpt2[par]
    print("Parameter",par,"is not in any of the parameters defined\n")
    return
   
def parameterStudy(par,var):
    global tab, phiVary, deltaVary, C0Vary
    getParameter(par).vary = False    
    factor = 1
    if (par=='phi') : factor = 180/math.pi
    tab = np.array([[0,0,0,0,0,0,0]])
    xs = np.linspace(getParameter(par).min,getParameter(par).max,100,True)
    reinitParams(paramsOpt1,paramsOpt2)
    for x in xs:
        getParameter(par).value = x
        # find the solution
        myfit = Optimize(fittingFunction)
        aa = np.array([factor*x, Fc, Ft, math.sqrt((Toint-kchip)**2+(sigmaN-sigmaNmax)**2), C0, delta, 180/math.pi*phi])
        # Corretion for delta
        if (par=='delta') :
            if (math.sqrt((Toint-kchip)**2+(sigmaN-sigmaNmax)**2)>1) : aa[3]=0
        tab = np.vstack((tab,aa))
    tab = np.delete(tab,0,0)
    # Plot the first Figure
    pylab.rc('text', usetex = True)
    pylab.rcParams['xtick.labelsize'] = 14
    pylab.rcParams['ytick.labelsize'] = 14
    fig, ax1 = pylab.subplots(figsize = (11.69,8.27))
    ax2 = ax1.twinx()
    ax2.ticklabel_format(style='sci', scilimits=(-2,2))
    ax1.grid(True)
    p_Fc, = ax1.plot(tab[:,0],tab[:,1],'r',linewidth = 2)
    p_Fn, = ax1.plot(tab[:,0],tab[:,2],'b',linewidth = 2)
    p_C, = ax2.plot(tab[:,0],tab[:,3],'g',linewidth = 2)
    locY = max(np.amax(tab[:,1]),np.amax(tab[:,2]))
    if (par=='C0'):
        pylab.axvline(x = C0_opt, color='k')
        ax1.annotate(' $C_0='+str(round(C0_opt,3))+'$', xy=(C0_opt, locY),fontsize = 18)
        ax1.set_xlim([getParameter(par).min,getParameter(par).max])
    if (par=='delta'):
        pylab.axvline(x = delta_opt, color='k')
        ax1.annotate(' $\delta='+str(round(delta_opt,3))+'$', xy=(delta_opt, locY),fontsize = 18)
        ax1.set_xlim([getParameter(par).min,getParameter(par).max])
    if (par=='phi'):
        pylab.axvline(x = 180/math.pi*phi_opt, color='k')
        ax1.annotate(' $\phi='+str(round(180/math.pi*phi_opt,3))+'$', xy=(180/math.pi*phi_opt, locY),fontsize = 18)
        ax1.set_xlim([180/math.pi*getParameter(par).min,180/math.pi*getParameter(par).max])
    ax1.set_xlabel(var, fontsize = 18)
    ax1.set_ylabel('$Cutting\ forces\ F_{C},\ F_{T}$', fontsize = 18)
    ax2.set_ylabel('$Error\ \Delta F$', fontsize = 18)
    pylab.legend(handles=[p_Fc,p_Fn,p_C], labels=['$F_C$','$F_T$','$\Delta F$'], fontsize = 16,  fancybox = True, shadow = True, frameon = True, loc = 'lower right')
    pylab.savefig(par+'.svg', transparent=True, bbox_inches='tight', pad_inches=0)
    pylab.show()
    #Plot the second figure
    fig, ax1 = pylab.subplots(figsize = (11.69,8.27))
#    pylab.rc('text', usetex = True)
#    pylab.rcParams['xtick.labelsize'] = 14
#    pylab.rcParams['ytick.labelsize'] = 14
    ax2 = ax1.twinx()
    ax1.grid(True)
    if (par=='C0'):
        locY = np.amax(tab[:,5])
        pylab.axvline(x = C0_opt, color='k')
        ax1.annotate(' $C_0='+str(round(C0_opt,3))+'$', xy=(C0_opt, locY),fontsize = 18)
        ax1.set_xlim([getParameter(par).min,getParameter(par).max])
        p_C1, = ax1.plot(tab[:,0],tab[:,5],'r',linewidth = 2)
        p_C2, = ax2.plot(tab[:,0],tab[:,6],'b',linewidth = 2)
        ax1.set_ylabel('$Plastic\ zone\ thickness\ \delta$', fontsize = 18)
        ax2.set_ylabel('$Shear\ angle\ \phi$', fontsize = 18)
        pylab.legend(handles=[p_C1,p_C2], labels=['$\delta$','$\phi$'], fontsize = 16,  fancybox = True, shadow = True, frameon = True, loc = 'lower right')
    if (par=='delta'):
        locY = np.amax(tab[:,4])
        pylab.axvline(x = delta_opt, color='k')
        ax1.annotate(' $\delta='+str(round(delta_opt,3))+'$', xy=(delta_opt, locY),fontsize = 18)
        ax1.set_xlim([getParameter(par).min,getParameter(par).max])
        p_C1, = ax1.plot(tab[:,0],tab[:,4],'r',linewidth = 2)
        p_C2, = ax2.plot(tab[:,0],tab[:,6],'b',linewidth = 2)
        ax1.set_ylabel('$Strain\ rate\ constant\ C_0$', fontsize = 18)
        ax2.set_ylabel('$Shear\ angle\ \phi$', fontsize = 18)
        pylab.legend(handles=[p_C1,p_C2], labels=['$C_0$','$\phi$'], fontsize = 16,  fancybox = True, shadow = True, frameon = True, loc = 'lower right')
    if (par=='phi'):
        locY = np.amax(tab[:,4])
        pylab.axvline(x = 180/math.pi*phi_opt, color='k')
        ax1.annotate(' $\phi='+str(round(180/math.pi*phi_opt,3))+'$', xy=(180/math.pi*phi_opt, locY),fontsize = 18)
        ax1.set_xlim([180/math.pi*getParameter(par).min,180/math.pi*getParameter(par).max])
        p_C1, = ax1.plot(tab[:,0],tab[:,4],'r',linewidth = 2)
        p_C2, = ax2.plot(tab[:,0],tab[:,5],'b',linewidth = 2)
        ax1.set_ylabel('$Strain\ rate\ constant\ C_0$', fontsize = 18)
        ax2.set_ylabel('$Plastic\ zone\ thickness\ \delta$', fontsize = 18)
        pylab.legend(handles=[p_C1,p_C2], labels=['$C_0$','$\delta$'], fontsize = 16,  fancybox = True, shadow = True, frameon = True, loc = 'lower right')
    ax1.set_xlabel(var, fontsize = 18)
    pylab.savefig(par+'Internal.svg', transparent=True, bbox_inches='tight', pad_inches=0)
    pylab.show()   
    getParameter(par).vary = True    

def plotCurves(y, lab, ylab, tit, nam):
    pylab.figure(figsize = (11.69,8.27)) # for a4 landscape 
    pylab.rc('text', usetex=True)
    pylab.rcParams['xtick.labelsize'] = 16
    pylab.rcParams['ytick.labelsize'] = 16
    ii = 0
    for advance in advances:
        pylab.plot (xs[ii], y[ii], label='$'+lab+'\ (t_1='+str(round(advances[ii]*1e3,2))+'\ mm)$', linewidth=2)
        ii += 1
    pylab.xlabel('$Cutting\ speed\ V\ (m/min)$', fontsize = 16)
    pylab.ylabel(ylab, fontsize = 16)
    pylab.legend(loc = 'upper right', bbox_to_anchor = (1.0, 1.0), fontsize = 12, frameon = True,
                         fancybox = True, shadow = True, ncol = 1, numpoints = 1)
    pylab.grid(True)
    pylab.title(tit, y = 1.04, fontsize = 20)
    pylab.savefig(nam, transparent=True, bbox_inches='tight', pad_inches=0)
    pylab.show()   

def plotVariations(speeds,advances):
    global V, t1, xs
    xs = []
    ys_Fc = []
    ys_Ft = [] 
    ys_TAB  = [] 
    ys_Tint  = [] 
    ys_phi  = [] 
    ys_t2  = [] 
    ys_lc  = [] 
    for advance in advances:
        x = []
        y_Fc = []
        y_Ft = [] 
        y_TAB  = [] 
        y_Tint  = [] 
        y_phi  = [] 
        y_t2  = [] 
        y_lc  = [] 
        for speed in speeds:
            V = speed / 60
            t1 = advance
            # Initialisation de l'optimisation
            reinitParams(paramsOpt1,paramsOpt2)
            myfit = Optimize(fittingFunction)
            if (TAB > 0) :
                x.append(speed)
                y_Fc.append(Fc)
                y_Ft.append(Ft)
                y_TAB.append(TAB-T0)
                y_Tint.append(Tint-T0)
                y_phi.append(phi*180/math.pi)
                y_t2.append(1000*t2)
                y_lc.append(1000*lc)
        xs.append(x)
        ys_Fc.append(y_Fc)
        ys_Ft.append(y_Ft)
        ys_TAB.append(y_TAB)
        ys_Tint.append(y_Tint)
        ys_phi.append(y_phi)
        ys_t2.append(y_t2)
        ys_lc.append(y_lc)
    plotCurves(ys_Fc, 'F_C', '$Force\ F_C\ (N)$', 'Cutting force $F_C$ v.s. cutting speed $V$', 'Fc_vs_Speed.svg')
    plotCurves(ys_Ft, 'F_T', '$Force\ F_T\ (N)$', 'Advancing\ force $F_T$ v.s. cutting speed $V$', 'Fa_vs_Speed.svg')
    plotCurves(ys_TAB, 'T_{AB}', '$Temperature\ T_{AB}\ (^{\circ} C)$', 'Temperature $T_{AB}$ v.s. cutting speed $V$', 'TAB_vs_Speed.svg')
    plotCurves(ys_Tint, 'T_{int}', '$Temperature\ T_{int}\ (^{\circ}C)$', 'Temperature $T_{int}$ v.s. cutting speed $V$', 'Tint_vs_Speed.svg')
    plotCurves(ys_t2, 't_2', '$Thickness\ t_2\ (mm)$', 'Chip thickness $t_2$ v.s. cutting speed $V$', 't2_vs_Speed.svg')
    plotCurves(ys_lc, 'l_c', '$Length\ l_c\ (mm)$', 'Contact length $l_c$ v.s. cutting speed $V$', 'lc_vs_Speed.svg')
    plotCurves(ys_phi, '\phi', '$Angle\ \phi\ (^{\circ})$', 'Shear plane angle $\phi$ v.s. cutting speed $V$', 'phi_vs_Speed.svg')


# List of parameters
#global paramsOpt1, paramsOpt2
paramsOpt1 = lmfit.Parameters()
paramsOpt2 = lmfit.Parameters()
paramsOpt1.add('delta', value = 0.015, min = 0.005, max = 0.2)
paramsOpt2.add('C0', value = 6, min = 2, max = 10)
paramsOpt2.add('phi', value = 26.5*math.pi/180, min = 8*math.pi/180, max = 45*math.pi/180)

# Initialisation de l'optimisation
reinitParams(paramsOpt1,paramsOpt2)

# Débit massique de copeau
mchip = rho*V*t1*w

# Calcul du système d'équations
myfit = Optimize(fittingFunction)

# Recalcul de l'optimal au cas où mais normalement, ça ne sert à rien
#Toint, kchip = compute_Toint_Kchip()
#sigmaN, sigmaNmax = compute_sigmaN_sigmaNp()

print("C0 : ",round(C0,3))
print("delta : ",round(delta,3))

print("Norme erreur interne : ",round(math.sqrt((Toint-kchip)**2+(sigmaNmax-sigmaN)**2),10),"Pa")
print("Loops : ",nLoops_1, nLoops_2)

print("Angle de cisaillement : ",round(phi*180/math.pi,1)," deg")

print("Effort de coupe Fc : ",round(Fc,1),"N (",round(Fc/w/1000,1),"N/mm)")
print("Effort d'avance Fa : ",round(Ft,1),"N (",round(Ft/w/1000,1),"N/mm)")

print("Epaisseur du copeau : ",round(t2*1000,2)," mm")
print("Longueur de contact outil/copeau : ",round(lc*1000,2)," mm")

print("Deformation zone 1 : ",round(EpsAB,2))
print("Taux de def zone 1 : ",round(EpspAB,1)," ms-1")
print("Temperature zone 1 : ",round(TAB-T0,1)," C")
print("Contrainte zone 1 : ",round(kAB/1e6,1)," MPa")

print("Deformation zone 2 : ",round(Epsint,2))
print("Taux de def zone 2 : ",round(Epspint,1)," ms-1")
print("Temperature zone 2 : ",round(Tint-T0,1)," C")
print("Contrainte zone 2 : ",round(sigmaNmax/1e6,1)," MPa")
	
C0_opt = C0 #=paramsOpt2['C0'].value
phi_opt = phi #=paramsOpt2['phi'].value
delta_opt = delta #=paramsOpt1['delta'].value


# SECTION 1
# This zone serves to generate the graphs of the evolution of 1 internal parameter vs. another one
"""
parameterStudy('delta','$Plastic\ zone\ thickness\ \delta$')
parameterStudy('C0','$Strain\ rate\ constant\ C_0$')
parameterStudy('phi','$Shear\ angle\ \phi$')
"""

# SECTION 2
# Sensivity study
"""
speeds = np.linspace(100, 400, 25, True)
advances =  np.array([0.15e-3, 0.2e-3, 0.25e-3, 0.3e-3, 0.4e-3, 0.5e-3])
plotVariations(speeds,advances)
"""

# SECTION 3
# Tests the unicity of the solution by running 15625 runs
# Warning, this can be quite long to execute
"""
deltaVals = np.linspace(paramsOpt1['delta'].min,paramsOpt1['delta'].max,26,False)
deltaVals = np.delete(deltaVals,0,0)
phiVals = np.linspace(paramsOpt2['phi'].min,paramsOpt2['phi'].max,26,False)
phiVals = np.delete(phiVals,0,0)
C0Vals = np.linspace(paramsOpt2['C0'].min,paramsOpt2['C0'].max,26,False)
C0Vals = np.delete(C0Vals,0,0)
ii = 0
totLoops=deltaVals.size*phiVals.size*C0Vals.size
deltaFVals = np.zeros(totLoops)
phiFVals = np.zeros(totLoops)
C0FVals = np.zeros(totLoops)
nLoops1Vals = np.zeros(totLoops)
nLoops2Vals = np.zeros(totLoops)
FcVals = np.zeros(totLoops)
FtVals = np.zeros(totLoops)
TintVals = np.zeros(totLoops)
TABVals = np.zeros(totLoops)
t2Vals = np.zeros(totLoops)
lcVals = np.zeros(totLoops)

for deltaI in deltaVals:
    for phiI in phiVals:
        print (100*ii/totLoops, "% done")
        for C0I in C0Vals:
            paramsOpt2['C0'].value=C0I
            paramsOpt2['phi'].value=phiI
            paramsOpt1['delta'].value=deltaI
            myfit = Optimize(fittingFunction)
            deltaFVals[ii] = delta
            phiFVals[ii] = phi*180/math.pi
            C0FVals[ii] = C0
            FcVals[ii] = Fc
            FtVals[ii] = Ft
            TintVals[ii] = Tint-T0
            TABVals[ii] = TAB-T0
            t2Vals[ii] = t2*1000
            lcVals[ii] = lc*1000
            nLoops1Vals[ii] = nLoops_1
            nLoops2Vals[ii] = nLoops_2
            ii=ii+1
nLoops1Vals=nLoops1Vals.astype(int)
nLoops2Vals=nLoops2Vals.astype(int)
np.save('nLoops1Vals.npy',nLoops1Vals)
np.save('nLoops2Vals.npy',nLoops2Vals)
np.save('deltaFVals.npy',deltaFVals)
np.save('FcVals.npy',FcVals)
np.save('FtVals.npy',FtVals)
np.save('phiFVals.npy',phiFVals)
np.save('TintVals.npy',TintVals)
np.save('TABVals.npy',TABVals)
np.save('t2Vals.npy',t2Vals)
np.save('lcVals.npy',lcVals)
np.save('C0FVals.npy',C0FVals)

def plotHisto(X, Y, mean, title, xlab, ylab, figname):
    dx = 0.8*(X[1]-X[0])
    pylab.figure(figsize = (11.69,8.27)) # for a4 landscape
    pylab.rc('text', usetex=True)
    pylab.rcParams['xtick.labelsize'] = 16
    pylab.rcParams['ytick.labelsize'] = 16
    pylab.bar(X, Y, dx, color='#0000cc')
    pylab.ylabel(ylab, fontsize = 16)
    pylab.xlabel(xlab, fontsize = 16)
    pylab.title(title, fontsize=20)
    pylab.axvline(x = mean, color='#cc0000')
    pylab.annotate(' $mean='+str(round(mean,1))+'$', xy=(mean, 0.95*(Y.max()-Y.min())+Y.min()),fontsize = 16)
    pylab.grid(True)
    pylab.savefig(figname)
    pylab.show()

def histogramCreate(data, width):
    Y=np.bincount((data/width).astype(int))
    xm=int(data.min()/width)
    xM=int(data.max()/width)+1
    X=np.arange(width*xm, width*xM, width)
    return X, Y[xm:]
    
nLoops1Vals=np.load('nLoops1Vals.npy')
nLoops2Vals=np.load('nLoops2Vals.npy')
deltaFVals=np.load('deltaFVals.npy')
FcVals=np.load('FcVals.npy')
FtVals=np.load('FtVals.npy')
phiFVals=np.load('phiFVals.npy')
TintVals=np.load('TintVals.npy')
TABVals=np.load('TABVals.npy')
t2Vals=np.load('t2Vals.npy')
lcVals=np.load('lcVals.npy')
C0FVals=np.load('C0FVals.npy')

print('nLoops1Vals ',nLoops1Vals.min(),' ',nLoops1Vals.max(),' ',nLoops1Vals.mean(),' ',(nLoops1Vals.max()-nLoops1Vals.min())/nLoops1Vals.mean())
print('nLoops2Vals ',nLoops2Vals.min(),' ',nLoops2Vals.max(),' ',nLoops2Vals.mean(),' ',(nLoops2Vals.max()-nLoops2Vals.min())/nLoops2Vals.mean())
print('deltaFVals ',deltaFVals.min(),' ',deltaFVals.max(),' ',deltaFVals.mean(),' ',(deltaFVals.max()-deltaFVals.min())/deltaFVals.mean())
print('FcVals ',FcVals.min(),' ',FcVals.max(),' ',FcVals.mean(),' ',(FcVals.max()-FcVals.min())/FcVals.mean())
print('FtVals ',FtVals.min(),' ',FtVals.max(),' ',FtVals.mean(),' ',(FtVals.max()-FtVals.min())/FtVals.mean())
print('phiFVals ',phiFVals.min(),' ',phiFVals.max(),' ',phiFVals.mean(),' ',(phiFVals.max()-phiFVals.min())/phiFVals.mean())
print('TintVals ',TintVals.min(),' ',TintVals.max(),' ',TintVals.mean(),' ',(TintVals.max()-TintVals.min())/TintVals.mean())
print('TABVals ',TABVals.min(),' ',TABVals.max(),' ',TABVals.mean(),' ',(TABVals.max()-TABVals.min())/TABVals.mean())
print('t2Vals ',t2Vals.min(),' ',t2Vals.max(),' ',t2Vals.mean(),' ',(t2Vals.max()-t2Vals.min())/t2Vals.mean())
print('lcVals ',lcVals.min(),' ',lcVals.max(),' ',lcVals.mean(),' ',(lcVals.max()-lcVals.min())/lcVals.mean())
print('C0FVals ',C0FVals.min(),' ',C0FVals.max(),' ',C0FVals.mean(),' ',(C0FVals.max()-C0FVals.min())/C0FVals.mean())

X, Y = histogramCreate(nLoops1Vals,10)
plotHisto(X, Y, nLoops1Vals.mean(), '$Histogram\ of\ the\ number of\ loops\ to\ get\ a\ solution$', '$Number\ of\ loops\ needed\ to\ converge$', '$Frequency$', 'Loops.svg')
"""

