#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
These are the source files of the implementation of the Oxley's machining model 
using the LMFIT library in Python. 

This work is related to the PhD thesis of Maxime Dawoua Kaoutoing:
Maxime. D. Kaoutoing, Contributions à la modélisation et la simulation de la coupe des métaux: 
    vers un outil d'aide à la surveillance par apprentissage, PhD Thesis, Toulouse University, 2020

@author: Olivier Pantalé
"""

# Initialisation
import math
import numpy as np
import pylab
import matplotlib.pyplot as plt
plt.style.use('mime')

# Import the LMFIT module
import lmfit

# --------------------------------------------------------------------------------------------------
# Definition of the cutting parameters
# --------------------------------------------------------------------------------------------------
V = 200 / 60                # Cutting speed (m/s)
alpha = -7 * (math.pi/180)  # Cutting angle (radians)
w = 1.6e-3                  # Width of cut (mm)
t1 = 0.15e-3                # Depth of cut (mm)

# --------------------------------------------------------------------------------------------------
# Material definition
# --------------------------------------------------------------------------------------------------
T0 = 273.15                 # Initial temperature (Kelvin)
Tw = T0 + 25                # Ambient temperature (Kelvin)
rho = 8000.0                # Density (km/m^3)
eta = 0.9                   # Value of the eta parameter
psi = 0.9                   # Value of the psi parameter
A = 553.1e6                 # A coefficient for the Johnson-Cook law (Pa)
B = 600.8e6                 # B coefficient for the Johnson-Cook law (Pa)
n = 0.234                   # n coefficient for the Johnson-Cook law
C = 0.0134                  # C coefficient for the Johnson-Cook law
m = 1                       # m coefficient for the Johnson-Cook law
Tm = 1460 + T0              # Melting temperature (Kelvin)
Epsp0 = 1                   # epislon dot coefficient for the Johnson-Cook law

# --------------------------------------------------------------------------------------------------
# Start of the main program
# Noting to change here after berfore sections 1 to 3 at the end of the program
# --------------------------------------------------------------------------------------------------
nLoops_1 = 0
nLoops_2 = 0
TAB_precision = 1e-3        # Precision on the evaluation of TAB
Tc_precision = 1e-3         # Precision on the evaluation of Tc
sqrt3 = math.sqrt(3)        # Constant value for square root of 3

# Computes the mchip value from cutting parameters
mchip = rho * V * t1 * w

# Computes the Johnson-Cook equivalent stress
def JohnsonCook(eps, epsp, T):
    return (A + B * eps**n) * (1.0 + C * math.log(epsp/Epsp0)) * (1.0 - ((T - T0) / (Tm - T0))**m)

# Computes the K law    
def KLaw(T):
    return 52.61 - 0.0281 * (T - T0)

# Computes the CP law    
def CpLaw(T):
    return 420 + 0.504 * (T - T0)

# Computes the TAB temperature, returns 0 if TAB  >  Tm
def ComputeAB(): 
    # Sets TAB equal to Tw
    TAB = Tw
    # Initialisation of the max number of loops
    maxLoops = 1000
    # Evaluates the temperature due to plastic deformation 
    while (maxLoops > 0) :
        # Computes Cp and K for TAB 
        Cp = CpLaw(TAB)
        K = KLaw(TAB)
        # Computes the flow stress of the material 
        kAB = (1.0/sqrt3) * JohnsonCook(EpsAB, EpspAB, TAB)
        # Computes the Fs value
        Fs = kAB * lAB * w
        # Computes coefficient RT 
        RTtanPhi = math.tan(phi) * (rho * Cp * V * t1) / K
        # Computes beta 
        if (RTtanPhi > 10) :
            beta = 0.3 - 0.15 * math.log10(RTtanPhi)
        else :
            beta = 0.5 - 0.35 * math.log10(RTtanPhi)
        # Computes the delta T
        deltaTsz = ((1-beta) * Fs * Vs) / (mchip * Cp)
        # Computes the new TAB temperature
        NewTAB = Tw + eta * deltaTsz
        # Tests if TAB > Tm and return zero
        if (NewTAB > Tm) : return 0, 0, 0, 0
        # Tests for the convergence of TAB (criterion is TAB_precision)
        if (abs(NewTAB - TAB) <= TAB_precision) :
            return NewTAB, Fs, deltaTsz, kAB
        # Affects the new TAB 
        TAB = NewTAB
        maxLoops -= 1
    # Oups ! no convergence at least after so many loops 
    return 0, 0, 0, 0
   
# Computes the Tc temperature
def ComputeTc(deltaTsz): 
    Tc = Tw + deltaTsz
    while (True) :
        # Computes Cp and K for temperature Tc 
        Cp = CpLaw(Tc)
        # K = KLaw(Tc) Not used here
        # Increment of temperature at the interface 
        deltaTc = Ff * Vc / (mchip * Cp)
        # New interfacial temperature TC 
        NewTc = Tw + deltaTsz + deltaTc
        # Tests for the convergence of Tc (criterion is Tc_precision)
        if (abs(NewTc - Tc) <= Tc_precision) :
            return Tc, deltaTc
        # Affects the new Tc 
        Tc = NewTc
    
# Computes the Toint and Kchip values
def Compute_Toint_Kchip():
    global lc, neq, EpsAB, EpspAB, TAB, lAB, Vs
    global Ff, Fn, Fc, Ft, Vc, t2, Epsint
    global Epspint, Tint, R, kAB, Lambda, theta
    # Length of the first shear band lAB
    lAB = t1 / math.sin(phi)   
    # Speed along the first shear band
    Vs = V * math.cos(alpha) / math.cos(phi - alpha)   
    # Chip thickness
    t2 = t1 * math.cos(phi - alpha) / math.sin(phi)
    # Chip speed
    Vc = V * math.sin(phi) / math.cos(phi - alpha)
    # Plastic deformation in the AB zone
    gammaAB = 1/2 * math.cos(alpha) / (math.sin(phi) * math.cos(phi - alpha))
    EpsAB = gammaAB / sqrt3
    # Deformation rate in the AB zone
    gammapAB = C0 * Vs / lAB # not here
    EpspAB = gammapAB / sqrt3
    # Computes the TAB temperature
    TAB, Fs, deltaTsz, kAB = ComputeAB()
    # If TAB > Tw returns an error
    if (TAB == 0): return 0, 1e10, 0, 0
    # Computes neq using Lalwani expression
    neq = (n * B * EpsAB**n) / (A + B * EpsAB**n)
    # Computes the theta angle
    theta = math.atan(1 + math.pi/2 - 2 * phi - C0 * neq)
    # Computes the resultant force R depending on Fs and theta
    R = Fs / math.cos(theta)
    # Computes the lambda parameter
    Lambda = theta + alpha - phi
    # Computes internal forces
    Ff = R * math.sin(Lambda)
    Fn = R * math.cos(Lambda)
    Fc = R * math.cos(theta - phi)
    Ft = R * math.sin(theta - phi)
    # Computes SigmaNp 
    sigmaNmax = kAB * (1 + math.pi / 2 - 2 * alpha - 2 * C0 * neq)
    # Tool/Chip contact length
    lc = t1 * math.sin(theta) / (math.cos(Lambda) * math.sin(phi)) * (1 + C0 * neq / (3 * (1 + 2 * (math.pi/4 - phi) - C0 * neq)))
    # Stress along the interface
    Toint = Ff / (lc * w)
    # Equivalent deformation along the interface
    gammaM = lc / (delta * t2)
    gammaInt = 2 * gammaAB + gammaM / 2
    Epsint = gammaInt / sqrt3
    # Rate of deformation along the interface
    gammapInt = Vc / (delta * t2)
    Epspint = gammapInt / sqrt3
    # Contact temperature along the interface
    Tc, deltaTc = ComputeTc(deltaTsz)
    # K and Cp function of the Tc temperature
    K = KLaw(Tc)
    Cp = CpLaw(Tc)
    # Computes the RT factor
    RT = (rho * Cp * V * t1) / K
    # Computes the delta Tm value
    deltaTm = deltaTc * 10**(0.06 - 0.195 * delta * math.sqrt(RT * t2 / lc)) * math.sqrt(RT * t2 / lc)
    # Mean temperature along the interface
    Tint = Tw + deltaTsz + psi * deltaTm
    # Stress flow within the chip
    kchip = (1 / sqrt3) * JohnsonCook(Epsint, Epspint, Tint)
    # Computes the normal stress
    sigmaN = Fn / (lc * w)
    return Toint, kchip, sigmaN, sigmaNmax
    
# Initialization of the internal parameters for the optimization procedure
def ReinitializeParameters(paramsOpt1, paramsOpt2):
    paramsOpt2['C0'].value = (paramsOpt2['C0'].max + paramsOpt2['C0'].min) / 2
    paramsOpt2['phi'].value = (paramsOpt2['phi'].max + paramsOpt2['phi'].min) / 2
    paramsOpt1['delta'].value = (paramsOpt1['delta'].max + paramsOpt1['delta'].min) / 2

# Internal fitting function on C0 and phi
def InternalFittingFunction(paramsOpt2):
    global C0, phi
    global Toint, kchip, sigmaN, sigmaNmax, nLoops_1
    C0 = paramsOpt2['C0'].value
    phi = paramsOpt2['phi'].value
    # Computes the internal parameters
    Toint, kchip, sigmaN, sigmaNmax = Compute_Toint_Kchip()
    # Test if there was a bug in the last run
    #if (Toint == 0): print(V * 60, t1, 180 / math.pi * phi, C0, delta, ' FAILED\n')
    # Computes the gap for the optimizer
    Gap = [(Toint - kchip), (sigmaN - sigmaNmax)]
    # Increases the number of loops
    nLoops_1 += 1
    # Return the gap value
    return Gap

# External fitting function on delta
def FittingFunction(paramsOpt1):
    global delta
    global paramsOpt2, nLoops_2
    delta = paramsOpt1['delta'].value
    fitOpt1 = lmfit.minimize(InternalFittingFunction, paramsOpt2)
    paramsOpt2 = fitOpt1.params
    # Increases the number of loops
    nLoops_2 += 1
    # Returns the cutting Force
    return [Fc]

# Optimization procedure
def Optimize(FittingFunction):
    global paramsOpt1, nLoops_1, nLoops_2
    # initialization for the number of loops
    nLoops_1 = 0
    nLoops_2 = 0
    # Calls the optimizer
    myfit = lmfit.minimize(FittingFunction, paramsOpt1)
    # Get the results
    paramsOpt1 = myfit.params

def RUN():
    # Initialisation of the Optimizer
    ReinitializeParameters(paramsOpt1, paramsOpt2)
    # Computes the solution
    Optimize(FittingFunction)

# Initialize the lists of parameters
paramsOpt1 = lmfit.Parameters()
paramsOpt2 = lmfit.Parameters()

# Initial values
paramsOpt1.add('delta', value = 0.015, min = 0.005, max = 0.2)
paramsOpt2.add('C0', value = 6, min = 2, max = 10)
paramsOpt2.add('phi', value = 26.5 * math.pi / 180, min = 8 * math.pi / 180, max = 45 * math.pi / 180)

# Initialisation of the Optimizer
ReinitializeParameters(paramsOpt1, paramsOpt2)

# Computes the solution
Optimize(FittingFunction)

# Recalcul de l'optimal au cas où mais normalement, ça ne sert à rien
#Toint, kchip = Compute_Toint_Kchip()
#sigmaN, sigmaNmax = compute_sigmaN_sigmaNp()

print("C0 : ", round(C0, 3))
print("delta : ", round(delta, 3))

print("Internal error : ", round(math.sqrt((Toint-kchip)**2 + (sigmaNmax-sigmaN)**2), 10), "Pa")
print("Number of loops : ", nLoops_1, "and", nLoops_2)

print("Shear angle : ", round(phi * 180/math.pi, 2), "deg")

print("Cutting force Fc   : ", round(Fc, 1), "N (", round(Fc / w / 1000, 1), "N/mm)")
print("Advancing force Fa : ", round(Ft, 1), "N (", round(Ft / w / 1000, 1), "N/mm)")

print("Chip thickness : ", round(t2 * 1000, 2), "mm")
print("Tool/Chip contact length : ", round(lc * 1000, 2), "mm")

print("Strain in zone 1 : ", round(EpsAB, 2))
print("Strain rate in zone 1 : ", round(EpspAB, 1), "ms-1")
print("Temperature in zone 1 : ", round(TAB - T0, 1), "°C")
print("Stress in zone 1 : ", round(kAB / 1e6, 1), "MPa")

print("Strain in zone 2 : ", round(Epsint, 2))
print("Strain rate in zone 2 : ", round(Epspint, 1), "ms-1")
print("Temperature in zone 2 : ", round(Tint - T0, 1), "°C")
print("Stress in zone 2 : ", round(sigmaNmax / 1e6, 1), "MPa")
	
C0_opt = C0 # = paramsOpt2['C0'].value
phi_opt = phi # = paramsOpt2['phi'].value
delta_opt = delta # = paramsOpt1['delta'].value

# --------------------------------------------------------------------------------------------------
# SECTION 1
# --------------------------------------------------------------------------------------------------
# This zone serves to generate the graphs of the evolution of 1 internal parameter vs. another one

# def GetParameter(par):
#     if (par in paramsOpt1.valuesdict()) :
#         return paramsOpt1[par]
#     if (par in paramsOpt2.valuesdict()):
#         return paramsOpt2[par]
#     print("Parameter", par, "is not in any of the parameters defined\n")
#     return

# def ParameterStudy(par, var):
#     global tab, phiVary, deltaVary, C0Vary
#     GetParameter(par).vary = False    
#     factor = 1
#     if (par == 'phi') : factor = 180/math.pi
#     tab = np.array([[0, 0, 0, 0, 0, 0, 0]])
#     xs = np.linspace(GetParameter(par).min, GetParameter(par).max, 100, True)
#     ReinitializeParameters(paramsOpt1, paramsOpt2)
#     for x in xs:
#         GetParameter(par).value = x
#         # find the solution
#         Optimize(FittingFunction)
#         aa = np.array([factor*x, Fc, Ft, math.sqrt((Toint-kchip)**2+(sigmaN-sigmaNmax)**2), C0, delta, 180/math.pi*phi])
#         # Corretion for delta
#         if (par == 'delta') :
#             if (math.sqrt((Toint-kchip)**2+(sigmaN-sigmaNmax)**2) > 1) : aa[3] = 0
#         tab = np.vstack((tab, aa))
#     tab = np.delete(tab, 0, 0)
#     # Plot the first Figure
#     pylab.rc('text', usetex = True)
#     pylab.rcParams['xtick.labelsize'] = 18
#     pylab.rcParams['ytick.labelsize'] = 18
#     fig, ax1 = pylab.subplots(figsize = (11.69, 8.27))
#     ax2 = ax1.twinx()
#     ax2.ticklabel_format(style = 'sci', scilimits = (-2, 2))
#     ax2.grid(False)
#     p_Fc, = ax1.plot(tab[:, 0], tab[:, 1], 'r', linewidth = 3)
#     p_Fn, = ax1.plot(tab[:, 0], tab[:, 2], 'b', linewidth = 3)
#     p_C, = ax2.plot(tab[:, 0], tab[:, 3], 'g', linewidth = 3)
#     locY = max(np.amax(tab[:, 1]), np.amax(tab[:, 2]))
#     if (par == 'C0'):
#         pylab.axvline(x = C0_opt, color = 'k')
#         ax1.annotate(' $C_0 = '+str(round(C0_opt, 3))+'$', xy = (C0_opt, locY), fontsize = 22)
#         ax1.set_xlim([GetParameter(par).min, GetParameter(par).max])
#     if (par == 'delta'):
#         pylab.axvline(x = delta_opt, color = 'k')
#         ax1.annotate(' $\delta = '+str(round(delta_opt, 3))+'$', xy = (delta_opt, locY), fontsize = 22)
#         ax1.set_xlim([GetParameter(par).min, GetParameter(par).max])
#     if (par == 'phi'):
#         pylab.axvline(x = 180/math.pi*phi_opt, color = 'k')
#         ax1.annotate(' $\phi = '+str(round(180/math.pi*phi_opt, 3))+'$', xy = (180/math.pi*phi_opt, locY), fontsize = 22)
#         ax1.set_xlim([180/math.pi*GetParameter(par).min, 180/math.pi*GetParameter(par).max])
#     ax1.set_xlabel(var, fontsize = 22)
#     ax1.set_ylabel('$Cutting\ forces\ F_{C}, \ F_{T}\ (N)$', fontsize = 22)
#     ax2.set_ylabel('$Error\ \Delta F\ (N)$', fontsize = 22)
#     ax2.set_ylim([0,tab[:, 3].max()])
#     pylab.legend(handles = [p_Fc, p_Fn, p_C], labels = ['$F_C$', '$F_T$', '$\Delta F$'], fontsize = 20, fancybox = True, shadow = True, frameon = True, loc = 'lower right')
#     pylab.savefig(par+'.svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
#     pylab.show()
#     # Plot the second figure
#     fig, ax1 = pylab.subplots(figsize = (11.69, 8.27))
#     ax2 = ax1.twinx()
#     ax2.grid(False)
#     if (par == 'C0'):
#         locY = np.amax(tab[:, 5])
#         pylab.axvline(x = C0_opt, color = 'k')
#         ax1.annotate(' $C_0 = '+str(round(C0_opt, 3))+'$', xy = (C0_opt, locY), fontsize = 22)
#         ax1.set_xlim([GetParameter(par).min, GetParameter(par).max])
#         p_C1, = ax1.plot(tab[:, 0], tab[:, 5], 'r', linewidth = 3)
#         p_C2, = ax2.plot(tab[:, 0], tab[:, 6], 'b', linewidth = 3)
#         ax1.set_ylabel('Plastic zone thickness $\delta$', fontsize = 22)
#         ax2.set_ylabel('Shear angle $\phi$', fontsize = 22)
#         pylab.legend(handles = [p_C1, p_C2], labels = ['$\delta$', '$\phi$'], fontsize = 20, fancybox = True, shadow = True, frameon = True, loc = 'lower right')
#     if (par == 'delta'):
#         locY = np.amax(tab[:, 4])
#         pylab.axvline(x = delta_opt, color = 'k')
#         ax1.annotate(' $\delta = '+str(round(delta_opt, 3))+'$', xy = (delta_opt, locY), fontsize = 22)
#         ax1.set_xlim([GetParameter(par).min, GetParameter(par).max])
#         p_C1, = ax1.plot(tab[:, 0], tab[:, 4], 'r', linewidth = 3)
#         p_C2, = ax2.plot(tab[:, 0], tab[:, 6], 'b', linewidth = 3)
#         ax1.set_ylabel('Strain rate constant $C_0$', fontsize = 22)
#         ax2.set_ylabel('Shear angle $\phi$', fontsize = 22)
#         pylab.legend(handles = [p_C1, p_C2], labels = ['$C_0$', '$\phi$'], fontsize = 20, fancybox = True, shadow = True, frameon = True, loc = 'lower right')
#     if (par == 'phi'):
#         locY = np.amax(tab[:, 4])
#         pylab.axvline(x = 180/math.pi*phi_opt, color = 'k')
#         ax1.annotate(' $\phi = '+str(round(180/math.pi*phi_opt, 3))+'$', xy = (180/math.pi*phi_opt, locY), fontsize = 22)
#         ax1.set_xlim([180/math.pi*GetParameter(par).min, 180/math.pi*GetParameter(par).max])
#         p_C1, = ax1.plot(tab[:, 0], tab[:, 4], 'r', linewidth = 3)
#         p_C2, = ax2.plot(tab[:, 0], tab[:, 5], 'b', linewidth = 3)
#         ax1.set_ylabel('Strain rate constant $C_0$', fontsize = 22)
#         ax2.set_ylabel('Plastic zone thickness $\delta$', fontsize = 22)
#         pylab.legend(handles = [p_C1, p_C2], labels = ['$C_0$', '$\delta$'], fontsize = 20, fancybox = True, shadow = True, frameon = True, loc = 'lower right')
#     ax1.set_xlabel(var, fontsize = 22)
#     pylab.savefig(par+'Internal.svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
#     pylab.show()   
#     GetParameter(par).vary = True
    
# ParameterStudy('delta', 'Plastic zone thickness $\delta$')
# ParameterStudy('C0', 'Strain rate constant $C_0$')
# ParameterStudy('phi', 'Shear angle $\phi$')


# --------------------------------------------------------------------------------------------------
# SECTION 2
# --------------------------------------------------------------------------------------------------
# Tests the unicity of the solution by running 15625 runs
# Warning, this can be quite long to execute

# deltaVals = np.linspace(paramsOpt1['delta'].min, paramsOpt1['delta'].max, 26, False)
# deltaVals = np.delete(deltaVals, 0, 0)
# phiVals = np.linspace(paramsOpt2['phi'].min, paramsOpt2['phi'].max, 26, False)
# phiVals = np.delete(phiVals, 0, 0)
# C0Vals = np.linspace(paramsOpt2['C0'].min, paramsOpt2['C0'].max, 26, False)
# C0Vals = np.delete(C0Vals, 0, 0)
# ii = 0
# totLoops = deltaVals.size*phiVals.size*C0Vals.size
# deltaFVals = np.zeros(totLoops)
# phiFVals = np.zeros(totLoops)
# C0FVals = np.zeros(totLoops)
# nLoops1Vals = np.zeros(totLoops)
# nLoops2Vals = np.zeros(totLoops)
# FcVals = np.zeros(totLoops)
# FtVals = np.zeros(totLoops)
# TintVals = np.zeros(totLoops)
# TABVals = np.zeros(totLoops)
# t2Vals = np.zeros(totLoops)
# lcVals = np.zeros(totLoops)

# for deltaI in deltaVals:
#     for phiI in phiVals:
#         print (100*ii/totLoops, "% done")
#         for C0I in C0Vals:
#             paramsOpt2['C0'].value = C0I
#             paramsOpt2['phi'].value = phiI
#             paramsOpt1['delta'].value = deltaI
#             Optimize(FittingFunction)
#             deltaFVals[ii] = delta
#             phiFVals[ii] = phi*180/math.pi
#             C0FVals[ii] = C0
#             FcVals[ii] = Fc
#             FtVals[ii] = Ft
#             TintVals[ii] = Tint-T0
#             TABVals[ii] = TAB-T0
#             t2Vals[ii] = t2*1000
#             lcVals[ii] = lc*1000
#             nLoops1Vals[ii] = nLoops_1
#             nLoops2Vals[ii] = nLoops_2
#             ii = ii+1
# nLoops1Vals = nLoops1Vals.astype(int)
# nLoops2Vals = nLoops2Vals.astype(int)
# np.save('nLoops1Vals.npy', nLoops1Vals)
# np.save('nLoops2Vals.npy', nLoops2Vals)
# np.save('deltaFVals.npy', deltaFVals)
# np.save('FcVals.npy', FcVals)
# np.save('FtVals.npy', FtVals)
# np.save('phiFVals.npy', phiFVals)
# np.save('TintVals.npy', TintVals)
# np.save('TABVals.npy', TABVals)
# np.save('t2Vals.npy', t2Vals)
# np.save('lcVals.npy', lcVals)
# np.save('C0FVals.npy', C0FVals)

# def plotHisto(X, Y, mean, title, xlab, ylab, figname):
#     dx = 0.8*(X[1]-X[0])
#     pylab.figure(figsize = (11.69, 8.27)) # for a4 landscape
#     pylab.rc('text', usetex = True)
#     pylab.rcParams['xtick.labelsize'] = 18
#     pylab.rcParams['ytick.labelsize'] = 18
#     pylab.bar(X, Y, dx, color = '#0000cc')
#     pylab.ylabel(ylab, fontsize = 22)
#     pylab.xlabel(xlab, fontsize = 22)
#     pylab.title(title, fontsize = 22)
#     pylab.axvline(x = mean, color = '#cc0000')
#     pylab.annotate('mean = '+str(round(mean, 1)), xy = (mean+10, 0.95*(Y.max()-Y.min())+Y.min()), fontsize = 20)
#     #pylab.grid(True)
#     pylab.savefig(figname)
#     pylab.show()

# def histogramCreate(data, width):
#     Y = np.bincount((data/width).astype(int))
#     xm = int(data.min()/width)
#     xM = int(data.max()/width)+1
#     X = np.arange(width*xm, width*xM, width)
#     return X, Y[xm:]
    
# nLoops1Vals = np.load('nLoops1Vals.npy')
# nLoops2Vals = np.load('nLoops2Vals.npy')
# deltaFVals = np.load('deltaFVals.npy')
# FcVals = np.load('FcVals.npy')
# FtVals = np.load('FtVals.npy')
# phiFVals = np.load('phiFVals.npy')
# TintVals = np.load('TintVals.npy')
# TABVals = np.load('TABVals.npy')
# t2Vals = np.load('t2Vals.npy')
# lcVals = np.load('lcVals.npy')
# C0FVals = np.load('C0FVals.npy')

# print('nLoops1Vals ', nLoops1Vals.min(), ' ', nLoops1Vals.max(), ' ', nLoops1Vals.mean(), ' ', (nLoops1Vals.max()-nLoops1Vals.min())/nLoops1Vals.mean())
# print('nLoops2Vals ', nLoops2Vals.min(), ' ', nLoops2Vals.max(), ' ', nLoops2Vals.mean(), ' ', (nLoops2Vals.max()-nLoops2Vals.min())/nLoops2Vals.mean())
# print('deltaFVals ', deltaFVals.min(), ' ', deltaFVals.max(), ' ', deltaFVals.mean(), ' ', (deltaFVals.max()-deltaFVals.min())/deltaFVals.mean())
# print('FcVals ', FcVals.min(), ' ', FcVals.max(), ' ', FcVals.mean(), ' ', (FcVals.max()-FcVals.min())/FcVals.mean())
# print('FtVals ', FtVals.min(), ' ', FtVals.max(), ' ', FtVals.mean(), ' ', (FtVals.max()-FtVals.min())/FtVals.mean())
# print('phiFVals ', phiFVals.min(), ' ', phiFVals.max(), ' ', phiFVals.mean(), ' ', (phiFVals.max()-phiFVals.min())/phiFVals.mean())
# print('TintVals ', TintVals.min(), ' ', TintVals.max(), ' ', TintVals.mean(), ' ', (TintVals.max()-TintVals.min())/TintVals.mean())
# print('TABVals ', TABVals.min(), ' ', TABVals.max(), ' ', TABVals.mean(), ' ', (TABVals.max()-TABVals.min())/TABVals.mean())
# print('t2Vals ', t2Vals.min(), ' ', t2Vals.max(), ' ', t2Vals.mean(), ' ', (t2Vals.max()-t2Vals.min())/t2Vals.mean())
# print('lcVals ', lcVals.min(), ' ', lcVals.max(), ' ', lcVals.mean(), ' ', (lcVals.max()-lcVals.min())/lcVals.mean())
# print('C0FVals ', C0FVals.min(), ' ', C0FVals.max(), ' ', C0FVals.mean(), ' ', (C0FVals.max()-C0FVals.min())/C0FVals.mean())

# X, Y = histogramCreate(nLoops1Vals, 10)
# plotHisto(X, Y, nLoops1Vals.mean(), 'Histogram of the number of loops to get a solution', 'Number of loops needed to converge', 'Frequency', 'Loops.svg')

# --------------------------------------------------------------------------------------------------
# SECTION 3
# --------------------------------------------------------------------------------------------------
# Sensivity study

# def PlotCurves(y, advances, lab, ylab, tit, nam, lp, bb):
#     pylab.figure(figsize = (11.69, 8.27)) # for a4 landscape 
#     pylab.rc('text', usetex = True)
#     pylab.rcParams['xtick.labelsize'] = 18
#     pylab.rcParams['ytick.labelsize'] = 18
#     ii = 0
#     for advance in advances:
#         pylab.plot (xs[ii], y[ii], label = lab+' ($t_1$ = '+ '%.2f'%round(advances[ii]*1e3, 2)+' mm)', linewidth = 3)
#         ii += 1
#     pylab.xlabel('Cutting speed $V$ (m/min)', fontsize = 22)
#     pylab.ylabel(ylab, fontsize = 22)
#     pylab.legend(loc = lp, bbox_to_anchor = bb, fontsize = 16, frameon = True, 
#                           fancybox = True, shadow = True, ncol = 1, numpoints = 1)
#     #pylab.grid(True)
#     pylab.title(tit, y = 1.04, fontsize = 22)
#     pylab.savefig(nam, transparent = True, bbox_inches = 'tight', pad_inches = 0)
#     pylab.show()   

# def PlotVariations(speeds, advances):
#     global V, t1, xs
#     xs = []
#     ys_Fc = []
#     ys_Ft = [] 
#     ys_TAB  = [] 
#     ys_Tint  = [] 
#     ys_phi  = [] 
#     ys_t2  = [] 
#     ys_lc  = [] 
#     for advance in advances:
#         x = []
#         y_Fc = []
#         y_Ft = [] 
#         y_TAB  = [] 
#         y_Tint  = [] 
#         y_phi  = [] 
#         y_t2  = [] 
#         y_lc  = [] 
#         for speed in speeds:
#             V = speed / 60
#             t1 = advance
#             # Initialisation de l'optimisation
#             ReinitializeParameters(paramsOpt1, paramsOpt2)
#             Optimize(FittingFunction)
#             if (TAB  >  0) :
#                 x.append(speed)
#                 y_Fc.append(Fc)
#                 y_Ft.append(Ft)
#                 y_TAB.append(TAB-T0)
#                 y_Tint.append(Tint-T0)
#                 y_phi.append(phi*180/math.pi)
#                 y_t2.append(1000*t2)
#                 y_lc.append(1000*lc)
#         xs.append(x)
#         ys_Fc.append(y_Fc)
#         ys_Ft.append(y_Ft)
#         ys_TAB.append(y_TAB)
#         ys_Tint.append(y_Tint)
#         ys_phi.append(y_phi)
#         ys_t2.append(y_t2)
#         ys_lc.append(y_lc)
#     PlotCurves(ys_Fc, advances, '$F_C$', 'Force $F_C$ (N)', 'Cutting force $F_C$ v.s. cutting speed $V$', 'Fc_vs_Speed.svg', 'upper right', (1.0, 1.0))
#     PlotCurves(ys_Ft, advances, '$F_T$', 'Force $F_T$ (N)', 'Advancing force $F_T$ v.s. cutting speed $V$', 'Fa_vs_Speed.svg', 'upper right', (1.0, 1.0))
#     PlotCurves(ys_TAB, advances, '$T_{AB}$', 'Temperature $T_{AB}$ ($^{\circ}$C)', 'Temperature $T_{AB}$ v.s. cutting speed $V$', 'TAB_vs_Speed.svg', 'lower right',(1.0, 0.0))
#     PlotCurves(ys_Tint, advances, '$T_{int}$', 'Temperature $T_{int}$ ($^{\circ}$C)', 'Temperature $T_{int}$ v.s. cutting speed $V$', 'Tint_vs_Speed.svg', 'lower right',(1.0, 0.0))
#     PlotCurves(ys_t2, advances, '$t_2$', 'Thickness $t_2$ (mm)', 'Chip thickness $t_2$ v.s. cutting speed $V$', 't2_vs_Speed.svg', 'upper right', (1.0, 1.0))
#     PlotCurves(ys_lc, advances, '$l_c$', 'Length $l_c$ (mm)', 'Contact length $l_c$ v.s. cutting speed $V$', 'lc_vs_Speed.svg', 'upper right', (1.0, 1.0))
#     PlotCurves(ys_phi, advances, '$\phi$', 'Angle $\phi$ ($^{\circ}$)', 'Shear plane angle $\phi$ v.s. cutting speed $V$', 'phi_vs_Speed.svg', 'lower right',(1.0, 0.0))

# speeds = np.linspace(100, 400, 25, True)
# advances =  np.array([0.15e-3, 0.2e-3, 0.25e-3, 0.3e-3, 0.4e-3, 0.5e-3])
# PlotVariations(speeds, advances)
