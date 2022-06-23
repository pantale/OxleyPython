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
import numpy

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

# Défintion des valeurs de boucles
deltamin = 0.005
deltamax = 0.2
deltainc = 0.005

C0min = 2
C0max = 10
C0inc = 0.1

phimin = 5
phimax = 45
phiinc = 0.1

FcMin = float("inf")

# Generates the 40*81*401 solutions
deltas = numpy.linspace(deltamin, deltamax, int(1+(deltamax-deltamin)/deltainc), True)
C0s = numpy.linspace(C0min, C0max, int((C0max-C0min)/C0inc+1), True)
phis = numpy.linspace(phimin*math.pi/180, phimax*math.pi/180, int(1+(phimax-phimin)/phiinc), True)

print("%g x %g x %g" %(deltas.shape[0],C0s.shape[0],phis.shape[0]))

# BOUCLE 1 delta
for delta in deltas:
    print(round(100*(delta-deltamin)/(deltamax+deltainc-deltamin),1)," % done (",delta,")")
    C0Min = -1
    ecartC0Min = float("inf")
    
    # BOUCLE2 C0
    for C0 in C0s:
        phiMin0 = -1
        ecartPhiMin = float("inf")
        
        # BOUCLE 3 phi
        for phi in phis: 
            Toint, kchip, sigmaN, sigmaNmax = Compute_Toint_Kchip()
            if (Toint!=0) :
                # Test solution optimale 
                ecart = abs(Toint - kchip)
                if (ecart < ecartPhiMin):
                    phiMin0 = phi
                    ecartPhiMin = ecart
        
        # Recalcul de lAB'optimal 
        phi = phiMin0
        Toint, kchip, sigmaN, sigmaNmax = Compute_Toint_Kchip()
        
        # Test solution optimale 
        ecart = abs(sigmaN - sigmaNmax)
        if (ecart < ecartC0Min) :
            C0Min = C0
            phiMin1 = phi
            ecartC0Min = ecart
    
    # Transfert des optimaux
    C0 = C0Min
    phi = phiMin1
    
    # Recalcul de lAB'optimal 
    Toint, kchip, sigmaN, sigmaNmax = Compute_Toint_Kchip()
    #--------------------------------------------------------------------------------------------------
    
    print("C0=",C0," phi=",phi*180/math.pi," delta=",delta," Fc=",Fc)
    
    # Test solution optimale 
    if (Fc <= FcMin) :
        print("New Optimal Solution (gain is ", FcMin - Fc, ")")
        FcMin = Fc
        C0Optimal = C0
        phiOptimal = phi
        deltaOptimal = delta
    else :    
        print("Keeping Old solution (gain is ", FcMin - Fc, ")")

# Affichage avancement 
print("100 % done")

# Transfert des résultats 
delta = deltaOptimal
C0 = C0Optimal
phi = phiOptimal

# Recalcul de lAB'optimal 
Toint, kchip, sigmaN, sigmaNmax = Compute_Toint_Kchip()

print("C0 : ", round(C0, 3))
print("delta : ", round(delta, 3))

print("critere 1 : ",round(abs(Toint - kchip),1))
print("critere 2 : ",round(abs(sigmaNmax - sigmaN),1))

print("Internal error : ", round(math.sqrt((Toint - kchip)**2 + (sigmaNmax - sigmaN)**2), 10), "Pa")

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

