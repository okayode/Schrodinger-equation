# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:26:36 2021

@author: kayode olumoyin
"""
import numpy as np
import pandas as pd
import csv
import sys
import os
import glob
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import sympy as sym
from sympy import lambdify, pi, sqrt
from tabulate import tabulate
from scipy import linalg
from scipy.integrate import quad
from scipy.optimize import minimize

############################ functions starts here #############################################
    
def norm_1G(alpha,r):
    gtf = sym.exp(-alpha*r**2)
    S_int = sym.integrate(4*sym.pi*(r**2)*(gtf**2),(r,0,sym.oo))
    Na = sym.sqrt(1/S_int)
    return Na

def nKPE(alpha,r):
    Na = norm_1G(alpha,r);
    gtf = sym.exp(-alpha*r**2);
    ngtf = Na*gtf
    KE = (-1/(2*r**2))*sym.diff((r**2)*sym.diff(ngtf,r),r)
    PE = -1*ngtf/r
    TE = KE + PE
    E = sym.integrate(4*sym.pi*(r**2)*ngtf*TE,(r,0,sym.oo))
    return ngtf, KE, PE, E

def STVH(alp1, alp2, r):
    g1 = sym.exp(-alp1*r**2); g2 = sym.exp(-alp2*r**2); g = sym.exp(-(alp1+alp2)*r**2)
    S_alp1 = sym.integrate(4*sym.pi*(r**2)*(g1**2),(r,0,sym.oo))
    S_alp2 = sym.integrate(4*sym.pi*(r**2)*(g2**2),(r,0,sym.oo))
    N_alp1 = sym.sqrt(1/S_alp1); N_alp2 = sym.sqrt(1/S_alp2)
    ss = sym.integrate(4*sym.pi*(r**2)*N_alp1*N_alp2*g,(r,0,sym.oo))
    t = (-1/(2*r**2))*sym.diff((r**2)*sym.diff(N_alp2*g2,r),r)
    tt = sym.integrate(4*sym.pi*(r**2)*N_alp1*g1*t,(r,0,sym.oo))
    v = (-1/r)*(N_alp2*g2)
    vv = sym.integrate(4*sym.pi*(r**2)*N_alp1*g1*v,(r,0,sym.oo))
    h = t + v
    hh = sym.integrate(4*sym.pi*(r**2)*N_alp1*g1*h,(r,0,sym.oo))
    return ss, tt, vv, hh

def VarSchrodinger_exact(rr):
    psi = (1/np.sqrt(np.pi))*np.exp(-rr)
    return psi

def VarSchrodinger_1(r,rr):
    Na_STO_1G = norm_1G(res.x[0],r); Na_np = np.array(Na_STO_1G).astype(np.longdouble)
    phi_1G = Na_np*np.exp(-res.x[0]*rr**2)
    return phi_1G

def VarSchrodinger_n(arr,r,rr):
    n = len(arr)
    S = np.zeros((n,n)); T = np.zeros((n,n)); V = np.zeros((n,n)); H = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            [S[i,j],T[i,j],V[i,j],H[i,j]] = STVH(arr[i],arr[j],r)

    S_np = np.array(S).astype(np.float64); T_np = np.array(T).astype(np.float64)
    V_np = np.array(V).astype(np.float64); H_np = np.array(H).astype(np.float64)

    S_sym = sym.Matrix(S_np)
    P, D = S_sym.diagonalize()

    P_np = np.array(P).astype(np.float64); D_np = np.array(D).astype(np.float64)
    P_inv = linalg.inv(P_np); D_sqrt = np.sqrt(D_np)
    D_inv_sqrt = linalg.inv(D_sqrt); S_inv_sqrt = np.matmul(P_np,np.matmul(D_inv_sqrt,P_inv))
    H_prime = np.matmul(S_inv_sqrt,np.matmul(H_np,S_inv_sqrt))    

    H_prime_sym = sym.Matrix(H_prime)
    PP, DD = H_prime_sym.diagonalize()

    PP_np = np.array(PP).astype(np.float64); DD_np = np.array(DD).astype(np.float64)
    PP_inv = linalg.inv(PP_np); C = np.matmul(S_inv_sqrt,PP_np)    

    for i in range(0,n):
        for j in range(0,n):
            if DD_np[i,j] != 0 and DD_np[i,j] == np.amin(DD_np):
                Cmin = C[:,j]                                               # Array of coefficients

    NN = np.zeros(n); S_count = 0; KE_count = 0; PE_count = 0; TE_count = 0
    for i in range(0,len(Cmin)):
        NN[i] = Cmin[i]*norm_1G(arr[i],r)
    NN_np = np.array(NN).astype(np.float64)
    phi = np.zeros(len(rr))
    for i in range(0,n):
        phi_STO = NN_np[i]*np.exp(-arr[i]*rr**2)
        phi = phi_STO + phi                                                 # phi

    for i in range(0,len(Cmin)):
        for j in range(0,len(Cmin)):
            SS2 = Cmin[i]*Cmin[j]*S_np[i,j]; S_count = SS2 + S_count
            KE = Cmin[i]*Cmin[j]*T_np[i,j]; KE_count = KE + KE_count
            PE = Cmin[i]*Cmin[j]*V_np[i,j]; PE_count = PE + PE_count
            TE = Cmin[i]*Cmin[j]*H_np[i,j]; TE_count = TE + TE_count

    S_norm = S_count                                                        # normalization integral
    KE_norm = KE_count/S_norm                                               # Kinetic energy
    PE_norm = PE_count/S_norm                                               # Potential energy
    TE_norm = TE_count/S_norm                                               # Total energy
    VR_norm = -PE_norm/KE_norm                                              # Virial ratio

    print('\n Matrix, S = \n {}'.format(S_np))
    print('\n Matrix, T = \n {}'.format(T_np))
    print('\n Matrix, V = \n {}'.format(V_np))
    print('\n Matrix, H = \n {}'.format(H_np))
    print('\n Matrix, H_prime = \n {}'.format(H_prime))
    print('\n Matrix, C_prime = \n {}'.format(PP_np))
    print('\n Matrix, energy = \n {}'.format(DD_np))
    print('\n Matrix, C = \n {}'.format(C))
    print('\n Array of coefficients, Cmin = {}'.format(Cmin))
    print(' normalization integral = {}'.format(S_norm))
    print(' Kinetic Energy, STO_{}G = {}'.format(n,KE_norm))
    print(' Potential Energy, STO_{}G = {}'.format(n,PE_norm))
    print(' Virial Ratio, STO_{}G = {}'.format(n,VR_norm))
    print(' Total Energy, STO_{}G = {}'.format(n,TE_norm))
    return phi, S_norm, KE_norm, PE_norm, VR_norm, TE_norm

def plotting_STO_1G(sty,rr):
    mpl.style.use(sty)
    fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(25,15))
    psi = VarSchrodinger_exact(rr)
    func1 = VarSchrodinger_1(r,rr)
    ax.set_title(r'variational solution for hydrogen atom, exact vs STO-1G',fontsize=20)
    ax.plot(rr,psi,'b')
    ax.plot(rr,func1,'r')
    ax.legend((r'exact, $\psi(r)$', r'STO-1G, $\phi(r)$'),fontsize=20)
    ax.set_xlabel('r (a.u)',fontsize=25)
    ax.set_ylabel(r'$\psi(r),\phi(r)$',fontsize=25)
    plt.setp(ax.get_xticklabels(), fontsize=25)
    plt.setp(ax.get_yticklabels(), fontsize=25)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    savefig('./STO_1G')

def plotting_STO_nG(sty,phi,arr,rr):
    mpl.style.use(sty)
    fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(25,15))
    psi = VarSchrodinger_exact(rr)
    func1 = VarSchrodinger_1(r,rr)
    n = len(arr)
    func2 = np.abs(phi)
    ax.set_title(r'variational solution for hydrogen atom, exact vs STO-{}G'.format(n),fontsize=20)
    ax.plot(rr,psi,'b')
    ax.plot(rr,func1,'r')
    ax.plot(rr,func2,'g')
    ax.legend((r'exact, $\psi(r)$', r'STO-1G, $\phi(r)$', r'STO-{}G, $\phi(r)$'.format(n)),fontsize=20)
    ax.set_xlabel('r (a.u)',fontsize=25)
    ax.set_ylabel(r'$\psi(r),\phi(r)$',fontsize=25)
    plt.setp(ax.get_xticklabels(), fontsize=25)
    plt.setp(ax.get_yticklabels(), fontsize=25)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    savefig('./STO_{}G'.format(n))

def savefig(filename,crop=True):
    plt.savefig('{}.pdf'.format(filename))
    plt.savefig('{}.jpg'.format(filename))

############################# functions ends here ##############################################
start = time.time()    
sys.stdout = open("proj5_output.txt","w")                               # write all output to file

print('#######################################################################################')
print('#         Variational solution of the Schrodinger equation for Hydrogen atom          #')
print('#######################################################################################')
print('#                        Kayode Olumoyin                                              #')
print('#                                                                                     #')
print('#                                                                                     #')
print('#######################################################################################')

r = sym.Symbol('r'); alpha = sym.Symbol('alpha', positive=True)         # set (alpha > 0)
np.set_printoptions(suppress=True)                                      # np.set_printoptions(precision=3)
    
# load datafiles
path = os.getcwd()
STO_files = glob.glob(os.path.join(path, "*G.txt"))                     # list of all the files in our path

LL = len(STO_files)
print('\n The number of STO-nG files detected = {}'.format(LL))
print('\n The files are: ')
for f in STO_files:
    df = pd.read_csv(f)                 # read the csv file
    df.columns = ['col']
    print('File Name:', f.split("\\")[-1])

ra = np.linspace(0,5,1000)

print('\n#####################################################################################')
print(' Exact ')

sym_psi = (1/sym.sqrt(sym.pi))*sym.exp(-r)                              # symbolic psi, exact
T_psi = (-1/(2*r**2))*sym.diff((r**2)*sym.diff(sym_psi,r),r)
V_psi = -1*sym_psi/r
S_exact = sym.integrate(4*sym.pi*(r**2)*(sym_psi**2),(r,0,sym.oo))      # normalization integral
fun_S_exact = lambdify(alpha, S_exact, 'numpy')
SE_exact = fun_S_exact(0)                                               # since SE_exact is alpha constant function
T_exact = sym.integrate(4*sym.pi*(r**2)*sym_psi*T_psi,(r,0,sym.oo))     # Kinetic Energy, exact
fun_T_exact = lambdify(alpha, T_exact, 'numpy')
KE_exact = fun_T_exact(0)                                               # since KE_exact is alpha constant function
V_exact = sym.integrate(4*sym.pi*(r**2)*sym_psi*V_psi,(r,0,sym.oo))     # Potential Energy, exact
fun_V_exact = lambdify(alpha, V_exact, 'numpy')
PE_exact = fun_V_exact(0)                                               # since PE_exact is alpha constant function
VR_exact = -PE_exact/KE_exact                                           # virial ratio, exact
TE_exact = KE_exact+PE_exact                                            # Total Energy, exact

print(' normalization integral = {}'.format(SE_exact))
print(' Kinetic Energy, exact = {}'.format(KE_exact))
print(' Potential Energy, exact = {}'.format(PE_exact))
print(' Virial Ratio, exact = {}'.format(VR_exact))
print(' Total Energy, exact = {}'.format(TE_exact))

print('\n#####################################################################################')
print(' STO-1G ')

[ngtf_1, KE_1, PE_1, Energy_1] = nKPE(alpha,r)
fun = lambdify(alpha, Energy_1, 'numpy')
res = minimize(fun, x0=1.0,method='Nelder-Mead', tol=1e-6)
Emin = res.fun                                                          # obtain optimimal alpha that minimizes E
SE_1G = sym.integrate(4*sym.pi*(r**2)*(ngtf_1**2),(r,0,sym.oo))
fun_SE_1G = lambdify(alpha, SE_1G, 'numpy')
SE_1G = fun_SE_1G(res.x[0])                                             # normalization integral
KE_1G = sym.integrate(4*sym.pi*(r**2)*ngtf_1*KE_1,(r,0,sym.oo))
fun_KE_1G = lambdify(alpha, KE_1G, 'numpy')
KE_1G = fun_KE_1G(res.x[0])                                             # Kinetic Energy, STO_1G
PE_1G = sym.integrate(4*sym.pi*(r**2)*ngtf_1*PE_1,(r,0,sym.oo))
fun_PE_1G = lambdify(alpha, PE_1G, 'numpy')
PE_1G = fun_PE_1G(res.x[0])                                             # Potential Energy, STO_1G
TE_1G = KE_1G+PE_1G                                                     # Total Energy, STO_1G
VR_1G = -PE_1G/KE_1G                                                    # Virial Ratio, STO_1G

print(' alpha, STO_1G = {}'.format(res.x[0]))
print(' Emin, STO_1G = {}'.format(res.fun))
print(' normalization integral = {}'.format(SE_1G))
print(' Kinetic Energy, STO_1G = {}'.format(KE_1G))
print(' Potential Energy, STO_1G = {}'.format(PE_1G))
print(' Total Energy, STO_1G = {}'.format(TE_1G))
print(' Virial Ratio, STO_1G = {}'.format(VR_1G))

plotting_STO_1G('seaborn',ra)                                           # plotting Exact and STO-1G

print('\n#####################################################################################')

# plotting Exact and STO-nG
SG=[]; KEG=[]; PEG=[]; VRG=[]; TEG=[]; STO=[]; all_phi=[]
SG.append(SE_1G); KEG.append(KE_1G); PEG.append(PE_1G)
VRG.append(VR_1G); TEG.append(TE_1G); STO.append('STO_1G')
for i in range(0,LL):
    lines=[]
    nlines=[]
    with open(STO_files[i]) as g0:
        for line in g0:
            lines.append(line.split())
    nlines = lines[2:]
    g1 = np.array(nlines)
    g2 = g1[:,0]
    g3 = np.char.replace(g2,'D','E')      
    arr = g3.astype(np.float64)
    nn = len(arr)
    print(' STO_{}G '.format(nn))
    [phi, S_G, KE_G, PE_G, VR_G, TE_G] = VarSchrodinger_n(arr,r,ra)
    plotting_STO_nG('seaborn',phi,arr,ra)
    all_phi.append(phi); SG.append(S_G); KEG.append(KE_G); PEG.append(PE_G)
    VRG.append(VR_G); TEG.append(TE_G); STO.append('STO_{}G'.format(nn))
    print('\n#####################################################################################')
g0.close()

SG.append(SE_exact); KEG.append(KE_exact); PEG.append(PE_exact)
VRG.append(VR_exact); TEG.append(TE_exact); STO.append('STO_exact')

# Plotting all STO-nG,
fig, ax1 = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(25,15))
ax1.set_title(r'variational solution for hydrogen atom, exact vs STO-nG, $n \in \{1,2,3,4,5,6,12,20\}$',fontsize=20)
psi = VarSchrodinger_exact(ra)
phi_1G = VarSchrodinger_1(r,ra)
ax1.plot(ra,psi,'b')
ax1.plot(ra,phi_1G,'r')
for i in range(0,LL):
    ax1.plot(ra,np.abs(all_phi[i]))
ax1.set_xlabel('r (a.u)',fontsize=25)
ax1.set_ylabel(r'$\psi(r),\phi(r)$',fontsize=25)
plt.setp(ax1.get_xticklabels(), fontsize=25)
plt.setp(ax1.get_yticklabels(), fontsize=25)
ax1.set_xlim(xmin=0); ax1.set_ylim(ymin=0)
savefig('./STO_nG')

# tabulate
print('\n')
headers=["STO-File","Normalization", " KE (a.u.) ", " PE (a.u.) ", " TE (a.u.) ", "-V/T "]
table = zip(STO,SG,KEG,PEG,TEG,VRG)
print(tabulate(table, headers=headers, floatfmt=".15f"))

end = time.time()
print("Elapsed time = {} secs ".format(end - start))
sys.stdout.close()
