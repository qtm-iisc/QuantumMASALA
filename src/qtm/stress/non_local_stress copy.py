import numpy as np
from qtm.crystal import Crystal
from qtm.pseudo.nloc import NonlocGenerator
from qtm.constants import RYDBERG_HART, FPI
from qtm.config import NDArray


TOL=1e-7

def stress_nonloc(wavefun:tuple,
                   cryst: Crystal,
                   ) -> NDArray:
    ##Getting the characteristics of the crystal
    l_atoms = cryst.l_atoms
    tot_atom = np.sum([sp.numatoms for sp in l_atoms])
    labels=np.repeat(np.arange(len(l_atoms)), [sp.numatoms for sp in l_atoms])
    atom_label= np.concatenate([np.arange(sp.numatoms) for sp in l_atoms])

    ##Initializing the stress tensor
    stress_nl=np.zeros((3,3)).astype(np.complex128)

    diag_stress_entry=0
    stress_dy_total=np.zeros((3,3)).astype(np.complex128) 
    stress_dj_total=np.zeros((3,3)).astype(np.complex128)

    ##Looping over the wavefunctions
    for inum in range(tot_atom):
        sp=l_atoms[labels[inum]]
        atom_label_sp=atom_label[inum]
        
        counter=0
        for wfn in wavefun:
            for k_wfn in wfn:
                ## Getting the evc and gk space characteristics form the wavefunction
                evc=k_wfn.evc_gk
                nbnd=evc.shape[0]
                evc_data=evc.data.T
                gkspace=k_wfn.gkspc
                k_weight=k_wfn.k_weight
                gkcart=gkspace.gk_cart.T
                gknorm=gkspace.gk_norm
                k_cryst=k_wfn.k_cryst
                k_tpiba=cryst.recilat.cryst2tpiba(k_cryst)
                beta_fac = FPI / np.sqrt(cryst.reallat.cellvol)
                ## Getting the non-local beta projectors and dij matrices from the wavefun
                k_nonloc= NonlocGenerator(sp=sp,
                                            gwfn=gkspace.gwfn)
                numvkb=k_nonloc.numvkb
                vkb_full, djvkb_full, dyvkb_full, dij, vkb_diag = k_nonloc.gen_vkb_dij(k_wfn.gkspc)
                row_vkb=int(vkb_full.data.shape[0]/sp.numatoms)
                vkb=vkb_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                dj_vkb=djvkb_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                dy_vkbx_full, dy_vkby_full, dy_vkbz_full=dyvkb_full
                dy_vkbx=dy_vkbx_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                dy_vkby=dy_vkby_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                dy_vkbz=dy_vkbz_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                dy_vkb=np.array([dy_vkbx, dy_vkby, dy_vkbz])
                dij_sp=dij[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb, atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]/RYDBERG_HART


                ## Calculation of the Diagonal Terms
                betaPsi=np.conj(vkb)@evc_data
                abs2_betaPsi=np.abs(betaPsi)**2
                quant=np.sum(dij_sp@abs2_betaPsi)
                quant*=k_weight
                diag_stress_entry+=quant
                diag_stress=(np.eye(3)*quant).astype(np.complex128)
                print("the stress from diagonal contribution is\n", diag_stress_entry)

                betaPsi_d=dij_sp@betaPsi ## constrcution of the ps matrix 
                #shape is numvkb*nbnd

                '''for ibnd in range(nbnd):
                    for ikb in range(numvkb):
                        for igk, gk in enumerate(gkcart):
                            worksum=betaPsi_d[ikb, ibnd]*dj_vkb[ikb, igk]
                            cv=evc_data[igk, ibnd]*np.outer(gk, gk)
                            if gknorm[igk]>TOL:
                                cv/=gknorm[igk]
                            else:
                                cv*=0
                            stress_nl+=2*np.real(np.conj(cv)*worksum)*k_weight
                            print(stress_nl)
                stress_nl+=diag_stress
                print(diag_stress_entry)'''
                ##Printing the djvkb matrix with counters for k points
                with open('dj_vkb_matrices.txt', 'a') as f:
                        f.write("the counter " + str(counter) + "\n")
                        f.write("the coordinates of k tpiba " + str(k_tpiba) + "\n")
                        f.write("the djvkb matrix is given by " + str(dj_vkb) + "\n")
                

                
                ##The derivative of Spherical Bessel function                
                with open('ps.txt', 'a') as f: 
                    f.write("-----------------------------------------------")
                    f.write(f"the counter {counter}\n")
                    f.write(f"the coordinates of k tpiba, {k_tpiba}\n")
                    f.write(f"absolute value of ps matrix  {np.abs(betaPsi_d).T}\n")
                
                beta_dj=dj_vkb.T@betaPsi_d   #(shape is G,numbnd)
                Sigma_j_nl=2*np.real(np.conj(evc_data)*beta_dj)
                Sigma_j_nl=np.sum(Sigma_j_nl, axis=1)
                ##writting Sigma_j_nl in a appendable file
                with open('write_sigma_j_nl.txt', 'a') as f:
                    f.write("-----------------------------------------------\n")
                    f.write(f"the counter {counter}\n")
                    f.write(f"the coordinates of k tpiba, {k_tpiba}\n")
                    f.write(f"the Sigma_j_nl matrix is given by {Sigma_j_nl}\n")
                #Sigma_j_nl_diag=np.diag(Sigma_j_nl)
                for igk, gk in enumerate(gkcart):
                    with open('write_nl.txt', 'a') as f:
                        f.write("-----------------------------------------------\n")
                        f.write("the counter on k is " + str(counter) + "\n")
                        f.write("the coordinates of k tpiba" + str(k_tpiba) + "\n")
                        f.write(f"the counter on atoms is {inum}\n")
                        f.write("the g counter is" + str(igk) + "\n")
                        f.write("gk: " + str(gk) + "\n")
                        f.write("gknorm: " + str(1/gknorm[igk]) + "\n")
                        #f.write("stress_nl: " + str(stress_nl) + "\n")
                    gk_tensor=np.outer(gk, gk)
                    with open("write_gk_tensor.txt", "a") as f:
                        f.write("-----------------------------------------------\n")
                        f.write("the counter on k is " + str(counter) + "\n")
                        f.write("the coordinates of k tpiba" + str(k_tpiba) + "\n")
                        f.write(f"the counter on atoms is {inum}\n")
                        f.write("the g counter is" + str(igk) + "\n")
                        f.write("gk: " + str(gk) + "\n")
                        f.write("gknorm: " + str(1/gknorm[igk]) + "\n")
                        f.write("gk_tensor: " + str(gk_tensor) + "\n")
                    if gknorm[igk]>TOL:
                        gk_tensor*=Sigma_j_nl[igk]/gknorm[igk]
                    else:
                        gk_tensor*=0.0
                    stress_nl+=k_weight*gk_tensor
                print("stress_nl after each iteration of k points", stress_nl, diag_stress)
                #print("stress_nl from the derivative of Bessel function\n", stress_dj_total)
                
                
                #### The derivative of the Spherical Harmonics
                #stress_nl+= diag_stress'''
                
                ##The derivative of spherical Harmnomics
                beta_dy=np.einsum("ijl, jk->ikl", dy_vkb, betaPsi_d)
                Sigma_y_nl=np.einsum("ijl, jk->ikl", beta_dy, np.conj(evc_data.T))
                #print(Sigma_y_nl.shape)
                Sigma_y_diagonal = np.array([np.diagonal(Sigma_y_nl[i]) for i in range(Sigma_y_nl.shape[0])])
                stress_dy=2*np.real(Sigma_y_diagonal@gkcart)
                stress_dy[0,1]=stress_dy[1,0]
                stress_dy[0,2]=stress_dy[2,0]
                stress_dy[1,2]=stress_dy[2,1]
                stress_dy_total+=stress_dy*k_weight

                stress_nl+=stress_dy*k_weight

                #print("the stress from the derivative of the spherical harmonics are\n", stress_dy_total)
                counter+=1


    return stress_nl
        

    